import { useRef, useEffect, useCallback, useState, useMemo } from 'react';
import type { TrainingEvent, TensorSnapshot } from '../hooks/useStream';

interface LayerDef {
  shortName: string;
  type:      'input' | 'conv' | 'pool' | 'linear' | 'output';
  units:     number;
  layer_key: string;
  filters?:  number;
  filterShape?: [number, number]; 
}

const TOPOLOGY: LayerDef[] = [
  { shortName: 'input',  type: 'input',  units: 5,  layer_key: '',            },
  { shortName: 'conv1',  type: 'conv',   units: 6,  layer_key: 'features.0',  filters: 32, filterShape: [3,3] },
  { shortName: 'pool1',  type: 'pool',   units: 4,  layer_key: '',            },
  { shortName: 'conv2',  type: 'conv',   units: 6,  layer_key: 'features.3',  filters: 64, filterShape: [3,3] },
  { shortName: 'pool2',  type: 'pool',   units: 4,  layer_key: '',            },
  { shortName: 'fc1',    type: 'linear', units: 5,  layer_key: 'classifier.1' },
  { shortName: 'output', type: 'output', units: 3,  layer_key: 'classifier.4' },
];

const LAYER_COLORS: Record<LayerDef['type'], string> = {
  input:'#888780', conv:'#534AB7', pool:'#3C3489', linear:'#0F6E56', output:'#085041',
};
const CONF_COLORS = ['#534AB7','#1D9E75','#D85A30'];

interface Node { id:string; li:number; ni:number; x:number; y:number; layer:LayerDef; }

function buildLayout(W:number, H:number): Node[] {
  return TOPOLOGY.flatMap((layer, li) =>
    Array.from({length:layer.units}, (_,ni) => ({
      id:`${li}-${ni}`, li, ni,
      x: W/(TOPOLOGY.length+1)*(li+1),
      y: H/(layer.units+1)*(ni+1),
      layer,
    }))
  );
}

const clamp = (v:number,lo:number,hi:number) => Math.max(lo,Math.min(hi,v));

function extractEdgeWeights(snap:TensorSnapshot, fromUnits:number, toUnits:number): number[][]|null {
  if (!snap.values.length || !snap.shape.length) return null;
  const [outFull,inFull,...spatial] = snap.shape;
  const kernelSize = spatial.reduce((a,b)=>a*b,1)||1;
  const outStep = Math.max(1,Math.floor(outFull/toUnits));
  const inStep  = Math.max(1,Math.floor(inFull/fromUnits));
  const matrix:number[][] = [];
  for (let fi=0; fi<fromUnits; fi++) {
    const row:number[] = [];
    for (let ti=0; ti<toUnits; ti++) {
      const outIdx = Math.min(ti*outStep, outFull-1);
      const inIdx  = Math.min(fi*inStep,  inFull-1);
      let sum=0;
      for (let k=0; k<kernelSize; k++) {
        const idx = outIdx*inFull*kernelSize + inIdx*kernelSize + k;
        sum += snap.values[Math.min(idx, snap.values.length-1)]??0;
      }
      row.push(sum/kernelSize);
    }
    matrix.push(row);
  }
  return matrix;
}

function extractActivations(snap:TensorSnapshot, units:number): number[]|null {
  if (!snap.values.length) return null;
  const [,channels,...spatial] = snap.shape;
  const spatialSize = spatial.reduce((a,b)=>a*b,1)||1;
  const channelsAvail = channels??snap.values.length;
  const step = Math.max(1,Math.floor(channelsAvail/units));
  return Array.from({length:units},(_,i)=>{
    const chIdx = Math.min(i*step, channelsAvail-1);
    let sum=0,count=0;
    for (let s=0; s<Math.min(spatialSize,8); s++) {
      const idx = chIdx*spatialSize+s;
      if (idx<snap.values.length) { sum+=Math.abs(snap.values[idx]); count++; }
    }
    return count>0 ? clamp(sum/count,0,1) : 0.3;
  });
}

function extractFilterGrid(snap:TensorSnapshot): Float32Array[]|null {
  if (!snap.values.length || snap.shape.length < 4) return null;
  const [outFilters, inCh, kH, kW] = snap.shape;
  const kernelSize = kH*kW;
  const result:Float32Array[] = [];
  for (let f=0; f<Math.min(outFilters,64); f++) {
    const kernel = new Float32Array(kernelSize);
    for (let k=0; k<kernelSize; k++) {
      let sum=0;
      for (let c=0; c<inCh; c++) {
        const idx = f*inCh*kernelSize + c*kernelSize + k;
        sum += snap.values[idx]??0;
      }
      kernel[k] = sum/inCh;
    }
    result.push(kernel);
  }
  return result.length>0 ? result : null;
}

function FilterGrid({ filters, kH, kW, layerName }: {
  filters: Float32Array[]; kH: number; kW: number; layerName: string;
}) {
  const canvasRefs = useRef<(HTMLCanvasElement|null)[]>([]);

  useEffect(() => {
    filters.forEach((kernel, fi) => {
      const canvas = canvasRefs.current[fi];
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      let mn=Infinity, mx=-Infinity;
      kernel.forEach(v => { if(v<mn)mn=v; if(v+0>mx)mx=v; });
      const range = Math.max(mx-mn, 0.001);

      const CELL = 8; 
      canvas.width  = kW * CELL;
      canvas.height = kH * CELL;
      const img = ctx.createImageData(kW*CELL, kH*CELL);

      for (let y=0; y<kH; y++) {
        for (let x=0; x<kW; x++) {
          const v = kernel[y*kW+x];
          const t = (v - mn) / range; 
          let r,g,b;
          if (t >= 0.5) {
            const s = (t-0.5)*2;
            r = Math.round(29  + (29  - 29 )*s);   
            g = Math.round(40  + (158 - 40 )*s);   
            b = Math.round(42  + (117 - 42 )*s);   
          } else {
            const s = t*2;
            r = Math.round(216 + (42  - 216)*s);   
            g = Math.round(90  + (42  - 90 )*s);   
            b = Math.round(48  + (42  - 48 )*s);   
          }
          for (let py=0; py<CELL; py++) {
            for (let px=0; px<CELL; px++) {
              const base = ((y*CELL+py)*kW*CELL + (x*CELL+px))*4;
              img.data[base]   = r;
              img.data[base+1] = g;
              img.data[base+2] = b;
              img.data[base+3] = 255;
            }
          }
        }
      }
      ctx.putImageData(img,0,0);
    });
  }, [filters, kH, kW]);

  const perRow = 8;

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'6px' }}>
      <div style={{ fontSize:'10px', color:'var(--color-text-tertiary)', letterSpacing:'.05em' }}>
        {layerName.toUpperCase()} FILTERS ({filters.length})
      </div>
      <div style={{ display:'grid', gridTemplateColumns:`repeat(${perRow},auto)`, gap:'2px' }}>
        {filters.map((_, fi) => (
          <canvas
            key={fi}
            ref={el => { canvasRefs.current[fi] = el; }}
            title={`filter ${fi}`}
            style={{
              width: `${kW*4}px`, height: `${kH*4}px`,
              imageRendering:'pixelated', border:'0.5px solid var(--color-border-tertiary)',
              borderRadius:'1px', cursor:'pointer',
            }}
          />
        ))}
      </div>
    </div>
  );
}

const WAVE_STEP_MS = 110;  

interface Props {
  events:    TrainingEvent[];
  snapshots: TensorSnapshot[];
  width?:    number;
  height?:   number;
}

export function NetworkGraph({
  events, snapshots,
  width = 620, height = 300,
}: Props) {
  const canvasRef       = useRef<HTMLCanvasElement>(null);
  const weightsRef      = useRef<(number[][]|null)[]>([]);
  const activationsRef  = useRef<number[][]>([]);
  const probsRef        = useRef<number[]>([0.33,0.33,0.34]);
  const rafRef          = useRef(0);
  const isRealWeights   = useRef(false);
  const lastStepRef     = useRef(-1);
  const waveTsRef       = useRef(-1);    

  const [selected,   setSelected]   = useState<string|null>(null);
  const [hovered,    setHovered]    = useState<string|null>(null);
  const [paused,     setPaused]     = useState(false);
  const [dataSource, setDataSource] = useState<'simulated'|'real'>('simulated');
  const [neuronInfo, setNeuronInfo] = useState('click a neuron to inspect');

  const [filterGrids, setFilterGrids] = useState<Record<string, {
    filters: Float32Array[]; kH: number; kW: number;
  }>>({});
  const [activeFilterLayer, setActiveFilterLayer] = useState<string|null>(null);

  const nodes = useMemo(() => buildLayout(width, height), [width, height]);

  useEffect(() => {
    activationsRef.current = TOPOLOGY.map(l => Array(l.units).fill(0.3));
    weightsRef.current = TOPOLOGY.slice(0,-1).map((layer,i) => {
      const from=layer.units, to=TOPOLOGY[i+1].units;
      return Array.from({length:from},()=>
        Array.from({length:to},()=>(Math.random()-.5)*1.3));
    });
  }, []);

  useEffect(() => {
    const ev = events.at(-1);
    if (!ev) return;

    if (ev.step !== lastStepRef.current) {
      lastStepRef.current = ev.step;
      const now = performance.now();
      const waveDuration = WAVE_STEP_MS*TOPOLOGY.length;
      if (waveTsRef.current === -1||now - waveTsRef.current >= waveDuration) {
        waveTsRef.current = now;
      }
    }

    const progress = clamp(ev.accuracy, 0, 1);
    TOPOLOGY.forEach((layer,li) => {
      if (!isRealWeights.current) {
        activationsRef.current[li] = Array.from({length:layer.units},(_,ni)=>
          clamp(progress*(0.3+0.7*Math.sin((ni+li)*1.1))+(Math.random()-.5)*0.07,0.05,1));
      }
      if (!isRealWeights.current && weightsRef.current[li]) {
        weightsRef.current[li]!.forEach(row=>row.forEach((_,j)=>{
          row[j]+=(Math.random()-.5)*ev.loss*0.012;
          row[j]=clamp(row[j],-3.2,3.2);
        }));
      }
    });

    if (!isRealWeights.current) {
      const out  = activationsRef.current.at(-1)??[.33,.33,.34];
      const exps = out.map(v=>Math.exp(v*2.2));
      const sum  = exps.reduce((a,b)=>a+b,0);
      probsRef.current = exps.map(v=>v/sum);
    }
  }, [events]);

  useEffect(() => {
    if (!snapshots.length) return;
    let gotReal = false;

    snapshots.forEach(snap => {
      const li = TOPOLOGY.findIndex(l=>l.layer_key && snap.layer_name.startsWith(l.layer_key));
      if (li<0) return;

      if (snap.tensor_type === 'WEIGHT') {
        const gapIdx = li-1;
        if (gapIdx>=0) {
          const matrix = extractEdgeWeights(snap, TOPOLOGY[gapIdx].units, TOPOLOGY[li].units);
          if (matrix) { weightsRef.current[gapIdx]=matrix; gotReal=true; }
        }

        const layerDef = TOPOLOGY[li];
        if (layerDef.type==='conv' && layerDef.filterShape) {
          const grids = extractFilterGrid(snap);
          if (grids) {
            const [kH,kW] = layerDef.filterShape;
            setFilterGrids(prev => ({
              ...prev,
              [layerDef.shortName]: { filters:grids, kH, kW }
            }));
            gotReal=true;
          }
        }
      }

      if (snap.tensor_type === 'ACTIVATION') {
        const acts = extractActivations(snap, TOPOLOGY[li].units);
        if (acts) {
          activationsRef.current[li]=acts;
          if (TOPOLOGY[li].type==='output') {
            const exps=acts.map(v=>Math.exp(v*2.2));
            const sum=exps.reduce((a,b)=>a+b,.001);
            probsRef.current=exps.map(v=>v/sum);
          }
          gotReal=true;
        }
      }
    });

    if (gotReal) { isRealWeights.current=true; setDataSource('real'); }
  }, [snapshots]);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    if (rect.width === 0 || rect.height === 0) return;

    const targetW = rect.width * dpr;
    const targetH = rect.height * dpr;

    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.scale(dpr * (rect.width / width), dpr * (rect.height / height));

    const W = weightsRef.current, A = activationsRef.current;
    const now = performance.now();
    const waveAge = waveTsRef.current >= 0 ? now - waveTsRef.current : -1;
    const waveLayer = waveAge >= 0 && waveAge < WAVE_STEP_MS * TOPOLOGY.length
      ? Math.floor(waveAge / WAVE_STEP_MS)
      : -1;
    const waveLayerPrev = waveLayer > 0 ? waveLayer - 1 : -1;

    for (let li = 0; li < TOPOLOGY.length - 1; li++) {
      const froms = nodes.filter(n => n.li === li);
      const tos = nodes.filter(n => n.li === li + 1);
      
      froms.forEach(f => {
        tos.forEach(t => {
          const w = W[li]?.[f.ni]?.[t.ni] ?? 0;
          const active = selected === f.id || selected === t.id || hovered === f.id || hovered === t.id;
          const waveBoost = (li === waveLayer) ? 0.35 : (li === waveLayerPrev ? 0.15 : 0);
          const aw = Math.abs(w);
          const op = clamp(aw * 0.38 + (active ? 0.2 : 0) + waveBoost, 0.04, 0.85);
          const sw = clamp(aw * 1.7 * (active ? 1.5 : 1), 0.3, active ? 4.5 : 3.5);
          const col = w >= 0 ? '#1D9E75' : '#D85A30';

          ctx.beginPath();
          ctx.moveTo(f.x, f.y);
          ctx.lineTo(t.x, t.y);
          ctx.strokeStyle = col;
          ctx.globalAlpha = op;
          ctx.lineWidth = sw;
          ctx.lineCap = 'round';
          ctx.stroke();
        });
      });
    }

    ctx.globalAlpha = 1.0;

    nodes.forEach(n => {
      const act = A[n.li]?.[n.ni] ?? 0.3;
      const isOut = n.li === TOPOLOGY.length - 1;
      const conf = isOut ? (probsRef.current[n.ni] ?? 0) * 0.3 : 0;
      const isWaveLit = n.li === waveLayer;
      const isWavePrev = n.li === waveLayerPrev;
      const waveActBoost = isWaveLit ? 0.4 : isWavePrev ? 0.15 : 0;
      const fop = clamp(0.15 + act * 0.55 + conf + waveActBoost, 0.1, 0.97);
      const isSel = selected === n.id, isHov = hovered === n.id;
      const r = isSel ? 12 : isHov ? 11 : isWaveLit ? 11 : 9;
      const sw = isSel ? 2.5 : isHov ? 1.8 : isWaveLit ? 1.5 : 0.7;
      const stroke = isSel ? '#EF9F27' : isHov ? '#FAC775' : isWaveLit ? '#EF9F27' : LAYER_COLORS[n.layer.type];
      const col = LAYER_COLORS[n.layer.type];

      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fillStyle = col;
      ctx.globalAlpha = fop;
      ctx.fill();

      ctx.globalAlpha = 1.0;
      ctx.strokeStyle = stroke;
      ctx.lineWidth = sw;
      ctx.stroke();

      if (n.layer.type === 'conv' && n.ni === 0 && filterGrids[n.layer.shortName]) {
        ctx.fillStyle = '#EF9F27';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('filters', n.x, n.y + 24);
      }

      if (isOut) {
        ctx.fillStyle = '#888888';
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(String(n.ni), n.x + 15, n.y + 4);
      }
    });

    const seen = new Set<number>();
    ctx.fillStyle = '#888888';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    nodes.forEach(n => {
      if (seen.has(n.li)) return; seen.add(n.li);
      ctx.fillText(n.layer.shortName, n.x, 12);
    });

  }, [nodes, selected, hovered, filterGrids, width, height]);

  const drawRef = useRef(drawCanvas);
  useEffect(() => { drawRef.current = drawCanvas; }, [drawCanvas]);

  useEffect(()=>{
    let active = true; 
    const tick = () => {
      if (!active) return;
      rafRef.current = requestAnimationFrame(tick);
      if (paused) return;
      drawRef.current();
    };
    rafRef.current = requestAnimationFrame(tick);
    return ()=>{
      active = false;
      cancelAnimationFrame(rafRef.current);
    }
  }, [paused]);

  const handleCanvasInteraction = useCallback((e: React.MouseEvent<HTMLCanvasElement>, isClick: boolean) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (width / rect.width);
    const my = (e.clientY - rect.top) * (height / rect.height);

    if (isClick) {
      let clickedFilter: string | null = null;
      nodes.forEach(n => {
        if (n.layer.type === 'conv' && n.ni === 0 && filterGrids[n.layer.shortName]) {
          if (mx > n.x - 20 && mx < n.x + 20 && my > n.y + 15 && my < n.y + 35) {
            clickedFilter = n.layer.shortName;
          }
        }
      });
      if (clickedFilter) {
        setActiveFilterLayer(f => f === clickedFilter ? null : clickedFilter);
        return;
      }
    }

    let bestId: string | null = null;
    let bestDist = 20 * 20; 
    nodes.forEach(n => {
      const dx = n.x - mx, dy = n.y - my;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestDist) {
        bestDist = d2;
        bestId = n.id;
      }
    });

    if (isClick) {
      if (bestId) {
        setSelected(s => {
          const next = s === bestId ? null : bestId;
          if(!next) { setNeuronInfo('click a neuron to inspect'); return null; }
          const [liS,niS] = bestId!.split('-').map(Number);
          const act = (activationsRef.current[liS]?.[niS]??0).toFixed(3);
          const gap = liS-1;
          const inW = gap>=0 ? (weightsRef.current[gap]?.[niS]??[]) : [];
          const outW = liS<weightsRef.current.length ? (weightsRef.current[liS]?.[niS]??[]) : [];
          const fmt = (arr:number[]) =>
            arr.slice(0,4).map(w=>`<span style="color:${w>=0?'#1D9E75':'#D85A30'}">${w.toFixed(2)}</span>`).join('&nbsp;')
            +(arr.length>4?'…':'');
          setNeuronInfo(
            `<b>${TOPOLOGY[liS].shortName}[${niS}]</b>&nbsp; act=${act}`
            +(inW.length ?`&nbsp;&nbsp;in: ${fmt(inW)}`:'')
            +(outW.length?`&nbsp;&nbsp;out: ${fmt(outW)}`:'')
          );
          return next;
        });
      } else {
        setSelected(null);
        setNeuronInfo('click a neuron to inspect');
      }
    } else {
      if (bestId !== hovered) setHovered(bestId);
    }
  }, [nodes, filterGrids, hovered]);

  const probs  = probsRef.current;
  const topIdx = probs.indexOf(Math.max(...probs));
  const activeGrid = activeFilterLayer ? filterGrids[activeFilterLayer] : null;

  return (
    <div style={{
      background:'var(--color-background-secondary)', borderRadius:'8px',
      padding:'12px', display:'flex', flexDirection:'column', gap:'8px',
      fontFamily:'"JetBrains Mono", monospace',
    }}>

      <div style={{display:'flex',alignItems:'center',gap:'10px',flexWrap:'wrap'}}>
        <span style={{fontSize:'10px',color:'var(--color-text-tertiary)',letterSpacing:'.1em'}}>
          NETWORK GRAPH
        </span>
        <span style={{
          fontSize:'10px',padding:'2px 6px',borderRadius:'4px',
          background:dataSource==='real'?'var(--color-background-success)':'var(--color-background-secondary)',
          color:dataSource==='real'?'var(--color-text-success)':'var(--color-text-tertiary)',
          border:'0.5px solid',
          borderColor:dataSource==='real'?'var(--color-border-success)':'var(--color-border-tertiary)',
        }}>
          {dataSource==='real'?'real weights':'simulated'}
        </span>
        <div style={{marginLeft:'auto',display:'flex',alignItems:'center',gap:'8px'}}>
          <button onClick={()=>setPaused(p=>!p)} style={{
            fontSize:'11px',padding:'2px 8px',cursor:'pointer',fontFamily:'inherit',
            border:'0.5px solid var(--color-border-secondary)',borderRadius:'4px',
            background:paused?'var(--color-background-secondary)':'transparent',
            color:'var(--color-text-secondary)',
          }}>{paused?'resume':'pause'}</button>
        </div>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'1fr 110px',gap:'10px',alignItems:'start'}}>
        <canvas 
          ref={canvasRef} 
          width={width}
          height={height}
          style={{display:'block', width: '100%', height: 'auto', cursor: hovered ? 'pointer' : 'default'}}
          onClick={(e) => handleCanvasInteraction(e, true)}
          onMouseMove={(e) => handleCanvasInteraction(e, false)}
          onMouseLeave={() => setHovered(null)}
        />

        <div style={{paddingTop:'10px'}}>
          <div style={{fontSize:'10px',color:'var(--color-text-tertiary)',marginBottom:'6px',letterSpacing:'.05em'}}>
            OUTPUT
          </div>
          {CONF_COLORS.map((col,i)=>{
            const pct=probs[i]*100;
            const isTop=i===topIdx;
            return (
              <div key={i} style={{marginBottom:'7px'}}>
                <div style={{
                  display:'flex',justifyContent:'space-between',fontSize:'10px',
                  color:isTop?'var(--color-text-primary)':'var(--color-text-secondary)',
                  marginBottom:'2px',
                }}>
                  <span>class {i}</span>
                  <span style={{fontWeight:isTop?500:400}}>{pct.toFixed(1)}%</span>
                </div>
                <div style={{height:'5px',background:'var(--color-border-tertiary)',borderRadius:'3px',overflow:'hidden'}}>
                  <div style={{
                    height:'100%',width:`${pct}%`,background:col,borderRadius:'3px',
                    transition:'width 50ms ease',
                  }}/>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {activeGrid && activeFilterLayer && (
        <div style={{
          borderTop:'0.5px solid var(--color-border-tertiary)',
          paddingTop:'10px',
        }}>
          <FilterGrid
            filters={activeGrid.filters}
            kH={activeGrid.kH}
            kW={activeGrid.kW}
            layerName={activeFilterLayer}
          />
        </div>
      )}

      <div
        style={{
          borderTop:'0.5px solid var(--color-border-tertiary)',paddingTop:'6px',
          fontSize:'11px',fontFamily:'monospace',minHeight:'20px',
          color:selected?'var(--color-text-secondary)':'var(--color-text-tertiary)',
        }}
        dangerouslySetInnerHTML={{__html: neuronInfo}}
      />
    </div>
  );
}