import { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import type { GraphTopology, TensorSnapshot } from '../hooks/useStream';

interface LayoutNode {
  id:   number;
  x:    number;
  y:    number;
  vx:   number;
  vy:   number;
  label: string;
}

function initLayout(n: number, W: number, H: number): LayoutNode[] {
  return Array.from({ length: n }, (_, i) => {
    const angle = (i / n) * Math.PI * 2;
    const r     = Math.min(W, H) * 0.35;
    return {
      id:    i,
      x:     W/2 + r * Math.cos(angle) + (Math.random()-0.5)*20,
      y:     H/2 + r * Math.sin(angle) + (Math.random()-0.5)*20,
      vx:    0,
      vy:    0,
      label: '',
    };
  });
}

function tickLayout(
  nodes:    LayoutNode[],
  src:      number[],
  dst:      number[],
  W:        number,
  H:        number,
  alpha:    number,
): number {
  const REPULSION  = 800 * alpha;
  const SPRING_K   = 0.04 * alpha;
  const SPRING_LEN = Math.min(W, H) / Math.sqrt(nodes.length + 1) * 0.9;
  const DAMPING    = 0.85;

  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const dx = nodes[i].x - nodes[j].x || 0.01;
      const dy = nodes[i].y - nodes[j].y || 0.01;
      const d2 = Math.max(dx*dx + dy*dy, 1.0); 
      const f  = REPULSION / d2;
      nodes[i].vx += f * dx; nodes[i].vy += f * dy;
      nodes[j].vx -= f * dx; nodes[j].vy -= f * dy;
    }
  }

  for (let e = 0; e < src.length; e++) {
    const a  = nodes[src[e]], b = nodes[dst[e]];
    if (!a || !b) continue;
    const dx = b.x - a.x, dy = b.y - a.y;
    const d  = Math.sqrt(dx*dx+dy*dy) || 1;
    const f  = SPRING_K * (d - SPRING_LEN);
    a.vx += f*dx/d; a.vy += f*dy/d;
    b.vx -= f*dx/d; b.vy -= f*dy/d;
  }

  let ke = 0;
  for (const n of nodes) {
    n.vx *= DAMPING; n.vy *= DAMPING;
    n.x   = Math.max(20, Math.min(W-20, n.x + n.vx));
    n.y   = Math.max(20, Math.min(H-20, n.y + n.vy));
    ke   += n.vx*n.vx + n.vy*n.vy;
  }
  return ke;
}

function lerp(t: number, lo: string, hi: string): string {
  const p = (h: string) => [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
  const [al, bl, cl] = p(lo), [ah, bh, ch] = p(hi);
  return `rgb(${Math.round(al+(ah-al)*t)},${Math.round(bl+(bh-bl)*t)},${Math.round(cl+(ch-cl)*t)})`;
}

interface Pulse { edge: number; t: number; }

interface Props {
  topology:  GraphTopology | null;
  snapshots: TensorSnapshot[];
}

const CANVAS_W = 300;
const CANVAS_H = 240;

export function GNNGraph({ topology, snapshots }: Props) {
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const nodesRef    = useRef<LayoutNode[]>([]);
  const pulsesRef   = useRef<Pulse[]>([]);
  const rafRef      = useRef(0);
  const alphaRef    = useRef(1.0);
  const [settled,   setSettled]   = useState(false);
  const [selected,  setSelected]  = useState<number | null>(null);
  const [animPulse, setAnimPulse] = useState(true);

  const embeddingMag = useMemo(() => {
    const snap = snapshots
      .filter(s => s.tensor_type === 'NODE_EMBEDDING')
      .at(-1);
    if (!snap || !topology) return null;

    const n       = topology.num_nodes;
    const featDim = snap.shape[snap.shape.length - 1] ?? 1;
    const mags: number[] = [];

    for (let i = 0; i < n; i++) {
      const base = i * featDim;
      let s = 0;
      for (let d = 0; d < featDim; d++) {
        const v = snap.values[base + d] ?? 0;
        s += v * v;
      }
      mags.push(Math.sqrt(s));
    }
    const max = Math.max(...mags, 1);
    return mags.map(m => m / max);
  }, [snapshots, topology]);

  useEffect(() => {
    if (!topology) return;
    nodesRef.current = initLayout(topology.num_nodes, CANVAS_W, CANVAS_H);
    nodesRef.current.forEach((n, i) => {
      n.label = topology.node_labels?.[i] ?? String(i);
    });
    pulsesRef.current = [];
    alphaRef.current  = 1.0;
    setSettled(false);
    setSelected(null);
  }, [topology?.num_nodes, topology?.src.join(',')]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const topo   = topology;
    if (!canvas || !topo) return;

    const ctx  = canvas.getContext('2d');
    if (!ctx) return;
    const nodes = nodesRef.current;
    const W = canvas.width, H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    if (!settled && alphaRef.current > 0.01) {
      const ke = tickLayout(nodes, topo.src, topo.dst, W, H, alphaRef.current);
      alphaRef.current *= 0.995;
      if (ke < 0.01 * nodes.length) {
        alphaRef.current = 0;
        setSettled(true);
      }
    }

    if (animPulse && topo.src.length > 0) {
      pulsesRef.current = pulsesRef.current
        .map(p => ({ ...p, t: p.t + 0.015 }))
        .filter(p => p.t < 1);

      if (Math.random() < 0.08) {
        const edge = Math.floor(Math.random() * topo.src.length);
        pulsesRef.current.push({ edge, t: 0 });
      }
    }

    for (let e = 0; e < topo.src.length; e++) {
      const a = nodes[topo.src[e]], b = nodes[topo.dst[e]];
      if (!a || !b) continue;
      const w   = topo.edge_weights?.[e] ?? 1;
      const aw  = Math.abs(w);
      const sel = selected === topo.src[e] || selected === topo.dst[e];
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = sel ? '#EF9F27' : `rgba(83,74,183,${0.15 + aw * 0.4})`;
      ctx.lineWidth   = sel ? 2 : Math.max(0.5, aw * 2);
      ctx.stroke();

      if (topo.src.length < 80) {
        const angle = Math.atan2(b.y - a.y, b.x - a.x);
        const ax = b.x - 8 * Math.cos(angle);
        const ay = b.y - 8 * Math.sin(angle);
        ctx.beginPath();
        ctx.moveTo(ax + 4*Math.cos(angle-2.5), ay + 4*Math.sin(angle-2.5));
        ctx.lineTo(b.x - 5*Math.cos(angle), b.y - 5*Math.sin(angle));
        ctx.lineTo(ax + 4*Math.cos(angle+2.5), ay + 4*Math.sin(angle+2.5));
        ctx.fillStyle = sel ? '#EF9F27' : `rgba(83,74,183,${0.3 + aw*0.4})`;
        ctx.fill();
      }
    }

    pulsesRef.current.forEach(p => {
      const e  = p.edge;
      const a  = nodes[topo.src[e]], b = nodes[topo.dst[e]];
      if (!a || !b) return;
      const px = a.x + (b.x - a.x) * p.t;
      const py = a.y + (b.y - a.y) * p.t;
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI*2);
      ctx.fillStyle = `rgba(239,159,39,${1 - p.t})`;
      ctx.fill();
    });

    nodes.forEach((n, i) => {
      const mag  = embeddingMag?.[i] ?? 0.5;
      const isSel = selected === i;
      const col  = lerp(mag, '#3C3489', '#7F77DD');
      const r    = isSel ? 9 : 6;

      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI*2);
      ctx.fillStyle   = col + (isSel ? 'ff' : 'cc');
      ctx.strokeStyle = isSel ? '#EF9F27' : col;
      ctx.lineWidth   = isSel ? 2 : 0.5;
      ctx.fill();
      ctx.stroke();

      if (nodes.length <= 20 || isSel) {
        ctx.fillStyle  = '#888';
        ctx.font       = '9px monospace';
        ctx.textAlign  = 'center';
        ctx.fillText(n.label || String(i), n.x, n.y - r - 3);
      }
    });
  }, [topology, settled, selected, animPulse, embeddingMag]);

  useEffect(() => {
    if (!topology) return;
    let active = true;
    const loop = () => {
      if (!active) return;
      draw();
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    
    return () => {
      active = false;
      cancelAnimationFrame(rafRef.current);
    };
  }, [draw, topology]);

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx   = (e.clientX - rect.left) * (canvas.width  / rect.width);
    const my   = (e.clientY - rect.top)  * (canvas.height / rect.height);

    let best: number | null = null, bestD = 100;
    nodesRef.current.forEach((n, i) => {
      const d = (n.x-mx)**2 + (n.y-my)**2;
      if (d < bestD) { bestD = d; best = i; }
    });
    setSelected(s => s === best ? null : best);
  }, []);

  if (!topology) {
    return (
      <div style={{
        background: '#141414', border: '1px solid #222', borderRadius: '8px',
        padding: '14px 16px', fontFamily: '"JetBrains Mono", monospace',
      }}>
        <div style={{ fontSize: '10px', color: '#444', letterSpacing: '0.1em', marginBottom: '8px' }}>
          GNN GRAPH
        </div>
        <div style={{ fontSize: '11px', color: '#333' }}>
          waiting for graph topology — emit GraphTopology from your GNN hook
        </div>
        <div style={{ fontSize: '10px', color: '#2a2a2a', marginTop: '6px' }}>
          hook.emit_topology(edge_index, num_nodes, node_labels)
        </div>
      </div>
    );
  }

  const selNode = selected !== null ? nodesRef.current[selected] : null;
  const selEdges = selected !== null
    ? topology.src.map((s, i) => ({ s, d: topology.dst[i], w: topology.edge_weights?.[i] ?? 1 }))
        .filter(e => e.s === selected || e.d === selected)
    : [];

  return (
    <div style={{
      background: '#141414', border: '1px solid #222', borderRadius: '8px',
      padding: '14px 16px', fontFamily: '"JetBrains Mono", monospace',
      display: 'flex', flexDirection: 'column', gap: '8px',
    }}>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '4px' }}>
        <div>
          <div style={{ fontSize: '10px', color: '#444', letterSpacing: '0.1em' }}>GNN GRAPH</div>
          <div style={{ fontSize: '10px', color: '#555', marginTop: '2px' }}>
            {topology.num_nodes} nodes · {topology.src.length} edges
            {settled ? ' · settled' : ' · simulating…'}
          </div>
        </div>
        <div style={{ display: 'flex', gap: '6px' }}>
          <button
            onClick={() => setAnimPulse(p => !p)}
            style={{
              fontSize: '10px', padding: '2px 7px',
              border: '1px solid #2a2a2a',
              background: animPulse ? '#1e1e1e' : 'transparent',
              color: animPulse ? '#ccc' : '#555',
              borderRadius: '4px', cursor: 'pointer', fontFamily: 'inherit',
            }}
          >
            {animPulse ? 'pulses on' : 'pulses off'}
          </button>
          <button
            onClick={() => {
              if (!topology) return;
              nodesRef.current = initLayout(topology.num_nodes, CANVAS_W, CANVAS_H);
              alphaRef.current = 1.0;
              setSettled(false);
            }}
            style={{
              fontSize: '10px', padding: '2px 7px',
              border: '1px solid #2a2a2a', background: 'transparent',
              color: '#555', borderRadius: '4px', cursor: 'pointer', fontFamily: 'inherit',
            }}
          >
            re-layout
          </button>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        width={CANVAS_W}
        height={CANVAS_H}
        onClick={handleClick}
        style={{
          width: '100%', height: `${CANVAS_H}px`,
          background: '#0d0d0d', borderRadius: '4px', cursor: 'pointer',
        }}
      />

      {selNode && (
        <div style={{ fontSize: '10px', color: '#666', fontFamily: 'monospace' }}>
          node <span style={{ color: '#7F77DD', fontWeight: 500 }}>{selNode.label || selected}</span>
          &nbsp;·&nbsp;embedding mag {(embeddingMag?.[selected!] ?? 0).toFixed(3)}
          &nbsp;·&nbsp;{selEdges.length} edge{selEdges.length !== 1 ? 's' : ''}
        </div>
      )}

      <div style={{ display: 'flex', gap: '10px', fontSize: '10px', color: '#444' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
          <span style={{ display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%', background: '#7F77DD' }} />
          high embedding
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
          <span style={{ display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%', background: '#3C3489' }} />
          low embedding
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
          <span style={{ display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%', background: '#EF9F27' }} />
          message pulse
        </span>
      </div>
    </div>
  );
}