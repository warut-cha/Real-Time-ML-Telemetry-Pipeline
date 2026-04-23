import { useEffect, useRef, useState, useMemo } from 'react';
import type { TensorSnapshot, TrainingEvent } from '../hooks/useStream';

// ── Minimal t-SNE ─────────────────────────────────────────────────────────────
// Barnes-Hut approximation is overkill for ≤500 points.
// This is a clean exact t-SNE implementation — correct perplexity binary search,
// symmetric P, gradient descent with momentum and gains.
// Runs in a Web Worker via inline blob URL so it never blocks the UI thread.

const TSNE_WORKER_SRC = `
const PERPLEXITY = 15;
const LEARNING_RATE = 100;
const MOMENTUM_EARLY = 0.5;
const MOMENTUM_LATE  = 0.8;
const EARLY_EXAGGERATION = 4.0;
const EARLY_STOP = 100;
const MAX_ITER = 500;

function euclideanSq(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i]-b[i])**2;
  return s;
}

function computeP(X, perplexity) {
  const n = X.length;
  const logPerp = Math.log(perplexity);
  const P = Array.from({length:n}, () => new Float64Array(n));

  for (let i = 0; i < n; i++) {
    let betaMin = -Infinity, betaMax = Infinity, beta = 1.0;
    const dists = X.map((x,j) => i===j ? 0 : euclideanSq(X[i], x));

    for (let iter = 0; iter < 50; iter++) {
      let sumP = 0, H = 0;
      const Pi = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        Pi[j] = Math.exp(-dists[j] * beta);
        sumP += Pi[j];
      }
      if (sumP === 0) sumP = 1e-10;
      for (let j = 0; j < n; j++) {
        Pi[j] /= sumP;
        if (Pi[j] > 1e-10) H -= Pi[j] * Math.log(Pi[j]);
      }
      const Hdiff = H - logPerp;
      if (Math.abs(Hdiff) < 1e-5) { P[i] = Pi; break; }
      if (Hdiff > 0) { betaMin = beta; beta = betaMax === Infinity ? beta*2 : (beta+betaMax)/2; }
      else           { betaMax = beta; beta = betaMin === -Infinity ? beta/2 : (beta+betaMin)/2; }
      P[i] = Pi;
    }
  }

  // Symmetrise and early exaggerate
  const Psym = Array.from({length:n}, () => new Float64Array(n));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      Psym[i][j] = (P[i][j] + P[j][i]) / (2*n);
  return Psym;
}

function tsne(X, labels) {
  const n = X.length;
  if (n < 3) return;

  const P = computeP(X, Math.min(PERPLEXITY, (n-1)/3));

  // Random init
  let Y = Array.from({length:n}, () => [
    (Math.random()-0.5)*0.01,
    (Math.random()-0.5)*0.01
  ]);
  let iY    = Array.from({length:n}, () => [0,0]);
  let gains = Array.from({length:n}, () => [1,1]);

  for (let iter = 0; iter < MAX_ITER; iter++) {
    const exag  = iter < EARLY_STOP ? EARLY_EXAGGERATION : 1.0;
    const mom   = iter < EARLY_STOP ? MOMENTUM_EARLY : MOMENTUM_LATE;

    // Q matrix (student t)
    const Q   = Array.from({length:n}, () => new Float64Array(n));
    let sumQ  = 0;
    for (let i = 0; i < n; i++)
      for (let j = i+1; j < n; j++) {
        const q = 1 / (1 + euclideanSq(Y[i], Y[j]));
        Q[i][j] = Q[j][i] = q;
        sumQ += 2*q;
      }
    if (sumQ === 0) sumQ = 1e-10;

    // Gradient
    const dY = Array.from({length:n}, () => [0,0]);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        if (i===j) continue;
        const qij  = Q[i][j] / sumQ;
        const mult = 4 * (exag * P[i][j] - qij) * Q[i][j];
        dY[i][0] += mult * (Y[i][0]-Y[j][0]);
        dY[i][1] += mult * (Y[i][1]-Y[j][1]);
      }

    // Update with momentum + adaptive gains
    for (let i = 0; i < n; i++)
      for (let d = 0; d < 2; d++) {
        gains[i][d] = (Math.sign(dY[i][d]) !== Math.sign(iY[i][d]))
          ? gains[i][d] + 0.2 : Math.max(0.01, gains[i][d]*0.8);
        iY[i][d]  = mom * iY[i][d] - LEARNING_RATE * gains[i][d] * dY[i][d];
        Y[i][d]  += iY[i][d];
      }

    // Zero-mean
    const my = Y.reduce((a,y)=>[a[0]+y[0]/n, a[1]+y[1]/n], [0,0]);
    Y = Y.map(y => [y[0]-my[0], y[1]-my[1]]);

    // Post progress every 25 iters
    if (iter % 25 === 0 || iter === MAX_ITER-1) {
      self.postMessage({ type:'progress', iter, points: Y.map((y,i)=>({x:y[0],y:y[1],label:labels[i]})) });
    }
  }
}

self.onmessage = (e) => {
  tsne(e.data.X, e.data.labels);
  self.postMessage({ type:'done' });
};
`;

// ── Types ─────────────────────────────────────────────────────────────────────

interface Point { x: number; y: number; label: number; }

// 10 MNIST digit colours — index = digit class
const DIGIT_COLORS = [
  '#E24B4A', '#D85A30', '#BA7517', '#639922', '#1D9E75',
  '#378ADD', '#534AB7', '#D4537E', '#5F5E5A', '#0F6E56',
];

// ── Props ─────────────────────────────────────────────────────────────────────

interface Props {
  snapshots:   TensorSnapshot[];
  events:      TrainingEvent[];
  layerName?:  string;   // which layer to project (default: 'classifier.1')
}

// ── Component ─────────────────────────────────────────────────────────────────

export function EmbeddingProjection({
  snapshots,
  events,
  layerName = 'classifier.1',
}: Props) {
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const workerRef    = useRef<Worker | null>(null);
  const [points,     setPoints]     = useState<Point[]>([]);
  const [iteration,  setIteration]  = useState(0);
  const [running,    setRunning]    = useState(false);
  const [pointCount, setPointCount] = useState(0);
  const [hoveredPt,  setHoveredPt]  = useState<Point | null>(null);

  // Collect activation snapshots for the target layer — keep a rolling buffer
  // of the last 200 samples (enough for meaningful t-SNE, not too slow)
  const activationBuffer = useRef<{ values: number[]; label: number }[]>([]);

  // Track predicted class from the latest event accuracy signal
  const labelRef = useRef(0);

  useEffect(() => {
    const latest = events.at(-1);
    if (!latest) return;
    // Derive predicted class from step mod 10 as a placeholder until
    // real label data flows through — replace with actual logit argmax
    // if you extend the TrainingEvent to include predicted_class.
    labelRef.current = latest.step % 10;
  }, [events]);

  useEffect(() => {
    const actSnap = snapshots
      .filter(s => s.layer_name === layerName && s.tensor_type === 'ACTIVATION')
      .at(-1);
    if (!actSnap || !actSnap.values.length) return;

    // Each activation snapshot is one forward pass through the layer.
    // Shape is [batch, features] — take the first sample from the batch.
    const featDim = actSnap.shape[actSnap.shape.length - 1] ?? actSnap.values.length;
    const sample  = actSnap.values.slice(0, featDim);

    activationBuffer.current.push({ values: sample, label: labelRef.current });
    if (activationBuffer.current.length > 200) activationBuffer.current.shift();
  }, [snapshots, layerName]);

  const runTSNE = () => {
    const buf = activationBuffer.current;
    if (buf.length < 5) return;

    // Terminate any running worker
    workerRef.current?.terminate();

    const blob   = new Blob([TSNE_WORKER_SRC], { type: 'application/javascript' });
    const worker = new Worker(URL.createObjectURL(blob));
    workerRef.current = worker;
    setRunning(true);
    setPointCount(buf.length);

    worker.onmessage = (e) => {
      if (e.data.type === 'progress') {
        setIteration(e.data.iter);
        setPoints(e.data.points);
      } else if (e.data.type === 'done') {
        setRunning(false);
      }
    };

    worker.postMessage({
      X:      buf.map(b => b.values),
      labels: buf.map(b => b.label),
    });
  };

  // Draw onto canvas whenever points update
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || points.length === 0) return;

    const W = canvas.width, H = canvas.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, W, H);

    // Normalise to canvas space with padding
    const xs   = points.map(p => p.x);
    const ys   = points.map(p => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const pad  = 20;
    const scaleX = (maxX - minX) === 0 ? 1 : (W - pad*2) / (maxX - minX);
    const scaleY = (maxY - minY) === 0 ? 1 : (H - pad*2) / (maxY - minY);
    const scale  = Math.min(scaleX, scaleY);

    const toScreen = (p: Point) => ({
      sx: pad + (p.x - minX) * scale + (W - pad*2 - (maxX-minX)*scale) / 2,
      sy: pad + (p.y - minY) * scale + (H - pad*2 - (maxY-minY)*scale) / 2,
    });

    points.forEach(p => {
      const { sx, sy } = toScreen(p);
      const col = DIGIT_COLORS[p.label % DIGIT_COLORS.length];
      const r   = hoveredPt === p ? 6 : 4;
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fillStyle = col + 'cc';
      ctx.fill();
      ctx.strokeStyle = col;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    });
  }, [points, hoveredPt]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || points.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const my = (e.clientY - rect.top)  * (canvas.height / rect.height);

    const xs    = points.map(p => p.x), ys = points.map(p => p.y);
    const minX  = Math.min(...xs), maxX = Math.max(...xs);
    const minY  = Math.min(...ys), maxY = Math.max(...ys);
    const pad   = 20;
    const W = canvas.width, H = canvas.height;
    const scale = Math.min(
      (maxX-minX)===0 ? 1 : (W-pad*2)/(maxX-minX),
      (maxY-minY)===0 ? 1 : (H-pad*2)/(maxY-minY)
    );
    const offX = pad + (W-pad*2-(maxX-minX)*scale)/2;
    const offY = pad + (H-pad*2-(maxY-minY)*scale)/2;

    let best: Point | null = null, bestD = 64;
    points.forEach(p => {
      const sx = offX + (p.x-minX)*scale;
      const sy = offY + (p.y-minY)*scale;
      const d  = (sx-mx)**2 + (sy-my)**2;
      if (d < bestD) { bestD = d; best = p; }
    });
    setHoveredPt(best);
  };

  const bufLen = activationBuffer.current.length;

  return (
    <div style={{
      background: '#141414',
      border: '1px solid #222',
      borderRadius: '8px',
      padding: '14px 16px',
      fontFamily: '"JetBrains Mono", monospace',
      display: 'flex',
      flexDirection: 'column',
      gap: '10px',
    }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '6px' }}>
        <div>
          <div style={{ fontSize: '10px', color: '#444', letterSpacing: '0.1em' }}>EMBEDDING PROJECTION</div>
          <div style={{ fontSize: '10px', color: '#555', marginTop: '2px' }}>
            {layerName} · t-SNE · {bufLen} samples
          </div>
        </div>
        <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
          {running && (
            <span style={{ fontSize: '10px', color: '#BA7517' }}>iter {iteration}/500</span>
          )}
          <button
            onClick={runTSNE}
            disabled={bufLen < 5}
            style={{
              fontSize: '11px', padding: '3px 10px',
              border: '1px solid #2a2a2a',
              background: bufLen >= 5 ? '#1e1e1e' : 'transparent',
              color: bufLen >= 5 ? '#ccc' : '#444',
              borderRadius: '4px', cursor: bufLen >= 5 ? 'pointer' : 'default',
              fontFamily: 'inherit',
            }}
          >
            {running ? 'running…' : 'project'}
          </button>
        </div>
      </div>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        width={280}
        height={220}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredPt(null)}
        style={{
          width: '100%', height: '220px',
          background: '#0d0d0d',
          borderRadius: '4px',
          cursor: 'crosshair',
        }}
      />

      {/* Hover info */}
      {hoveredPt && (
        <div style={{ fontSize: '10px', color: '#666', fontFamily: 'monospace' }}>
          digit <span style={{ color: DIGIT_COLORS[hoveredPt.label % DIGIT_COLORS.length], fontWeight: 500 }}>
            {hoveredPt.label}
          </span>
          &nbsp;·&nbsp;({hoveredPt.x.toFixed(2)}, {hoveredPt.y.toFixed(2)})
        </div>
      )}

      {/* Legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
        {DIGIT_COLORS.map((col, i) => (
          <span key={i} style={{ display: 'flex', alignItems: 'center', gap: '3px', fontSize: '10px', color: '#555' }}>
            <span style={{ display: 'inline-block', width: '7px', height: '7px', borderRadius: '50%', background: col }} />
            {i}
          </span>
        ))}
      </div>

      {bufLen < 5 && (
        <div style={{ fontSize: '10px', color: '#333' }}>
          waiting for activation snapshots from {layerName}…
        </div>
      )}
    </div>
  );
}
