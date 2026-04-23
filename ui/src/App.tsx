import { useMemo, useState } from 'react';
import { useStream }             from './hooks/useStream';
import { useMetrics }            from './hooks/useMetrics';
import { useReplay }             from './hooks/useReplay';
import { StatusBar }             from './components/StatusBar';
import { MetricChart }           from './components/MetricChart';
import { EnginePanel }           from './components/EnginePanel';
import { ReplayScrubber }        from './components/ReplayScrubber';
import { LayerExplorer }         from './components/LayerExplorer';
import { NetworkGraph }          from './components/NetworkGraph';
import { EmbeddingProjection }   from './components/EmbeddingProjection';
import { GNNGraph }              from './components/GNNGraph';
import type { TensorSnapshot }   from './hooks/useStream';

const WINDOW = 300;

export default function App() {
  const [mode, setMode] = useState<'live' | 'replay'>('live');
  const [vizTab, setVizTab] = useState<'cnn' | 'embed' | 'gnn'>('cnn');
  const [activeTape, setActiveTape] = useState<string|undefined>(undefined);
  const live = useStream({ url: '/stream', maxBuffer: 2000 });

  const { metrics, available } = useMetrics({ url: '/metrics' });

  const replay = useReplay({ wsUrl: 'ws://localhost:8083/stream', metricsUrl: 'http://localhost:9091/metrics', chaptersUrl: 'http://localhost:9091/chapters' });

  const activeEvents = mode === 'live' ? live.events : replay.events;
  const snapshots: TensorSnapshot[] = mode === 'live' ? live.snapshots : replay.snapshots;
  const topology = mode === 'live' ? live.topology : replay.topology;

  const lossData = useMemo(() => activeEvents.map(e => ({ step: e.step, value: e.loss })),      [activeEvents]);
  const accData  = useMemo(() => activeEvents.map(e => ({ step: e.step, value: e.accuracy })),  [activeEvents]);
  const gradData = useMemo(() => activeEvents.map(e => ({ step: e.step, value: e.grad_norm })), [activeEvents]);

  const gradMean = useMemo(() => {
    if (gradData.length < 20) return undefined;
    const w = gradData.slice(-50);
    return (w.reduce((s, p) => s + p.value, 0) / w.length) * 10;
  }, [gradData]);

  const latest = activeEvents.at(-1);

  const TAB_LABELS: { key: 'cnn' | 'embed' | 'gnn'; label: string }[] = [
    { key: 'cnn',   label: 'network graph' },
    { key: 'embed', label: 't-SNE projection' },
    { key: 'gnn',   label: 'GNN graph' },
  ];

  return (
    <div style={{ minHeight: '100vh', background: '#0d0d0d', color: '#ccc',
                  fontFamily: '"JetBrains Mono", "Fira Code", monospace' }}>

      {/* Top bar */}
      <div style={{ display: 'flex', alignItems: 'center', padding: '12px 20px',
                    borderBottom: '1px solid #1e1e1e', gap: '16px' }}>
        <span style={{ fontSize: '13px', color: '#666', letterSpacing: '0.15em' }}>OMNISTREAM</span>
        <span style={{ color: '#333' }}>/</span>
        <span style={{ fontSize: '13px', color: '#444' }}>DEMO</span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: '4px' }}>
          {(['live', 'replay'] as const).map(m => (
            <button key={m} onClick={() => setMode(m)} style={{
              background: mode === m ? '#1e1e1e' : 'transparent',
              border: '1px solid #222', color: mode === m ? '#ccc' : '#444',
              padding: '4px 12px', borderRadius: '4px', cursor: 'pointer',
              fontFamily: 'inherit', fontSize: '11px',
            }}>{m}</button>
          ))}
        </div>
      </div>

      <StatusBar
        status={mode === 'live' ? live.status : (replay.state === 'playing' ? 'connected' : 'disconnected')}
        latencyMs={mode === 'live' ? live.latencyMs : null}
        framesReceived={mode === 'live' ? live.framesReceived : replay.events.length}
        onClear={mode === 'live' ? live.clear : replay.clear}
      />

      {/* Summary row */}
      {latest && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
                      gap: '1px', background: '#1a1a1a', borderBottom: '1px solid #1e1e1e' }}>
          {[
            { label: 'step',      value: latest.step.toLocaleString(),             color: '#888' },
            { label: 'loss',      value: latest.loss.toFixed(6),                   color: '#378ADD' },
            { label: 'accuracy',  value: (latest.accuracy * 100).toFixed(2) + '%', color: '#1D9E75' },
            { label: 'grad_norm', value: latest.grad_norm.toFixed(4),              color: '#D85A30' },
          ].map(({ label, value, color }) => (
            <div key={label} style={{ padding: '14px 20px', background: '#0d0d0d' }}>
              <div style={{ fontSize: '10px', color: '#444', marginBottom: '4px', letterSpacing: '0.1em' }}>
                {label.toUpperCase()}
              </div>
              <div style={{ fontSize: '20px', fontWeight: 500, color }}>{value}</div>
            </div>
          ))}
        </div>
      )}

      {/* Metrics row */}
      <div style={{ display: 'grid',
                    gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr) minmax(0,1fr) 220px',
                    gap: '12px', padding: '16px 16px 8px' }}>
        <MetricChart title="Training loss"  data={lossData} color="#378ADD"
                     windowSize={WINDOW} domain={[0, 'auto']} />
        <MetricChart title="Accuracy"       data={accData}  color="#1D9E75"
                     windowSize={WINDOW} domain={[0, 1]} />
        <MetricChart title="Gradient norm"  data={gradData} color="#D85A30"
                     windowSize={WINDOW} anomalyThreshold={gradMean} />
        <EnginePanel metrics={metrics} available={available} mode={mode} onReloadTape={(filename) => {setActiveTape(filename); replay.reloadTape(filename);}} />
      </div>

      {/* Visualiser tabs */}
      <div style={{ padding: '0 16px 4px', display: 'flex', gap: '4px' }}>
        {TAB_LABELS.map(({ key, label }) => (
          <button key={key} onClick={() => setVizTab(key)} style={{
            fontSize: '11px', padding: '4px 12px',
            border: '1px solid #222',
            background: vizTab === key ? '#1e1e1e' : 'transparent',
            color: vizTab === key ? '#ccc' : '#444',
            borderRadius: '4px', cursor: 'pointer', fontFamily: 'inherit',
          }}>{label}</button>
        ))}
      </div>

      {/* Visualiser panels */}
      <div style={{ padding: '0 16px 8px' }}>

        <div style={{ display: vizTab === 'cnn' ? 'block' : 'none' }}>
          <NetworkGraph
            events={activeEvents}
            snapshots={snapshots}
            hz={4}
            width={900}
            height={300}
          />
        </div>

        <div style={{ 
          display: vizTab === 'embed' ? 'grid' : 'none', 
          gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr)', gap: '12px' 
        }}>
          <EmbeddingProjection
            snapshots={snapshots}
            events={activeEvents}
            layerName="classifier.1"
          />
          <LayerExplorer snapshots={snapshots} />
        </div>

        <div style={{ 
          display: vizTab === 'gnn' ? 'grid' : 'none', 
          gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr)', gap: '12px' 
        }}>
          <GNNGraph topology={topology} snapshots={snapshots} />
          <LayerExplorer snapshots={snapshots} />
        </div>

      </div>

      {/* Replay scrubber */}
      <div style={{ padding: '0 16px 16px' }}>
        {/*Replay mode */}
        {mode === 'replay' && (
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '8px' }}>
            <button 
              onClick={() => replay.reloadTape(activeTape)} 
              style={{
                background: '#1e1e1e', border: '1px solid #2a2a2a', borderRadius: '4px',
                color: '#ccc', fontSize: '10px', padding: '4px 8px', cursor: 'pointer', 
                fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: '4px'
              }}
            >
              Reload
            </button>
          </div>
        )}
        <ReplayScrubber
          state={replay.state}
          connected={replay.connected}
          currentStep={replay.currentStep}
          totalSteps={replay.totalSteps}
          chapters={replay.chapters}
          onPlay={replay.play}
          onPause={replay.pause}
          onSeek={replay.seek}
        />
      </div>

      {activeEvents.length === 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center',
                      padding: '60px 20px', color: '#333', gap: '12px' }}>
          <div style={{ fontSize: '13px' }}>
            {mode === 'live' ? 'waiting for training data...' : 'connect to replay engine on :8081'}
          </div>
        </div>
      )}

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
      `}</style>
    </div>
  );
}