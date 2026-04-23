import { useState } from 'react';
import type { EngineMetrics } from '../hooks/useMetrics';

interface Props {
  metrics:   EngineMetrics | null;
  available: boolean;
  mode: 'live' | 'replay';
  onReloadTape?: (filename: string) => void;
}

function GaugeBar({ pct, color }: { pct: number; color: string }) {
  return (
    <div style={{
      height: '6px', background: '#1e1e1e', borderRadius: '3px',
      overflow: 'hidden', marginTop: '6px',
    }}>
      <div style={{
        height: '100%', width:  `${Math.min(100, pct)}%`,
        background: color, borderRadius: '3px', transition: 'width 0.3s ease',
      }}/>
    </div>
  );
}

function Stat({ label, value, mono = true }: { label: string; value: string | number; mono?: boolean }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0' }}>
      <span style={{ fontSize: '11px', color: '#555' }}>{label}</span>
      <span style={{
        fontSize: '11px', color: '#aaa',
        fontFamily: mono ? '"JetBrains Mono", monospace' : 'inherit',
      }}>{value}</span>
    </div>
  );
}

export function EnginePanel({ metrics, available, mode, onReloadTape }: Props) {
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');

  const handleSaveBaseline = async () => {
    setSaveStatus('saving');
    try {
      const res = await fetch('http://localhost:9090/api/save-baseline', { method: 'POST' });
      
      if (res.ok) {
        const data = await res.json(); 
        setSaveStatus('saved');
        setTimeout(() => setSaveStatus('idle'), 2000);
        
        // Pass the filename up to App.tsx
        if (onReloadTape && data.file) {
           onReloadTape(data.file);
        }
      } else {
        setSaveStatus('error');
      }

    } catch (e) {
      setSaveStatus('error');
    }
  };

  const isReplay = metrics && !metrics.ring_buffer;

  // Offline & Replay States
  if (mode === 'replay' || isReplay) {
    return (
      <div style={{
        background: '#141414', border: '1px solid #222', borderRadius: '8px',
        padding: '14px 16px', fontFamily: '"JetBrains Mono", monospace',
      }}>
        <div style={{ fontSize: '10px', color: '#444', letterSpacing: '0.1em', marginBottom: '8px' }}>
          ENGINE INTERNALS
        </div>
        <div style={{ fontSize: '11px', color: '#555' }}>
          live internals offline (replay mode active)
        </div>
      </div>
    )
  }
  
  if (!available || !metrics) {
    return (
      <div style={{
        background: '#141414', border: '1px solid #222', borderRadius: '8px',
        padding: '14px 16px', fontFamily: '"JetBrains Mono", monospace',
      }}>
        <div style={{ fontSize: '10px', color: '#444', letterSpacing: '0.1em', marginBottom: '8px' }}>
          ENGINE INTERNALS
        </div>
        <div style={{ fontSize: '11px', color: '#555' }}>
          metrics endpoint unavailable — start engine with --mport 9090
        </div>
      </div>
    );
  }

  // Live Mode State
  const rb   = metrics.ring_buffer;
  const smp  = metrics.sampler;
  const log  = metrics.log;
  const ws   = metrics.ws;

  const fillColor = rb.fill_pct > 80 ? '#E24B4A' : rb.fill_pct > 50 ? '#BA7517' : '#1D9E75';
  const modeColor = smp.mode === 'passthrough' ? '#1D9E75' : '#BA7517';
  const dropPct   = smp.seen > 0 ? ((smp.dropped / smp.seen) * 100).toFixed(1) : '0.0';

  return (
    <div style={{
      background: '#141414', border: '1px solid #222', borderRadius: '8px',
      padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: '12px',
      fontFamily: '"JetBrains Mono", monospace',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ fontSize: '10px', color: '#444', letterSpacing: '0.1em' }}>
          ENGINE INTERNALS
        </div>
        
        <button 
          onClick={handleSaveBaseline}
          disabled={saveStatus === 'saving'}
          style={{
            background: 'transparent', border: '1px solid #2a2a2a', borderRadius: '4px',
            color: saveStatus === 'saved' ? '#1D9E75' : saveStatus === 'error' ? '#D85A30' : '#666',
            fontSize: '9px', padding: '2px 6px', cursor: 'pointer', fontFamily: 'inherit'
          }}
        >
          {saveStatus === 'idle'   && 'save baseline'}
          {saveStatus === 'saving' && 'saving...'}
          {saveStatus === 'saved'  && '✓ saved'}
          {saveStatus === 'error'  && '✕ error'}
        </button>
      </div>

      {/* Ring buffer */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
          <span style={{ fontSize: '11px', color: '#555' }}>ring buffer</span>
          <span style={{ fontSize: '14px', fontWeight: 500, color: fillColor }}>
            {rb.fill_pct.toFixed(1)}%
          </span>
        </div>
        <GaugeBar pct={rb.fill_pct} color={fillColor} />
        <div style={{ marginTop: '6px' }}>
          <Stat label="depth"     value={`${rb.size} / ${rb.capacity}`} />
          <Stat label="overflows" value={rb.overflows_total.toLocaleString()} />
          <Stat label="total in"  value={rb.writes_total.toLocaleString()} />
        </div>
      </div>

      {/* Sampler */}
      <div style={{ borderTop: '1px solid #1e1e1e', paddingTop: '10px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
          <span style={{ fontSize: '11px', color: '#555' }}>sampler</span>
          <span style={{
            fontSize: '10px', padding: '2px 6px', borderRadius: '4px',
            background: modeColor + '22', color: modeColor, letterSpacing: '0.05em',
          }}>
            {smp.mode.replace('_', '-')}
          </span>
        </div>
        <Stat label="drop rate"    value={`${dropPct}%`} />
        <Stat label="ema loss"     value={smp.ema_loss.toFixed(4)} />
        <Stat label="ema grad"     value={smp.ema_grad_norm.toFixed(4)} />
      </div>

      {/* Log + WS */}
      <div style={{ borderTop: '1px solid #1e1e1e', paddingTop: '10px' }}>
        <Stat label="log records"  value={log.records.toLocaleString()} />
        <Stat label="ws clients"   value={ws.clients} />
        <Stat label="frames sent"  value={ws.frames_sent.toLocaleString()} />
      </div>
    </div>
  );
}