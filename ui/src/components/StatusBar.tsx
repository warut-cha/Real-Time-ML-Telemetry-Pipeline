import type { ConnectionStatus } from '../hooks/useStream';

interface Props {
  status: ConnectionStatus;
  latencyMs: number | null;
  framesReceived: number;
  onClear: () => void;
}

const STATUS_COLORS: Record<ConnectionStatus, string> = {
  connected:    '#1D9E75',
  connecting:   '#BA7517',
  disconnected: '#888780',
  error:        '#E24B4A',
};

const STATUS_LABELS: Record<ConnectionStatus, string> = {
  connected:    'Live',
  connecting:   'Connecting',
  disconnected: 'Disconnected',
  error:        'Error',
};

export function StatusBar({ status, latencyMs, framesReceived, onClear }: Props) {
  const color = STATUS_COLORS[status];

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '16px',
      padding: '8px 16px',
      borderBottom: '1px solid #2a2a2a',
      fontFamily: '"JetBrains Mono", "Fira Code", monospace',
      fontSize: '12px',
      color: '#888',
      background: '#111',
      flexWrap: 'wrap',
    }}>
      {/* Status indicator */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <div style={{
          width: '7px', height: '7px', borderRadius: '50%',
          background: color,
          boxShadow: status === 'connected' ? `0 0 6px ${color}88` : 'none',
          animation: status === 'connecting' ? 'pulse 1s ease-in-out infinite' : 'none',
        }} />
        <span style={{ color }}>{STATUS_LABELS[status]}</span>
      </div>

      {/* Latency */}
      <div>
        latency:{' '}
        <span style={{ color: latencyMs !== null && latencyMs < 100 ? '#1D9E75' : '#E24B4A' }}>
          {latencyMs !== null ? `${latencyMs} ms` : '—'}
        </span>
      </div>

      {/* Frames */}
      <div>frames: <span style={{ color: '#ccc' }}>{framesReceived.toLocaleString()}</span></div>

      {/* Endpoint */}
      <div style={{ marginLeft: 'auto', color: '#555' }}>ws://localhost:8080/stream</div>

      {/* Clear button */}
      <button
        onClick={onClear}
        style={{
          background: 'transparent',
          border: '1px solid #333',
          color: '#666',
          padding: '2px 8px',
          borderRadius: '4px',
          cursor: 'pointer',
          fontFamily: 'inherit',
          fontSize: '11px',
        }}
      >
        clear
      </button>
    </div>
  );
}
