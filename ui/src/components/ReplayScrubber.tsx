import { useRef, useCallback } from 'react';
import type { ChapterEntry, ReplayState } from '../hooks/useReplay';

interface Props {
  state:       ReplayState;
  connected:   boolean;
  currentStep: number;
  totalSteps:  number;
  chapters:    ChapterEntry[];
  onPlay:      (speed: number) => void;
  onPause:     () => void;
  onSeek:      (step: number)  => void;
}

const ANOMALY_COLORS: Record<string, string> = {
  GRAD_EXPLOSION: '#E24B4A',
  LOSS_SPIKE:     '#BA7517',
  LOSS_PLATEAU:   '#378ADD',
};

const SPEED_OPTIONS = [1, 5, 10];

export function ReplayScrubber({
  state, connected, currentStep, totalSteps, chapters,
  onPlay, onPause, onSeek,
}: Props) {
  const barRef = useRef<HTMLDivElement>(null);
  const pct    = totalSteps > 0 ? Math.min(100, (currentStep / totalSteps) * 100) : 0;

  const handleBarClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!barRef.current || totalSteps === 0 || !connected) return;
    const rect  = barRef.current.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    onSeek(Math.round(Math.max(0, Math.min(1, ratio)) * totalSteps));
  }, [totalSteps, onSeek, connected]);

  const statusColor = connected
    ? (state === 'playing' ? '#1D9E75' : state === 'paused' ? '#BA7517' : '#555')
    : '#E24B4A';

  const statusLabel = !connected
    ? 'disconnected — start engine with --replay'
    : state === 'idle'    ? 'idle — press a speed button to play'
    : state === 'paused'  ? 'paused'
    : 'playing';

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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '10px', color: '#444', letterSpacing: '0.1em' }}>REPLAY</span>
          <span style={{
            display: 'inline-block', width: '7px', height: '7px',
            borderRadius: '50%', background: statusColor,
            boxShadow: state === 'playing' ? `0 0 5px ${statusColor}88` : 'none',
          }} />
          <span style={{ fontSize: '10px', color: statusColor }}>{statusLabel}</span>
        </div>
        <span style={{ fontSize: '11px', color: '#555' }}>
          {currentStep.toLocaleString()} / {totalSteps.toLocaleString()}
        </span>
      </div>

      {/* Timeline bar */}
      <div
        ref={barRef}
        onClick={handleBarClick}
        style={{
          position: 'relative',
          height: '28px',
          background: '#1a1a1a',
          borderRadius: '4px',
          cursor: connected ? 'pointer' : 'not-allowed',
          opacity: connected ? 1 : 0.5,
        }}
      >
        {/* Progress fill */}
        <div style={{
          position: 'absolute', left: 0, top: 0,
          width: `${pct}%`, height: '100%',
          background: '#2a2a2a', borderRadius: '4px',
          transition: 'width 0.2s linear',
        }} />

        {/* Playhead */}
        <div style={{
          position: 'absolute',
          left: `calc(${pct}% - 1px)`,
          top: 0, bottom: 0,
          width: '2px',
          background: state === 'playing' ? '#1D9E75' : '#666',
          borderRadius: '1px',
        }} />

        {/* Chapter ticks */}
        {totalSteps > 0 && chapters.map((ch, i) => {
          const left   = Math.min(99.5, (ch.step / totalSteps) * 100);
          const color  = ANOMALY_COLORS[ch.type] ?? '#888';
          const height = Math.round(8 + ch.severity * 14);
          return (
            <div
              key={i}
              onClick={e => { e.stopPropagation(); if (connected) onSeek(ch.step); }}
              title={`${ch.type} at step ${ch.step}\n${ch.description}`}
              style={{
                position: 'absolute',
                left: `${left}%`,
                bottom: 0,
                width: '3px', height: `${height}px`,
                background: color,
                borderRadius: '1px 1px 0 0',
                cursor: connected ? 'pointer' : 'not-allowed',
                opacity: 0.85,
                transform: 'translateX(-50%)',
                transition: 'height 0.2s',
              }}
            />
          );
        })}
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        {state === 'playing' ? (
          <button
            onClick={onPause}
            disabled={!connected}
            style={{
              background: 'transparent',
              border: '1px solid #333', color: '#ccc',
              padding: '4px 12px', borderRadius: '4px',
              cursor: 'pointer', fontFamily: 'inherit', fontSize: '11px',
            }}
          >
            pause
          </button>
        ) : (
          SPEED_OPTIONS.map(s => (
            <button
              key={s}
              onClick={() => onPlay(s)}
              disabled={!connected}
              style={{
                background: 'transparent',
                border: `1px solid ${connected ? '#333' : '#222'}`,
                color: connected ? '#999' : '#444',
                padding: '4px 10px', borderRadius: '4px',
                cursor: connected ? 'pointer' : 'not-allowed',
                fontFamily: 'inherit', fontSize: '11px',
              }}
            >
              {s}×
            </button>
          ))
        )}

        {/* Legend */}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: '10px' }}>
          {Object.entries(ANOMALY_COLORS).map(([type, color]) => (
            <span key={type} style={{ fontSize: '10px', color: '#444', display: 'flex', alignItems: 'center', gap: '3px' }}>
              <span style={{ display: 'inline-block', width: '8px', height: '8px', background: color, borderRadius: '1px' }} />
              {type.replace('_', ' ').toLowerCase()}
            </span>
          ))}
        </div>
      </div>

      {/* Chapter list */}
      {chapters.length > 0 && (
        <div style={{ borderTop: '1px solid #1e1e1e', paddingTop: '6px', maxHeight: '72px', overflowY: 'auto' }}>
          {chapters.map((ch, i) => (
            <div
              key={i}
              onClick={() => { if (connected) onSeek(ch.step); }}
              style={{
                display: 'flex', gap: '8px', alignItems: 'baseline',
                padding: '2px 0',
                cursor: connected ? 'pointer' : 'default',
                opacity: connected ? 1 : 0.5,
              }}
            >
              <span style={{ fontSize: '10px', color: ANOMALY_COLORS[ch.type], minWidth: '32px', textAlign: 'right' }}>
                {ch.step}
              </span>
              <span style={{ fontSize: '10px', color: '#444' }}>{ch.description}</span>
            </div>
          ))}
        </div>
      )}

      {!connected && (
        <div style={{ fontSize: '10px', color: '#333', paddingTop: '2px' }}>
          run: ./build/engine --replay baseline.omlog
        </div>
      )}
    </div>
  );
}
