import { useState, useMemo } from 'react';
import type { TensorSnapshot } from '../hooks/useStream';

export type { TensorSnapshot };

const TYPE_COLORS: Record<string, { low: string; high: string }> = {
  ACTIVATION:     { low: '#042C53', high: '#378ADD' },
  WEIGHT:         { low: '#04342C', high: '#1D9E75' },
  GRADIENT:       { low: '#412402', high: '#D85A30' },
  NODE_EMBEDDING: { low: '#26215C', high: '#7F77DD' },
};

function lerp(t: number, low: string, high: string): string {
  const p = (hex: string) => [
    parseInt(hex.slice(1,3),16),
    parseInt(hex.slice(3,5),16),
    parseInt(hex.slice(5,7),16),
  ];
  const lo = p(low), hi = p(high);
  return `rgb(${Math.round(lo[0]+(hi[0]-lo[0])*t)},`
       + `${Math.round(lo[1]+(hi[1]-lo[1])*t)},`
       + `${Math.round(lo[2]+(hi[2]-lo[2])*t)})`;
}

function normalise(vals: number[]): number[] {
  const min = Math.min(...vals), max = Math.max(...vals);
  const range = max - min || 1;
  return vals.map(v => (v - min) / range);
}

//Flat heatmap (activations, gradients, linear weights)

function FlatHeatmap({ values, cols, colorLow, colorHigh }: {
  values: number[]; cols: number; colorLow: string; colorHigh: string;
}) {
  const norm     = normalise(values);
  const rows     = Math.ceil(values.length / cols);
  const cellSize = Math.max(2, Math.min(8, Math.floor(200 / Math.max(cols, rows))));

  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`, gap: '1px' }}>
      {norm.map((t, i) => (
        <div key={i} title={values[i].toFixed(4)} style={{
          width: cellSize, height: cellSize,
          background: lerp(t, colorLow, colorHigh),
          borderRadius: '1px',
        }} />
      ))}
    </div>
  );
}

// Per-filter kernel grid (conv weights: shape [outFilters, inChannels, kH, kW]) 
//
// Renders each filter as a separate kH×kW heatmap tile.
// If inChannels > 1, tiles from all input channels are shown side by side
// for each filter row, separated by a thin gap.
//
// Layout:
//   Row 0: [ filter_0_ch0 | filter_0_ch1 | ... ]
//   Row 1: [ filter_1_ch0 | filter_1_ch1 | ... ]
//   Row x: [ filter_x_ch0 | filter_x_ch1 | ... ]

function FilterGrid({ values, shape, colorLow, colorHigh, maxFilters = 32 }: {
  values:      number[];
  shape:       number[];
  colorLow:    string;
  colorHigh:   string;
  maxFilters?: number;
}) {
  const [hoveredFilter, setHoveredFilter] = useState<number | null>(null);

  const [outFilters, inChannels, kH, kW] = shape;
  const kernelSize  = kH * kW;
  const filterSize  = inChannels * kernelSize;
  const filtersShow = Math.min(outFilters, maxFilters);

  // Cell size — keep kernels visible but not huge
  // 3×3 kernel → 10px cells, 5×5 → 8px, larger → 6px
  const cellSize = kH <= 3 ? 10 : kH <= 5 ? 8 : 6;
  const gap      = 1;
  const tileW    = kW * cellSize + (kW - 1) * gap;
  const tileH    = kH * cellSize + (kH - 1) * gap;

  // Normalise all values globally so colours are comparable across filters
  const norm = normalise(values);

  return (
    <div>
      {/* Legend row */}
      <div style={{ fontSize: '10px', color: '#555', marginBottom: '6px', fontFamily: 'monospace' }}>
        {outFilters} filters · {inChannels} channel{inChannels > 1 ? 's' : ''} · {kH}×{kW} kernel
        {outFilters > maxFilters && ` · showing first ${maxFilters}`}
      </div>

      {/* Filter grid */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
        {Array.from({ length: filtersShow }, (_, fi) => {
          const isHov = hoveredFilter === fi;
          return (
            <div
              key={fi}
              onMouseEnter={() => setHoveredFilter(fi)}
              onMouseLeave={() => setHoveredFilter(null)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                padding: '2px 3px',
                borderRadius: '3px',
                background: isHov ? '#1e1e1e' : 'transparent',
                cursor: 'default',
              }}
            >
              {/* Filter index */}
              <span style={{
                fontSize: '9px', color: '#444', fontFamily: 'monospace',
                minWidth: '16px', textAlign: 'right',
              }}>
                {fi}
              </span>

              {/* One tile per input channel */}
              {Array.from({ length: inChannels }, (_, ci) => {
                const baseIdx = fi * filterSize + ci * kernelSize;
                const kernelNorm  = norm.slice(baseIdx, baseIdx + kernelSize);
                const kernelVals  = values.slice(baseIdx, baseIdx + kernelSize);

                return (
                  <div
                    key={ci}
                    title={`filter ${fi} ch ${ci}\nmin=${Math.min(...kernelVals).toFixed(3)} max=${Math.max(...kernelVals).toFixed(3)}`}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: `repeat(${kW}, ${cellSize}px)`,
                      gap: `${gap}px`,
                      outline: isHov ? '1px solid #333' : 'none',
                      borderRadius: '2px',
                    }}
                  >
                    {kernelNorm.map((t, ki) => (
                      <div key={ki} style={{
                        width: cellSize, height: cellSize,
                        background: lerp(t, colorLow, colorHigh),
                        borderRadius: '1px',
                      }} />
                    ))}
                  </div>
                );
              })}

              {/* Weight magnitude indicator */}
              {isHov && (
                <span style={{ fontSize: '9px', color: '#555', fontFamily: 'monospace', marginLeft: '4px' }}>
                  μ={( values.slice(fi*filterSize, (fi+1)*filterSize)
                        .reduce((a,b)=>a+Math.abs(b),0) / filterSize
                     ).toFixed(3)}
                </span>
              )}
            </div>
          );
        })}
      </div>

      {outFilters > maxFilters && (
        <div style={{ fontSize: '10px', color: '#444', marginTop: '4px', fontFamily: 'monospace' }}>
          + {outFilters - maxFilters} more filters
        </div>
      )}
    </div>
  );
}

//Shape classifier
// Determines which renderer to use based on tensor shape and type.

type RenderMode = 'filter_grid' | 'flat_heatmap';

function getRenderMode(snap: TensorSnapshot): RenderMode {
  // Conv weight: exactly 4 dims [outFilters, inChannels, kH, kW]
  if (snap.tensor_type === 'WEIGHT' && snap.shape.length === 4) {
    return 'filter_grid';
  }
  return 'flat_heatmap';
}
interface Props {
  snapshots: TensorSnapshot[];
}

//Component

export function LayerExplorer({ snapshots }: Props) {
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null);
  const [selectedType,  setSelectedType]  = useState<string>('WEIGHT');

  const layers = useMemo(() =>
    [...new Set(snapshots.map(s => s.layer_name))].sort()
  , [snapshots]);

  const latest = useMemo(() => {
    const target = selectedLayer ?? layers[0];
    return snapshots
      .filter(s => s.layer_name === target && s.tensor_type === selectedType)
      .at(-1);
  }, [snapshots, selectedLayer, selectedType, layers]);

  if (layers.length === 0) {
    return (
      <div style={{
        background: '#141414', border: '1px solid #222', borderRadius: '8px',
        padding: '14px 16px', fontFamily: '"JetBrains Mono", monospace',
      }}>
        <div style={{ fontSize: '10px', color: '#444', letterSpacing: '0.1em', marginBottom: '8px' }}>
          LAYER EXPLORER
        </div>
        <div style={{ fontSize: '11px', color: '#333' }}>
          no snapshots yet — waiting for first capture_snapshot()
        </div>
      </div>
    );
  }

  const colors     = TYPE_COLORS[selectedType] ?? TYPE_COLORS.WEIGHT;
  const renderMode = latest ? getRenderMode(latest) : 'flat_heatmap';

  return (
    <div style={{
      background: '#141414', border: '1px solid #222', borderRadius: '8px',
      padding: '14px 16px', fontFamily: '"JetBrains Mono", monospace',
      display: 'flex', flexDirection: 'column', gap: '10px',
    }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: '10px', color: '#444', letterSpacing: '0.1em' }}>
          LAYER EXPLORER
        </span>
        {latest && (
          <span style={{
            fontSize: '9px', color: '#555', fontFamily: 'monospace',
            padding: '1px 5px', border: '1px solid #2a2a2a', borderRadius: '3px',
          }}>
            {renderMode === 'filter_grid' ? 'filter grid' : 'heatmap'}
            &nbsp;·&nbsp;step {latest.step}
          </span>
        )}
      </div>

      {/* Layer selector */}
      <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
        {layers.map(l => {
          const active = (selectedLayer ?? layers[0]) === l;
          return (
            <button key={l} onClick={() => setSelectedLayer(l)} style={{
              background: active ? '#2a2a2a' : 'transparent',
              border: '1px solid #2a2a2a',
              color: active ? '#ccc' : '#555',
              padding: '2px 7px', borderRadius: '4px',
              cursor: 'pointer', fontFamily: 'inherit', fontSize: '10px',
            }}>
              {l.split('.').slice(-2).join('.')}
            </button>
          );
        })}
      </div>

      {/* Type selector */}
      <div style={{ display: 'flex', gap: '4px' }}>
        {(['WEIGHT', 'GRADIENT', 'ACTIVATION'] as const).map(t => (
          <button key={t} onClick={() => setSelectedType(t)} style={{
            background: selectedType === t ? '#2a2a2a' : 'transparent',
            border: '1px solid #2a2a2a',
            color: selectedType === t ? TYPE_COLORS[t].high : '#444',
            padding: '2px 7px', borderRadius: '4px',
            cursor: 'pointer', fontFamily: 'inherit', fontSize: '10px',
          }}>
            {t.toLowerCase()}
          </button>
        ))}
      </div>

      {/* Shape info */}
      {latest && (
        <div style={{ fontSize: '10px', color: '#555', fontFamily: 'monospace' }}>
          [{latest.shape.join(' × ')}]
          &nbsp;·&nbsp;{latest.values.length.toLocaleString()} values
          {latest.sample_rate < 0.999 && ` · ${(latest.sample_rate*100).toFixed(0)}% sampled`}
        </div>
      )}

      {/* Renderer - filter grid or flat heatmap */}
      {latest ? (
        <div style={{ maxHeight: '320px', overflowY: 'auto', overflowX: 'hidden' }}>
          {renderMode === 'filter_grid' ? (
            <FilterGrid
              values={latest.values}
              shape={latest.shape}
              colorLow={colors.low}
              colorHigh={colors.high}
            />
          ) : (
            <FlatHeatmap
              values={latest.values}
              cols={latest.shape[latest.shape.length - 1] ?? 8}
              colorLow={colors.low}
              colorHigh={colors.high}
            />
          )}
        </div>
      ) : (
        <div style={{ fontSize: '11px', color: '#333' }}>
          no {selectedType.toLowerCase()} data for this layer
        </div>
      )}
    </div>
  );
}
