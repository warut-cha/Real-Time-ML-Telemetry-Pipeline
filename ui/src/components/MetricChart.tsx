import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';

interface DataPoint {
  step: number;
  value: number;
}

interface Props {
  title: string;
  data: DataPoint[];
  color: string;
  unit?: string;
  domain?: [number | 'auto', number | 'auto'];
  anomalyThreshold?: number;   // draws a dashed reference line
  windowSize?: number;         // how many points to display (default: all)
}

function CustomTooltip({ active, payload, label, unit }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#1a1a1a',
      border: '1px solid #333',
      borderRadius: '4px',
      padding: '6px 10px',
      fontFamily: '"JetBrains Mono", monospace',
      fontSize: '11px',
      color: '#ccc',
    }}>
      <div>step {label}</div>
      <div style={{ color: payload[0].color }}>
        {Number(payload[0].value).toFixed(4)} {unit ?? ''}
      </div>
    </div>
  );
}

export function MetricChart({
  title, data, color, unit, domain, anomalyThreshold, windowSize,
}: Props) {
  const displayData = windowSize && data.length > windowSize
    ? data.slice(data.length - windowSize)
    : data;

  const latest = data.at(-1);

  return (
    <div style={{
      background: '#141414',
      border: '1px solid #222',
      borderRadius: '8px',
      padding: '14px 16px',
      display: 'flex',
      flexDirection: 'column',
      gap: '8px',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <span style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: '11px',
          color: '#555',
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
        }}>
          {title}
        </span>
        <span style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: '18px',
          fontWeight: 500,
          color: latest ? color : '#333',
        }}>
          {latest ? latest.value.toFixed(4) : '—'}
        </span>
      </div>

      {/* Chart */}
      <div style={{ height: '120px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={displayData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
            <XAxis
              dataKey="step"
              tick={{ fill: '#444', fontSize: 10, fontFamily: 'monospace' }}
              tickLine={false}
              axisLine={{ stroke: '#222' }}
            />
            <YAxis
              domain={domain ?? ['auto', 'auto']}
              tick={{ fill: '#444', fontSize: 10, fontFamily: 'monospace' }}
              tickLine={false}
              axisLine={false}
              width={44}
              tickFormatter={(v: number) => v.toFixed(2)}
            />
            <Tooltip content={<CustomTooltip unit={unit} />} />
            {anomalyThreshold !== undefined && (
              <ReferenceLine
                y={anomalyThreshold}
                stroke="#E24B4A"
                strokeDasharray="4 4"
                strokeWidth={1}
              />
            )}
            <Line
              type="monotone"
              dataKey="value"
              stroke={color}
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}  // disable animation for live data
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
