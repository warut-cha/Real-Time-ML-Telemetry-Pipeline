import { useEffect, useRef, useState } from 'react';

export interface EngineMetrics {
  ring_buffer: {
    size:             number;
    capacity:         number;
    fill_pct:         number;
    writes_total:     number;
    reads_total:      number;
    overflows_total:  number;
  };
  sampler: {
    mode:       'passthrough' | 'change_point';
    seen:       number;
    forwarded:  number;
    dropped:    number;
    ema_loss:   number;
    ema_grad_norm: number;
  };
  log: {
    records: number;
    path:    string;
  };
  ws: {
    clients:     number;
    frames_sent: number;
  };
}

interface UseMetricsOptions {
  url?:             string;
  pollIntervalMs?:  number;
}

interface UseMetricsResult {
  metrics:   EngineMetrics | null;
  available: boolean;  // false if /metrics is unreachable
}

export function useMetrics({
  url            = '/metrics',
  pollIntervalMs = 500,
}: UseMetricsOptions = {}): UseMetricsResult {
  const [metrics,   setMetrics]   = useState<EngineMetrics | null>(null);
  const [available, setAvailable] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let alive = true;

    const poll = async () => {
      try {
        const res = await fetch(url, { signal: AbortSignal.timeout(400) });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: EngineMetrics = await res.json();
        if (alive) { setMetrics(data); setAvailable(true); }
      } catch {
        if (alive) setAvailable(false);
      }
    };

    poll();
    timerRef.current = setInterval(poll, pollIntervalMs);

    return () => {
      alive = false;
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [url, pollIntervalMs]);

  return { metrics, available };
}