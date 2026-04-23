import { useEffect, useRef, useState, useCallback } from 'react';
import type { TrainingEvent, TensorSnapshot } from './useStream';

export interface ChapterEntry {
  step:         number;
  type:         'GRAD_EXPLOSION' | 'LOSS_PLATEAU' | 'LOSS_SPIKE';
  severity:     number;
  metric_value: number;
  description:  string;
}

export type ReplayState = 'idle' | 'playing' | 'paused';

interface UseReplayOptions {
  wsUrl?:        string;
  metricsUrl?:   string;
  chaptersUrl?:  string;
}

interface UseReplayResult {
  events:      TrainingEvent[];
  snapshots:   TensorSnapshot[];
  topology:    any;
  state:       ReplayState;
  currentStep: number;
  totalSteps:  number;
  chapters:    ChapterEntry[];
  connected:   boolean;
  play:        (speed?: number) => void;
  pause:       () => void;
  seek:        (step: number)   => void;
  stop:        () => void;
  clear:       () => void;
  reloadTape:  (filename?: string) => void;
}

export function useReplay({
  wsUrl       = 'ws://localhost:8081/stream',
  metricsUrl  = '/metrics',
  chaptersUrl = '/chapters',
}: UseReplayOptions = {}): UseReplayResult {

  const [events,      setEvents]      = useState<TrainingEvent[]>([]);
  const [snapshots,   setSnapshots]   = useState<TensorSnapshot[]>([]);
  const [topology,    setTopology]    = useState<any>(null);
  const [state,       setState]       = useState<ReplayState>('idle');
  const [currentStep, setCurrentStep] = useState(0);
  const [totalSteps,  setTotalSteps]  = useState(0);
  const [chapters,    setChapters]    = useState<ChapterEntry[]>([]);
  const [connected,   setConnected]   = useState(false);

  const wsRef        = useRef<WebSocket | null>(null);
  const reconnTimer  = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef   = useRef(true);

  // Load chapters from HTTP API 
  useEffect(() => {
    const load = () => {
      fetch(chaptersUrl)
        .then(r => r.ok ? r.json() : Promise.reject(r.status))
        .then((data: ChapterEntry[]) => {
          if (mountedRef.current) setChapters(data);
        })
       .catch(() => {
         // Retry after 2s — engine may not be running yet.
         setTimeout(load, 2000);
       });
   };
  load();
  }, [chaptersUrl]);

  //Fetch total_steps from /metrics once
  useEffect(() => {
    fetch(metricsUrl)
      .then(r => r.ok ? r.json() : Promise.reject())
      .then((data: { total_steps?: number }) => {
        if (mountedRef.current && data.total_steps) {
          setTotalSteps(data.total_steps);
        }
      })
      .catch(() => {});
  }, [metricsUrl]);

  //WebSocket with auto-reconnect
  useEffect(() => {
    mountedRef.current = true;

    const connect = () => {
      if (!mountedRef.current) return;

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        setConnected(true);
        console.info('[replay] WebSocket connected');
        // Ask the engine for current status immediately on connect.
        ws.send(JSON.stringify({ cmd: 'status' }));
      };

      ws.onmessage = (msg: MessageEvent<string>) => {
        if (!mountedRef.current) return;
        let parsed: Record<string, unknown>;
        try { parsed = JSON.parse(msg.data); }
        catch { return; }

        if (parsed.type === 'topology') {
          if (mountedRef.current) setTopology(parsed);
        }

        if (parsed.type === 'snapshot') {
          if (mountedRef.current) {
            setSnapshots(prev => {
              const next = [...prev, parsed as unknown as TensorSnapshot];
              return next.length > 50 ? next.slice(-50) : next;
            });
          }
          return;
        }
        // replay_status messages carry state + step position
        if (parsed.type === 'replay_status') {
          if (parsed.state)       setState(parsed.state as ReplayState);
          if (parsed.total_steps) setTotalSteps(parsed.total_steps as number);
          if (parsed.step !== undefined) setCurrentStep(parsed.step as number);
          return;
        }

        // Training event add to events list and update step position
        if (parsed.type === 'event' || parsed.step !== undefined) {
          const ev = parsed as unknown as TrainingEvent;
          setCurrentStep(ev.step);
          setEvents(prev => {
            const next = [...prev, { ...ev, recv_ts: performance.now() }];
            return next.length > 5000 ? next.slice(-5000) : next;
          });
        }
      };

      ws.onerror = () => {
        // Don't log; expected when engine isn't running.
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        setConnected(false);
        // Reconnect after 2s — engine may not be in replay mode yet.
        reconnTimer.current = setTimeout(connect, 2000);
      };
    };

    connect();

    return () => {
      mountedRef.current = false;
      if (reconnTimer.current) clearTimeout(reconnTimer.current);
      wsRef.current?.close();
    };
  }, [wsUrl]);

  //Command sender
  const send = useCallback((cmd: object) => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(cmd));
    } else {
      console.warn('[replay] WebSocket not open — command dropped:', cmd);
    }
  }, []);

  const play  = useCallback((speed = 1.0) => send({ cmd: 'play',  speed }), [send]);
  const pause = useCallback(()            => send({ cmd: 'pause' }),          [send]);
  const seek  = useCallback((step: number)=> send({ cmd: 'seek',  step  }), [send]);
  const stop  = useCallback(()            => send({ cmd: 'stop'  }),          [send]);
  const clear = useCallback(()            => { setEvents([]); setCurrentStep(0); }, []);
  const reloadTape = useCallback((filename?: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const isString = typeof filename === 'string';
      const payload = isString ? { cmd: 'reload', file: filename } : { cmd: 'reload' };
      wsRef.current.send(JSON.stringify(payload));
      setEvents([]);
      setSnapshots([]);
      setTopology(null);
  }
},[]);
  return {
    events, snapshots, topology,state, currentStep, totalSteps,
    chapters, connected,
    play, pause, seek, stop, clear, reloadTape,
  };
}