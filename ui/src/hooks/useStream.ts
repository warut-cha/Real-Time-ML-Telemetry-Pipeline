import { useEffect, useRef, useCallback, useState } from 'react';

export interface TrainingEvent {
  step:      number;
  loss:      number;
  accuracy:  number;
  grad_norm: number;
  ts_ns:     number;
  recv_ts:   number;
}

export interface TensorSnapshot {
  step:        number;
  layer_name:  string;
  tensor_type: 'WEIGHT' | 'GRADIENT' | 'ACTIVATION' | 'NODE_EMBEDDING';
  shape:       number[];
  values:      number[];
  sample_rate: number;
  ts_ns:       number;
  recv_ts:     number;
}

export interface GraphTopology {
  step:          number;
  num_nodes:     number;
  src:           number[];
  dst:           number[];
  edge_weights:  number[];
  node_labels:   string[];
  ts_ns:         number;
  recv_ts:       number;
}

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

type WireMessage =
  | ({ type: 'event' }    & Omit<TrainingEvent, 'recv_ts'>)
  | ({ type: 'snapshot' } & Omit<TensorSnapshot, 'recv_ts'>)
  | ({ type: 'topology' } & Omit<GraphTopology, 'recv_ts'>);

interface UseStreamOptions {
  url?:         string;
  maxBuffer?:   number;
  onEvent?:     (e: TrainingEvent)   => void;
  onSnapshot?:  (s: TensorSnapshot)  => void;
  onTopology?:  (t: GraphTopology)   => void;
}

interface UseStreamResult {
  events:    TrainingEvent[];
  snapshots: TensorSnapshot[];
  topology:  GraphTopology | null;
  status:    ConnectionStatus;
  latencyMs: number | null;
  framesReceived: number;
  clear: () => void;
}

export function useStream({
  url       = '/stream',
  maxBuffer = 500,
  onEvent,
  onSnapshot,
  onTopology,
}: UseStreamOptions = {}): UseStreamResult {
  const [events,    setEvents]    = useState<TrainingEvent[]>([]);
  const [snapshots, setSnapshots] = useState<TensorSnapshot[]>([]);
  const [topology,  setTopology]  = useState<GraphTopology | null>(null);
  const [status,    setStatus]    = useState<ConnectionStatus>('connecting');
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [framesRx,  setFramesRx]  = useState(0);

  const wsRef         = useRef<WebSocket | null>(null);
  const latencyBuf    = useRef<number[]>([]);
  const framesRef     = useRef(0);
  
  const eventBuffer   = useRef<TrainingEvent[]>([]);
  const snapshotBuf   = useRef<TensorSnapshot[]>([]);

  const onEventRef    = useRef(onEvent);
  const onSnapRef     = useRef(onSnapshot);
  const onTopoRef     = useRef(onTopology);
  onEventRef.current  = onEvent;
  onSnapRef.current   = onSnapshot;
  onTopoRef.current   = onTopology;

  const clear = useCallback(() => {
    setEvents([]);
    setSnapshots([]);
    setTopology(null);
    latencyBuf.current = [];
    framesRef.current  = 0;
    eventBuffer.current = [];
    snapshotBuf.current = [];
    setFramesRx(0);
    setLatencyMs(null);
  }, []);

  useEffect(() => {
    let ws: WebSocket;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let isMounted = true;

    const flushInterval = setInterval(() => {
      if (!isMounted) return;
      
      if (eventBuffer.current.length > 0) {
        const newEvents = [...eventBuffer.current];
        eventBuffer.current = [];
        setEvents(prev => {
          const next = [...prev, ...newEvents];
          return next.length > maxBuffer ? next.slice(-maxBuffer) : next;
        });
      }

      if (snapshotBuf.current.length > 0) {
        const newSnaps = [...snapshotBuf.current];
        snapshotBuf.current = [];
        setSnapshots(prev => {
          let updated = [...prev];
          newSnaps.forEach(snap => {
            const key = `${snap.layer_name}::${snap.tensor_type}`;
            updated = updated.filter(s => `${s.layer_name}::${s.tensor_type}` !== key);
            updated.push(snap);
          });
          return updated;
        });
      }
    }, 50);

    const connect = () => {
      if (!isMounted) return;
      setStatus('connecting');
      ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => { if (!isMounted) return; setStatus('connected'); };

      ws.onmessage = (msg: MessageEvent<string>) => {
        if (!isMounted) return;
        const recvTs = performance.now();

        let parsed: WireMessage;
        try { parsed = JSON.parse(msg.data) as WireMessage; }
        catch { return; }

        framesRef.current += 1;
        if (framesRef.current % 10 === 0) setFramesRx(framesRef.current);

        if (parsed.type === 'event') {
          const ev: TrainingEvent = { ...parsed, recv_ts: recvTs };
          if (ev.ts_ns > 0) {
            const latency = Date.now() - ev.ts_ns / 1_000_000;
            if (latency > 0 && latency < 5000) {
              latencyBuf.current.push(latency);
              if (latencyBuf.current.length > 30) latencyBuf.current.shift();
              const avg = latencyBuf.current.reduce((a, b) => a + b, 0) / latencyBuf.current.length;
              setLatencyMs(Math.round(avg));
            }
          }
          onEventRef.current?.(ev);
          eventBuffer.current.push(ev); 

        } else if (parsed.type === 'snapshot') {
          const snap: TensorSnapshot = { ...parsed, recv_ts: recvTs };
          onSnapRef.current?.(snap);
          snapshotBuf.current.push(snap);

        } else if (parsed.type === 'topology') {
          const topo: GraphTopology = { ...parsed, recv_ts: recvTs };
          onTopoRef.current?.(topo);
          setTopology(topo);
        }
      };

      ws.onerror  = () => { if (!isMounted) return; setStatus('error'); };
      ws.onclose  = () => {
        if (!isMounted) return;
        setStatus('disconnected');
        reconnectTimer = setTimeout(connect, 2000);
      };
    };

    connect();
    return () => {
      isMounted = false;
      clearInterval(flushInterval);
      if (reconnectTimer) clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, [url, maxBuffer]);

  return { events, snapshots, topology, status, latencyMs, framesReceived: framesRx, clear };
}