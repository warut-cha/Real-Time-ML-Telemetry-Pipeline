"""
omnistream.hook
~~~~~~~~~~~~~~~

OmnistreamHook — non-blocking PyTorch telemetry hook (Phase 4).

Two message types over the same ZMQ PUSH socket:
  TAG_EVENT    (0x01) — TrainingEvent scalars, every step
  TAG_SNAPSHOT (0x02) — TensorSnapshot tensors, every N steps

"""
from __future__ import annotations

import queue
import threading
import time
from typing import Optional, List

import zmq
import torch
import torch.nn as nn

from serialise import encode, encode_snapshot


class OmnistreamHook:
    def __init__(
        self,
        zmq_endpoint:    str         = "tcp://127.0.0.1:5555",
        queue_maxsize:   int         = 1000,
        hwm:             int         = 500,
        snapshot_layers: List[str]   = None,
        snapshot_every:  int         = 25,
        snapshot_sample: float       = 1.0,
    ) -> None:
        self._endpoint       = zmq_endpoint
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=queue_maxsize)
        self._hwm            = hwm
        self._thread: Optional[threading.Thread] = None
        self._running        = threading.Event()

        self.snapshot_layers = snapshot_layers or []
        self.snapshot_every  = snapshot_every
        self.snapshot_sample = snapshot_sample

        # Mutable step counter — updated by set_step() each batch.
        # Forward hooks read this so activation snapshots carry the right step.
        self._current_step: int = 0

        self.steps_emitted   = 0
        self.steps_dropped   = 0
        self.snapshots_sent  = 0
        self.send_errors     = 0

    #Lifecycle

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(
            target=self._send_loop, daemon=True, name="omnistream-send")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=timeout)
        self._running.clear()
        total    = self.steps_emitted + self.steps_dropped
        drop_pct = (self.steps_dropped / total * 100) if total > 0 else 0.0
        print(f"[omnistream] stopped emitted={self.steps_emitted} "
              f"dropped={self.steps_dropped} ({drop_pct:.1f}%) "
              f"snapshots={self.snapshots_sent} errors={self.send_errors}")

    def set_step(self, step: int) -> None:
        """
        Update the current step counter.

        Call this at the top of each training iteration so that activation
        forward hooks (which fire during model.forward()) embed the correct
        step number in their TensorSnapshot messages.

        Usage in the training loop:
            hook.set_step(global_step)
            logits = model(x)          # forward hooks fire here, step is correct
            loss.backward()
            ...
            hook.on_step(global_step, ...)
        """
        self._current_step = step

    #Scalar step

    def on_step(
        self,
        step: int, loss: float, accuracy: float, grad_norm: float,
        epoch: int = 0, learning_rate: float = 0.0,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()
        self._enqueue(encode(step, loss, accuracy, grad_norm, timestamp_ns))

    def on_epoch_end(self, epoch: int) -> None:
        print(f"[omnistream] epoch {epoch} emitted={self.steps_emitted} "
              f"dropped={self.steps_dropped} snapshots={self.snapshots_sent}")

    # Tensor snapshot (weights + gradients

    def capture_snapshot(self, model: nn.Module, step: int) -> None:
        """
        Emit weight and gradient snapshots for all matching layers.
        Call after optimizer.step() so gradients are stable.
        """
        if not self.snapshot_layers:
            return
        ts = time.time_ns()
        for name, module in model.named_modules():
            if not any(pat in name for pat in self.snapshot_layers):
                continue
            for pname, param in module.named_parameters(recurse=False):
                full = f"{name}.{pname}"
                # Weight
                self._enqueue(encode_snapshot(
                    step, full, 1,
                    list(param.data.shape),
                    self._sample(param.data),
                    self.snapshot_sample, ts))
                # Gradient (if available)
                if param.grad is not None:
                    self._enqueue(encode_snapshot(
                        step, f"{full}.grad", 2,
                        list(param.grad.shape),
                        self._sample(param.grad),
                        self.snapshot_sample, ts))
        self.snapshots_sent += 1

    # Activation hooks (forward-time capture) 

    def emit_topology(
        self,
        step:         int,
        num_nodes:    int,
        src:          list,
        dst:          list,
        edge_weights: list | None = None,
        node_labels:  list | None = None,
    ) -> None:
        """
        Emit a graph topology message for GNN visualisation.

        Call once before training starts (or whenever the graph structure changes).
        The React GNNGraph component receives this and builds a force-directed layout.

        Parameters
        ----------
        step          Current training step.
        num_nodes     Total number of nodes in the graph.
        src           Edge source indices (list of int).
        dst           Edge destination indices (list of int).
        edge_weights  One float per edge (optional — defaults to all 1.0).
        node_labels   One string per node (optional — defaults to str(index)).

        Example (PyTorch Geometric):
            edge_index = data.edge_index  # shape [2, num_edges]
            hook.emit_topology(
                step      = global_step,
                num_nodes = data.num_nodes,
                src       = edge_index[0].tolist(),
                dst       = edge_index[1].tolist(),
            )
        """
        from serialise import encode_topology
        buf = encode_topology(
            step         = step,
            num_nodes    = num_nodes,
            src          = [int(x) for x in src],
            dst          = [int(x) for x in dst],
            edge_weights = [float(x) for x in (edge_weights or [1.0] * len(src))],
            node_labels  = [str(x) for x in (node_labels  or range(num_nodes))],
        )
        self._enqueue(buf)

    def register_activation_hooks(self, model: nn.Module) -> list:
        """
        Register PyTorch forward hooks on matching layers to capture
        activation tensors during the forward pass.
        """
        handles = []
        for name, module in model.named_modules():
            if not any(pat in name for pat in self.snapshot_layers):
                continue

            def _make_hook(layer_name: str):
                def _hook(mod, inp, output: torch.Tensor) -> None:
                    if not self._running.is_set():
                        return
                    
                    if self._current_step % self.snapshot_every != 0:
                        return
                    
                    step = self._current_step

                    t = output[0:1].detach()
                    
                    self._enqueue(encode_snapshot(
                        step, layer_name, 0,          # 0 = ACTIVATION
                        list(t.shape),
                        self._sample(t),
                        self.snapshot_sample,
                        time.time_ns()))
                return _hook

            handles.append(module.register_forward_hook(_make_hook(name)))

        print(f"[omnistream] registered activation hooks on "
              f"{len(handles)} layer(s): "
              f"{[n for n,_ in model.named_modules() if any(p in n for p in self.snapshot_layers)]}")
        return handles

    # Grad norm utility

    @staticmethod
    def compute_grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads:
            return 0.0
        return torch.norm(
            torch.stack([torch.norm(g.detach(), norm_type) for g in grads]),
            norm_type,
        ).item()

    # Helpers 

    def _sample(self, t: torch.Tensor) -> list:
        flat = t.detach().cpu().float().flatten()
        if self.snapshot_sample >= 1.0 or len(flat) == 0:
            return flat.tolist()
        n   = max(1, int(len(flat) * self.snapshot_sample))
        idx = torch.randperm(len(flat))[:n]
        return flat[idx].tolist()

    def _enqueue(self, buf: bytes) -> None:
        try:
            self._queue.put_nowait(buf)
            self.steps_emitted += 1
        except queue.Full:
            self.steps_dropped += 1

    def _send_loop(self) -> None:
        ctx  = zmq.Context()
        sock = ctx.socket(zmq.PUSH)
        sock.setsockopt(zmq.SNDHWM, self._hwm)
        sock.setsockopt(zmq.SNDTIMEO, 0)
        sock.connect(self._endpoint)
        print(f"[omnistream] connected to {self._endpoint}")
        while True:
            try:
                buf = self._queue.get(timeout=0.1)
            except queue.Empty:
                if not self._running.is_set():
                    break
                continue
            if buf is None:
                break
            try:
                sock.send(buf, zmq.NOBLOCK)
            except zmq.Again:
                self.steps_dropped += 1
            except zmq.ZMQError as e:
                self.send_errors += 1
                print(f"[omnistream] ZMQ error: {e}")
        sock.close()
        ctx.term()
