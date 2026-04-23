"""
omnistream.serialise

Wire serialisation for both message types.

Every ZMQ frame starts with a 1-byte type discriminator so the C++ receiver
can route without probing the FlatBuffer vtable:
  0x01  TrainingEvent  (scalars — emitted every step)
  0x02  TensorSnapshot (tensor data — emitted every N steps)
"""
from __future__ import annotations
import time
import struct
import flatbuffers
import flatbuffers
from generated.omnistream import TrainingEvent
from generated.omnistream import TensorSnapshot
from generated.omnistream import AnomalyMarker

TAG_EVENT    = b'\x01'
TAG_SNAPSHOT = b'\x02'

_INITIAL_BUILDER_SIZE = 64


def encode(step, loss, accuracy, grad_norm, timestamp_ns):
    builder = flatbuffers.Builder(1024)
    
    # 1. Start the event
    TrainingEvent.Start(builder)
    
    # 2. Add the fields one by one
    TrainingEvent.AddStep(builder, int(step))
    TrainingEvent.AddLoss(builder, float(loss))
    TrainingEvent.AddAccuracy(builder, float(accuracy))
    TrainingEvent.AddGradNorm(builder, float(grad_norm))
    TrainingEvent.AddTimestampNs(builder, int(timestamp_ns))
    
    # 3. End the event and finish the buffer
    event = TrainingEvent.End(builder)
    builder.Finish(event)
    
    return builder.Output()

def encode_snapshot(
    step:         int,
    layer_name:   str,
    tensor_type:  int,   # 0=ACTIVATION 1=WEIGHT 2=GRADIENT 3=NODE_EMBEDDING
    shape:        list[int],
    values:       list[float],
    sample_rate:  float = 1.0,
    timestamp_ns: int | None = None,
) -> bytes:
    """
    Encode a TensorSnapshot as TAG_SNAPSHOT + FlatBuffer bytes.

    Uses the numpy-backed FlatBuffers vector builder for speed —
    avoids Python-level iteration over potentially large value arrays.
    """
    if timestamp_ns is None:
        timestamp_ns = time.time_ns()

    import numpy as np
    builder = flatbuffers.Builder(256 + len(values) * 4)

    shape_vec  = builder.CreateNumpyVector(np.array(shape,  dtype='uint32'))
    values_vec = builder.CreateNumpyVector(np.array(values, dtype='float32'))
    name_str   = builder.CreateString(layer_name)

    # Field order must match phase4_generated.h VTableOffset constants:
    # VT_STEP=4, VT_LAYER_NAME=6, VT_TENSOR_TYPE=8, VT_SHAPE=10,
    # VT_VALUES=12, VT_SAMPLE_RATE=14, VT_TIMESTAMP_NS=16
    builder.StartObject(7)
    builder.PrependUint32Slot (0, step,         0)
    builder.PrependUOffsetTRelativeSlot(1, name_str,   0)
    builder.PrependInt8Slot   (2, tensor_type,  0)
    builder.PrependUOffsetTRelativeSlot(3, shape_vec,  0)
    builder.PrependUOffsetTRelativeSlot(4, values_vec, 0)
    builder.PrependFloat32Slot(5, sample_rate,  1.0)
    builder.PrependUint64Slot (6, timestamp_ns, 0)
    root = builder.EndObject()
    builder.Finish(root)
    return TAG_SNAPSHOT + bytes(builder.Output())


TAG_TOPOLOGY = b'\x03'


def encode_topology(
    step:         int,
    num_nodes:    int,
    src:          list[int],
    dst:          list[int],
    edge_weights: list[float],
    node_labels:  list[str],
    timestamp_ns: int | None = None,
) -> bytes:
    """
    Encode a GraphTopology message as TAG_TOPOLOGY + JSON bytes.

    Uses JSON (not FlatBuffers) because graph topology is sent once or rarely —
    the serialisation cost is negligible compared to the per-step tensor work.

    Wire format: TAG_TOPOLOGY (1 byte) + UTF-8 JSON
    C++ engine routes tag 0x03 to a new to_json() path and broadcasts as
    {"type":"topology", ...}
    """
    import json, time
    payload = json.dumps({
        'type':         'topology',
        'step':         step,
        'num_nodes':    num_nodes,
        'src':          src,
        'dst':          dst,
        'edge_weights': edge_weights,
        'node_labels':  node_labels,
        'ts_ns':        timestamp_ns if timestamp_ns is not None else time.time_ns(),
    }, separators=(',', ':'))
    return TAG_TOPOLOGY + payload.encode('utf-8')
