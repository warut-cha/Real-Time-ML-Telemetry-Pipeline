# Real Time ML Telemetry Pipeline

Real Time ML Telemetry Pipeline is a high-performance, real-time observability engine for Machine Learning. Traditional tools like TensorBoard are built for post-training analysis. Omnistream acts as a live, 60-FPS "security camera" for your PyTorch models, streaming weights, activations, and gradients directly from the GPU to a hardware-accelerated dashboard without choking your training loop.

## Features

* **Sub-Millisecond Latency:** Bypasses slow HTTP requests by utilizing **ZeroMQ** for a non-blocking, asynchronous pipeline.
* **Zero Training Penalty:** PyTorch forward hooks utilize "batch slicers" and step-throttles to extract micro-samples, dropping telemetry latency from 2500ms down to ~100ms per step.
* **Hardware-Level Memory Mapping:** The C++ backend uses OS-level `mmap` to write live training "tapes" directly to disk, allowing for zero-downtime hot-swapping and historical rewinding.
* **60+ FPS Canvas Engine:** HTML5 Canvas engine, visualizing thousands of edges and neurons at native monitor refresh rates.

---

##  Architecture Overview

The telemetry operates on a decoupled, 3-tier Producer-Consumer architecture:

1. **The PyTorch Hook (`hook.py`):** Sits inside the training loop. Captures telemetry and tensor snapshots, serializes them via a custom binary format, and fires them over a non-blocking ZMQ socket.
2. **The C++ Engine (`engine/`):** The high-speed middleman. Ingests the ZMQ stream, writes tapes to disk via `mmap`, and broadcasts state to the frontend via a `cpp-httplib` WebSocket server.
3. **The React Dashboard (`ui/`):** A frontend that buffers incoming WebSocket traffic to prevent React state firehoses, rendering the neural network topology cleanly using High-DPI Canvas math.

---

## Tech Stack

* **AI Hook:** Python 3.10+, PyTorch
* **Stream Engine:** C++17, ZeroMQ, `cpp-httplib`, CMake
* **Dashboard:** React, TypeScript, HTML5 Canvas

---

## Getting Started (Local Development)

### Prerequisites
* **C++ Compiler** supporting C++17 (GCC, Clang, or MSVC)
* **CMake** (v3.15+)
* **ZeroMQ** (`libzmq` installed on your system)
* **Node.js** (v18+)
* **Python** (v3.10+) with PyTorch

### 0. Clone the Repository
```bash
git clone [https://github.com/warut-cha/Real-Time-ML-Telemetry-Pipeline.git](https://github.com/warut-cha/Real-Time-ML-Telemetry-Pipeline.git)
cd Real-Time-ML-Telemetry-Pipeline 
```

### 1. Build the C++ Engine
```bash
cd engine
cmake -B build
cmake --build build --config Realease
```

### 2. Start the Engine
```bash
# Windows
./engine/build/Release/engine.exe

# Linux / Mac
./engine/build/engine
```

### 3. Launch the React Dashboard
Open a new terminal and start fronend server
```bash
cd ui
npm install
npm run dev
```
Open your browser to (`http://localhost:3000`)

### 4. Run the Traning Script
Open a new terminal and execute the sample PyTorch training script (currently only CNN and GNN is avaiable). The hook will automatically connect to C++ engine.
```bash
pyton train_mnist.py #For CNN
python train_gnn.py #For GNN
```
