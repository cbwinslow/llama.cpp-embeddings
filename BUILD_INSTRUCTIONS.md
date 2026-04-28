# llama.cpp Embeddings Server Build Instructions

This repository contains a customized build of llama.cpp optimized for embedding models on NVIDIA Tesla K80 and K40m GPUs.

## System Requirements
- NVIDIA Tesla K80 and/or K40m GPUs
- Ubuntu 24.04 or similar Linux distribution
- CUDA Toolkit 11.4
- GCC 10 (for compatibility with CUDA 11.4)

## Build Instructions

### 1. Install Dependencies
```bash
# Install CUDA 11.4 (already installed in this environment)
# Install build dependencies
sudo apt-get install -y build-essential cmake git libssl-dev

# Install GCC 10 for CUDA 11.4 compatibility
sudo apt-get install -y gcc-10 g++-10
```

### 2. Build llama.cpp
```bash
# Configure build with CUDA support for K80 (compute capability 3.7) and K40m (compute capability 3.5)
CC=gcc-10 CXX=g++-10 cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.4/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES="35;37"

# Build
cmake --build build --config Release -j 8
```

### 3. Download Embedding Model
```bash
# Create embeddings directory if it doesn't exist
mkdir -p models/embeddings

# Download nomic-embed-text model (768-dimensional embeddings)
wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf -O models/embeddings/nomic-embed-text-v1.5.Q4_K_M.gguf
```

### 4. Run the Embedding Server
```bash
# Start the server (adjust port as needed)
./build/bin/llama-server -m ./models/embeddings/nomic-embed-text-v1.5.Q4_K_M.gguf --embedding -ngl 999 --port 8082
```

### 5. Test the Server
```bash
curl -X POST http://localhost:8082/embedding \
  -H "Content-Type: application/json" \
  -d '{"input": ["Your test text here"]}'
```

## Performance Notes
- The server uses all three GPUs (1x K40m + 2x K80) with tensor splitting
- Each GPU loads approximately 1/3 of the model
- CUDA graphs are enabled for optimal performance
- The model is quantized to Q4_K_M for balance of size and accuracy

## Model Details
- Architecture: nomic-bert
- Embedding dimension: 768
- Context length: 2048
- Vocabulary size: 30,522
- Quantization: Q4_K_M (4-bit)
- File size: ~81MB
