#!/bin/bash

# Create models directory if it doesn't exist
mkdir -p ~/.ollama/models

# Download the model
echo "Downloading Qwen3-7B-Instruct model..."
curl -L https://huggingface.co/Qwen/Qwen3-7B-Instruct-GGUF/resolve/main/qwen3-7b-instruct.Q4_K_M.gguf -o ~/.ollama/models/qwen3-7b-instruct.gguf

# Create the model in OLLAMA
echo "Creating OLLAMA model..."
ollama create qwen3-7b-instruct -f Qwen3-7B-Instruct.modelfile

echo "Setup complete! You can now use the model with: ollama run qwen3-7b-instruct" 