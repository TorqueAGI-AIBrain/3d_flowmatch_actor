FROM ghcr.io/torqueagi-aibrain/transformers-pytorch-gpu:2025-10-02

WORKDIR /app

# Install project dependencies
RUN pip install --no-cache-dir \
    einops tqdm 'zarr<3' numcodecs diffusers kornia \
    tensorboard pyyaml scipy matplotlib mcap 'rosbags==0.9.23' \
    gradio plotly \
    git+https://github.com/openai/CLIP.git

# Copy project
COPY . /app

# Install project in editable mode
RUN pip install --no-cache-dir -e .
