FROM ghcr.io/torqueagi-aibrain/transformers-pytorch-gpu:2025-10-02

WORKDIR /app

# Install Python dependencies
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy project and install in editable mode
COPY . /app
RUN pip install --no-cache-dir -e .
