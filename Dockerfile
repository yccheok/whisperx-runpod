FROM runpod/pytorch:cuda12

# Set the working directory in the container
WORKDIR /app

# Install ffmpeg, vim
RUN apt-get update && \
    apt-get install -y ffmpeg vim

# Install WhisperX via pip
RUN pip install --upgrade pip && \
    pip install --no-cache-dir runpod==1.7.7 whisperx==3.3.1

# Download large-v3 model
RUN python -c "import whisperx; whisperx.load_model('large-v3', device='cpu', compute_type='int8')"

# Copy source code into image
COPY src src

# -u disables output buffering so logs appear in real-time.
CMD [ "python", "-u", "src/handler.py" ]