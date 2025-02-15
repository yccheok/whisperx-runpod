FROM runpod/pytorch:cuda12

# Set the working directory in the container
WORKDIR /app

# Install ffmpeg, vim
RUN apt-get update && \
    apt-get install -y ffmpeg vim

# Install WhisperX via pip
RUN pip install --upgrade pip && pip install --no-cache-dir whisperx

# Download large-v3 model into /app/.cache
RUN python -c "import whisperx; whisperx.load_model('large-v3', device='cpu', compute_type='int8')"

# Copy your Python script into the container
COPY script.py .
COPY audio.mp3 .
RUN python script.py
RUN python script.py

# Set the default command to bash so you get an interactive shell
CMD ["/bin/bash"]
