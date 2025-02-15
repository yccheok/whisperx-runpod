FROM runpod/pytorch:cuda12

# Set the working directory in the container
WORKDIR /app

# Install ffmpeg, vim
RUN apt-get update && \
    apt-get install -y ffmpeg vim

# Install WhisperX via pip
RUN pip install --upgrade pip && pip install --no-cache-dir whisperx

# Copy your Python script into the container
COPY script.py .
COPY audio.mp3 .
RUN python script.py

# Set the default command to bash so you get an interactive shell
CMD ["/bin/bash"]
