import os
import time
import whisperx

# Store cached models inside /app/.cache
os.environ["HF_HOME"] = "/app/.cache"

# Function to print the contents of the cache directory
def print_cache_contents():
    cache_dir = "/app/.cache"
    if os.path.exists(cache_dir):
        print("\n📂 Contents of /app/.cache:")
        for root, dirs, files in os.walk(cache_dir):
            for name in files:
                file_path = os.path.join(root, name)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                print(f"📄 {file_path} ({file_size:.2f} MB)")
    else:
        print("\n🚫 /app/.cache directory does not exist")

# Print cache contents before loading the model
print_cache_contents()

# Set the device; use "cuda" if you have a supported GPU, otherwise "cpu"
device = "cpu"

# Path to your audio file (e.g., WAV or MP3)
audio_file = "audio.mp3"

# Load the WhisperX model using the "large-v3" version and a lighter compute type if needed
print("👉 whisperx.load_model")
start_time = time.time()
model = whisperx.load_model("large-v3", device=device, compute_type="int8")
end_time = time.time()
print(f"✅ Model loaded in {end_time - start_time:.2f} seconds")

# Print cache contents after loading the model
print_cache_contents()

# Load the audio data
print("👉 whisperx.load_audio")
start_time = time.time()
audio = whisperx.load_audio(audio_file)
end_time = time.time()
print(f"✅ Audio loaded in {end_time - start_time:.2f} seconds")

# Run the transcription
print("👉 model.transcribe")
start_time = time.time()
result = model.transcribe(audio)
end_time = time.time()
print(f"✅ Transcribe done in {end_time - start_time:.2f} seconds")

print("👉 result[\"segments\"]")
# Print the transcription segments with timestamps
for segment in result["segments"]:
    print(f"Start: {segment['start']:.2f}s  End: {segment['end']:.2f}s")
    print(f"Text: {segment['text']}\n")