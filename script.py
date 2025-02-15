import os
import whisperx

# Store cached models inside /app/.cache
os.environ["HF_HOME"] = "/app/.cache"

# Set the device; use "cuda" if you have a supported GPU, otherwise "cpu"
device = "cpu"

# Path to your audio file (e.g., WAV or MP3)
audio_file = "audio.mp3"

# Load the WhisperX model using the "large-v3" version and a lighter compute type if needed
print("👉 whisperx.load_model")
model = whisperx.load_model("large-v3", device=device, compute_type="int8")

# Load the audio data
print("👉 whisperx.load_audio")
audio = whisperx.load_audio(audio_file)

# Run the transcription
print("👉 model.transcribe")
result = model.transcribe(audio)

print("👉 result[\"segments\"]")
# Print the transcription segments with timestamps
for segment in result["segments"]:
    print(f"Start: {segment['start']:.2f}s  End: {segment['end']:.2f}s")
    print(f"Text: {segment['text']}\n")