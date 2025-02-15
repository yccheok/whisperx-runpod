import runpod
import whisperx

model = whisperx.load_model(
    "large-v3", "cuda"
)

def run_whisperx_job(job):
    job_input = job['input']
    url = job_input.get('url', "")

    print(f"🚧 Loading audio from {url}...")
    audio = whisperx.load_audio(url)
    print("✅ Audio loaded")

    print("Transcribing...")
    result = model.transcribe(audio, batch_size=16)
    print("🎉 Transcription done:")
    print(result["segments"])

    return result["segments"]

runpod.serverless.start({"handler": run_whisperx_job})