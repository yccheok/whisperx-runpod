import runpod

def run_whisperx_job(job):
    return "this is whisperx"

runpod.serverless.start({"handler": run_whisperx_job})