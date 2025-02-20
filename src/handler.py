import runpod

def run_faster_whisper_job(job):
    # For easy migration, we are following the output format of runpod's 
    # official faster whisper.
    # https://github.com/runpod-workers/worker-faster_whisper/blob/main/src/predict.py#L111
    output = {
        'detected_language' : "",
        'segments' : ""
    }

    return output

runpod.serverless.start({"handler": run_faster_whisper_job})