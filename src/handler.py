import runpod
from runpod.serverless.utils import download_files_from_urls, rp_cleanup
import time
from faster_whisper import WhisperModel

# Load Faster-Whisper model
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

def run_faster_whisper_job(job):
    start_time = time.time()
    
    job_input = job['input']
    url = job_input.get('url', "")

    print(f"🚧 Downloading audio from {url}...")
    audio_path = download_files_from_urls(job['id'], [job_input['audio']])[0]
    print("✅ Audio downloaded")
    
    print("Transcribing...")
    segments, info = model.transcribe(audio_path, beam_size=5)
    
    output_segments = []
    for segment in segments:
        output_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })
    
    end_time = time.time()
    time_s = (end_time - start_time)
    print(f"🎉 Transcription done: {time_s:.2f} s")
    
    output = {
        'detected_language': info.language,
        'segments': output_segments
    }
    
    rp_cleanup(['input_objects'])

    return output

runpod.serverless.start({"handler": run_faster_whisper_job})
