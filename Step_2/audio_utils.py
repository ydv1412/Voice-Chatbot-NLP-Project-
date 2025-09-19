import queue, time, wave
import numpy as np
import sounddevice as sd
import webrtcvad
from config import DEBUG
import io
import soundfile as sf
from pydub import AudioSegment

from config import (
    MIC_SAMPLE_RATE, MIC_CHANNELS, MIC_BLOCK_MS,
    VAD_AGGRESSIVENESS, MAX_UTTERANCE_SECONDS
)

FRAME_SAMPLES = int(MIC_SAMPLE_RATE * (MIC_BLOCK_MS / 1000.0))


def record_utterance_wav(out_path: str) -> str:
    """
    VAD-gated recording from default microphone.
    Stops after 500 ms silence following speech or when MAX_UTTERANCE_SECONDS(25s) is reached.
    """
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:  # ignore over/underflows
            pass
        audio_q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=MIC_SAMPLE_RATE,
        channels=MIC_CHANNELS,
        dtype="int16",
        blocksize=FRAME_SAMPLES,
        callback=callback,
    )
    stream.start()

    voiced_frames: list[bytes] = []
    ring: list[bytes] = []
    triggered = False
    silence_ms = 0
    start = time.time()

    print("🎤 Speak now… (auto-stops on silence)")
    try:
        while True:
            chunk = audio_q.get()
            pcm = chunk.tobytes()
            is_speech = vad.is_speech(pcm, MIC_SAMPLE_RATE)

            if not triggered:
                ring.append(pcm)
                if len(ring) > max(1, int(300 / MIC_BLOCK_MS)):  # ~0.3s ring buffer
                    ring.pop(0)
                if is_speech:
                    triggered = True
                    voiced_frames.extend(ring)
                    ring.clear()
            else:
                voiced_frames.append(pcm)
                if is_speech:
                    silence_ms = 0
                else:
                    silence_ms += MIC_BLOCK_MS

            if triggered and silence_ms >= 500:
                break

            if (time.time() - start) >= MAX_UTTERANCE_SECONDS:
                break
    finally:
        stream.stop()
        stream.close()

    # Write raw frames to WAV
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(MIC_CHANNELS)
        wf.setsampwidth(2)  # int16
        wf.setframerate(MIC_SAMPLE_RATE)
        wf.writeframes(b"".join(voiced_frames))

    # Light normalize & save
    seg = AudioSegment.from_wav(out_path)
    seg = seg.apply_gain(-3.0)
    seg.export(out_path, format="wav")
    return out_path


## tried to create an streamlit app (ignore) 
def vad_trim_wav_bytes(wav_bytes: bytes, aggressiveness: int = VAD_AGGRESSIVENESS) -> bytes | None:
    """
    Apply same VAD logic as record_utterance_wav, but on an in-memory WAV
    (recorded via browser Streamlit component).
    Returns new WAV bytes or None if no speech detected.
    """
    import webrtcvad

    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="int16")
    if sr != MIC_SAMPLE_RATE:
        raise ValueError(f"Expected {MIC_SAMPLE_RATE}Hz, got {sr}")
    if y.ndim > 1:
        y = y[:, 0]

    vad = webrtcvad.Vad(aggressiveness)

    frame_len = int(MIC_SAMPLE_RATE * (MIC_BLOCK_MS / 1000.0))
    pcm = y.tobytes()

    frames = [pcm[i:i+frame_len*2] for i in range(0, len(pcm), frame_len*2)]
    voiced = []
    triggered = False
    silence_ms = 0
    voiced_frames = []

    for f in frames:
        is_speech = vad.is_speech(f, MIC_SAMPLE_RATE)
        if not triggered:
            if is_speech:
                triggered = True
                voiced_frames.append(f)
        else:
            voiced_frames.append(f)
            if is_speech:
                silence_ms = 0
            else:
                silence_ms += MIC_BLOCK_MS
        if triggered and silence_ms >= 500:
            break

    if not voiced_frames:
        return None

    buf = io.BytesIO()
    wf = wave.open(buf, "wb")
    wf.setnchannels(MIC_CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(MIC_SAMPLE_RATE)
    wf.writeframes(b"".join(voiced_frames))
    wf.close()
    return buf.getvalue()
