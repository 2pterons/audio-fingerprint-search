import os
import math
import torch
import torchaudio
from pydub import AudioSegment

def split_audio_torch(file_path, segment_length=5, model_name="clap"):
    # try:
    #     torchaudio.set_audio_backend("sox_io")
    # except Exception as e:
    #     print("백엔드 설정 실패:", e)
        
    if file_path.endswith(".mp3"):
        wav_path = os.path.splitext(file_path)[0] + ".wav"
        audio = AudioSegment.from_mp3(file_path)
        audio.export(wav_path, format="wav")
        print(f"변환됨: {file_path} => {wav_path}")
        file_path = wav_path

    target_sr = 48000 if model_name == "clap" else 16000
    try:
        waveform, sr = torchaudio.load(file_path)
    except:
        waveform, sr = sf.read(file_path, dtype="float32")
        

    # Resample
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    segment_samples = segment_length * target_sr
    segments = []
    for start in range(0, waveform.shape[1], segment_samples):
        end = start + segment_samples
        segment = waveform[:, start:end]

        if segment.shape[1] < segment_samples:
            padded = torch.zeros((1, segment_samples), dtype=segment.dtype)
            padded[:, :segment.shape[1]] = segment
            segment = padded

        segments.append(segment)

    return segments, target_sr

def split_audio_pydub(input_path, output_dir="./audio_segments", segment_duration=3):
    os.makedirs(output_dir, exist_ok=True)
    audio_name = os.path.basename(input_path)
    audio = AudioSegment.from_file(input_path)
    duration_ms = len(audio)
    seg_len_ms = segment_duration * 1000

    num_segments = math.ceil(duration_ms / seg_len_ms)
    print(f"{audio_name} → 총 {num_segments}개 segment 생성 중...")

    output_paths = []

    for i in range(num_segments):
        start_ms = i * seg_len_ms
        end_ms = min((i + 1) * seg_len_ms, duration_ms)
        segment = audio[start_ms:end_ms]

        start_sec = start_ms // 1000
        minutes = start_sec // 60
        seconds = start_sec % 60
        time_str = f"{minutes:02d}m{seconds:02d}s"

        output_name = f"{os.path.splitext(audio_name)[0]}_{i*segment_duration:03.0f}_{time_str}.wav"
        output_path = os.path.join(output_dir, output_name)
        segment.export(output_path, format="wav")
        output_paths.append(output_path)
        print(f"저장됨: {output_path}")

    return output_paths