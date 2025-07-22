import math
import librosa
import hashlib
import numpy as np
import soundfile as sf  # 설치: pip install soundfile
from pydub import AudioSegment

class FingerPrint:
    @staticmethod
    def split_audio(file_path, segment_duration_ms=10000):
        audio = AudioSegment.from_file(file_path)
        segments = []
        num_segments = math.ceil(len(audio) / segment_duration_ms)
        for i in range(num_segments):
            start_ms = i * segment_duration_ms
            segment = audio[start_ms:start_ms+segment_duration_ms]
            segments.append((segment, start_ms/1000.0))
        return segments

    @staticmethod
    def audiosegment_to_np(audio_segment):
        samples = audio_segment.get_array_of_samples()
        np_audio = np.array(samples).astype(np.float32)
        return np_audio / (2**15)
    
    @staticmethod
    def extract_peaks_from_audio_data(audio_data, sr=22050):
        S = librosa.stft(audio_data, n_fft=2048, hop_length=512)
        magnitude = np.abs(S)
        peaks = np.argwhere(magnitude > np.percentile(magnitude, 99))
        return peaks, S

    @staticmethod
    def hash_peaks(peaks, fan_value=5):
        hashes = []
        for i in range(len(peaks)):
            for j in range(1, fan_value):
                if i + j < len(peaks):
                    freq1, time1 = peaks[i]
                    freq2, time2 = peaks[i+j]
                    t_delta = time2 - time1
                    h = hashlib.sha1(f"{freq1}|{freq2}|{t_delta}".encode()).hexdigest()
                    hashes.append(h)
        return hashes

    @staticmethod
    def compute_fingerprint(audio_data, sr=22050):
        peaks, S = FingerPrint.extract_peaks_from_audio_data(audio_data, sr=sr)
        h_list = FingerPrint.hash_peaks(peaks)
        return set(h_list)

    @staticmethod
    def build_fingerprint_db(file_paths, segment_duration_ms=10000, sr=22050):
        db = {}
        for path in file_paths:
            print(f"파일명: {path}")
            segments = FingerPrint.split_audio(path, segment_duration_ms)
            print(f"총 {len(segments)}개의 세그먼트 생성됨")

            seg_data = []
            for seg, start_time in segments:
                np_audio = FingerPrint.audiosegment_to_np(seg)
                fp = FingerPrint.compute_fingerprint(np_audio, sr=sr)
                # print(f"세그먼트 0의 fingerprint 개수: {len(fp)}")
                seg_data.append((start_time, fp))
            db[path] = seg_data
        return db

    @staticmethod
    def extract_segment(audio_path, start_time, duration, sr=22050, output_path=None):
        y, _ = librosa.load(audio_path, sr=sr, offset=start_time, duration=duration)
        if output_path:
            sf.write(output_path, y, sr)
        return y, sr
    
    @staticmethod
    def search_query(query_audio_path, fp_db, segment_duration_ms=10000, sr=22050):
        y, _ = librosa.load(query_audio_path, sr=sr)
        query_fp = FingerPrint.compute_fingerprint(y, sr=sr)
        print(f"[QUERY] fingerprint 개수: {len(query_fp)}")

        best_match = None
        best_score = 0
        best_segment_info = None
        
        for file, segments in fp_db.items():
            for start_time, seg_fp in segments:
                score = len(query_fp.intersection(seg_fp))
                print(f"[{file}] {start_time:.1f}s | fp 수: {len(seg_fp)} | 점수: {score}")
                # if score > best_score:
                if score > best_score and score > 0:
                    best_score = score
                    best_match = file
                    best_segment_info = start_time
        return best_match, best_segment_info, best_score


class WavLM:
    pass

class Clap:
    pass