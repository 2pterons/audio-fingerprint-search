import os
import argparse
import torchaudio
import torch
from src.core import faiss_ops
from src.utils.audio_utils import split_audio_pydub

def main():
    parser = argparse.ArgumentParser(description="Audio Embedding Search Controller")

    parser.add_argument("--model", choices=["clap", "wavlm"], required=True)
    parser.add_argument("--task", choices=["init", "embed", "split", "search"], required=True)
    parser.add_argument("--audio_path", type=str, help="Path to single audio file")
    parser.add_argument("--audio_dir", type=str, help="Directory with audio files")
    parser.add_argument("--query_path", type=str, help="Path to query segment")
    parser.add_argument("--save_query", action="store_true", help="Save query waveform as .pt file")
    parser.add_argument("--segment_duration", type=int, default=5)

    args = parser.parse_args()

    if args.task == "embed":
        if not args.audio_dir:
            raise ValueError("`--audio_dir` is required for embedding")
        faiss_ops.embed_and_save(args.model, args.audio_dir, args.segment_duration)

    elif args.task == "split":
        if not args.audio_path:
            raise ValueError("`--audio_path` is required for splitting")
        split_audio_pydub(input_path=args.audio_path, segment_duration=args.segment_duration)

    elif args.task == "search":
        if not args.query_path:
            raise ValueError("`--query_path` is required for search")
        
        waveform = args.query_path

        if args.query_path.endswith(".wav"):
            waveform, sr = torchaudio.load(args.query_path)
            waveform = waveform.mean(dim=0, keepdim=True)
            target_sr = 48000 if args.model == "clap" else 16000
            if sr != target_sr:
                waveform = torchaudio.functional.resample(waveform, sr, target_sr)

            # 임시 torch 파일로 저장 (faiss_ops가 torch.load 사용 중)
            if args.save_query:
                os.makedirs("queries", exist_ok=True)
                base_name = os.path.splitext(os.path.basename(args.query_path))[0]
                save_path = os.path.join("queries", f"{base_name}.pt")
                torch.save(waveform, save_path)
                print(f"쿼리 waveform 저장됨: {save_path}")

        faiss_ops.search(args.model, waveform)

if __name__ == "__main__":
    main()
