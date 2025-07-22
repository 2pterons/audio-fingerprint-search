
import os
import faiss
import torch
import pickle
import numpy as np
from src.utils.model_clap import get_clap_embedding
from src.utils.model_wavlm import get_wavlm_embedding
from src.utils.audio_utils import split_audio_torch

INDEX_DIR = "faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)

def embed_and_save(model_name, audio_dir, segment_duration):
    index = faiss.IndexFlatIP(768)
    metadata = []
    # segment_duration = 5 if model_name == "clap" else 10

    audio_list = os.listdir(audio_dir)
    for audio_name in audio_list:
        song_name, _ = os.path.splitext(audio_name)
        audio_path = os.path.join(audio_dir, audio_name)
        segments, sr = split_audio_torch(audio_path, segment_duration, model_name)

        for i, segment in enumerate(segments):
            segment = segment.to("cuda")
            with torch.no_grad():
                emb = get_clap_embedding(segment) if model_name == "clap" else get_wavlm_embedding(segment)
                emb = emb.unsqueeze(0).numpy().astype("float32")
                emb /= np.linalg.norm(emb) + 1e-10  # normalize for cosine
                index.add(emb)

            total_sec = i * segment_duration
            timestamp = f"{total_sec//60:02}:{total_sec%60:02}"
            metadata.append({
                "audio_name": song_name,
                "timestamp": timestamp
            })

    # 저장
    save_path = os.path.join(INDEX_DIR, f"index_{model_name}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"index": index, "metadata": metadata}, f)
    print(f"FAISS index 저장 완료: {save_path}")

def search(model_name, waveform, topk=5):
    # .pt 파일 경로인 경우 로딩
    if isinstance(waveform, str) and waveform.endswith(".pt") and os.path.isfile(waveform):
        waveform = torch.load(waveform)

    # 인덱스 로딩
    index_path = os.path.join(INDEX_DIR, f"index_{model_name}.pkl")
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    index, metadata = data["index"], data["metadata"]

    # 쿼리 임베딩
    with torch.no_grad():
        emb = get_clap_embedding(waveform) if model_name == "clap" else get_wavlm_embedding(waveform)
    emb = emb.unsqueeze(0).numpy().astype("float32")
    emb /= np.linalg.norm(emb) + 1e-10

    D, I = index.search(emb, topk)
    print("\n검색 결과")
    for rank, idx in enumerate(I[0]):
        meta = metadata[idx]
        print(f"\t{rank+1:>2}. {meta['audio_name']} @ {meta['timestamp']}  (sim={D[0][rank]:.4f})")
