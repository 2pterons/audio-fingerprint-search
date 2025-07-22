import os
import argparse
import torchaudio
import torch
import numpy as np
from pymilvus import Collection, connections
from src.utils.model_clap import get_clap_embedding
from src.utils.model_wavlm import get_wavlm_embedding
from src.utils.audio_utils import split_audio_torch

def embed_and_insert(user_name, model_name, audio_dir, segment_duration):
    collection_name = f"{user_name}_audio_segments"
    collection = Collection(collection_name)
    collection.load()
    
    audio_list = os.listdir(audio_dir)
    for audio_name in audio_list:
        audio_path = os.path.join(audio_dir, audio_name)
        song_name, _ = os.path.splitext(audio_name)
        segments, sr = split_audio_torch(audio_path, segment_duration, model_name)

        embeddings = []
        for seg in segments:
            seg = seg.to("cuda")
            with torch.no_grad():
                emb = get_clap_embedding(seg) if model_name == "clap" else get_wavlm_embedding(seg)
                embeddings.append(emb)

        segment_ids = list(range(len(embeddings)))
        timestamps = [f"{(i * segment_duration)//60:02}:{(i * segment_duration)%60:02}" for i in segment_ids]
        audio_names = [f"{song_name}_{t.replace(':', 'm')}s.wav" for t in timestamps]
        embedding_array = torch.stack(embeddings).numpy()

        insert_data = [segment_ids, embedding_array.tolist(), audio_names, timestamps]
        collection.insert(insert_data)
        print(f"저장 완료: {audio_name}")

def search(user_name, model_name, query_path, topk=5):
    waveform, sr = torchaudio.load(query_path)
    waveform = waveform.mean(dim=0, keepdim=True)
    target_sr = 48000 if model_name == "clap" else 16000
    waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    with torch.no_grad():
        embedding = get_clap_embedding(waveform) if model_name == "clap" else get_wavlm_embedding(waveform)
    query_vector = embedding.unsqueeze(0).numpy().astype("float32")

    collection_name = f"{user_name}_audio_segments"
    collection = Collection(collection_name)
    collection.load()

    results = collection.search(
        data=query_vector,
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=topk,
        output_fields=["segment_id", "audio_name", "timestamp"]
    )

    print(f"검색 결과 for {query_path}")
    for result in results[0]:
        print(f"{result.entity.audio_name} @ {result.entity.timestamp}s - distance={result.distance:.4f}")

