import torch
from transformers import AutoProcessor, AutoModel

# Load model and processor once (lazy loading 가능하도록 개선 여지 있음)
processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
model = AutoModel.from_pretrained("laion/clap-htsat-unfused").to("cuda")
model.eval()

def get_clap_embedding(waveform):
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    inputs = processor(audios=waveform, sampling_rate=48000, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.audio_model(**inputs)
        embedding = outputs.pooler_output.mean(dim=0)
    return embedding.cpu()
