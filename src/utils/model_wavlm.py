import torch
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to("cuda")
model.eval()

def get_wavlm_embedding(waveform):
    # if waveform.dim() == 1:
    #     waveform = waveform.unsqueeze(0)

    # inputs = processor(audios=waveform, sampling_rate=48000, return_tensors="pt")
    # inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # with torch.no_grad():
    #     outputs = model.audio_model(**inputs)
    #     embedding = outputs.pooler_output.mean(dim=0)  # [768]
    
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu()
        
    inputs = extractor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.cpu().squeeze(0)
