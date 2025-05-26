from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import io
import numpy as np
import soundfile as sf
import onnxruntime as ort
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.utils.decoder_torch import GreedyCTCDecoder
from omegaconf import OmegaConf

app = FastAPI()

# Load ONNX model
onnx_model_path = "app/model/stt_hi_conformer_ctc_medium.onnx"
onnx_session = ort.InferenceSession(onnx_model_path)

# Load NeMo config for preprocessor and decoder
asr_config = OmegaConf.load("app/model/config.yaml")
featurizer = WaveformFeaturizer(sample_rate=16000, int_values=False, augmentor=None, pad_align=8)
decoder = GreedyCTCDecoder(asr_config.decoder.vocabulary)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .wav file.")
    
    # Read audio
    audio_bytes = await file.read()
    audio_np, sample_rate = sf.read(io.BytesIO(audio_bytes))
    if sample_rate != 16000:
        raise HTTPException(status_code=400, detail="Audio must be at 16kHz.")
    duration = len(audio_np) / sample_rate
    if not (5 <= duration <= 10):
        raise HTTPException(status_code=400, detail="Audio duration must be 5-10 seconds.")
    
    # Extract features (Mel spectrograms)
    processed_signal, processed_length = featurizer.process(audio_signal=audio_np, sampling_rate=sample_rate)
    processed_signal = processed_signal.unsqueeze(0).cpu().numpy()  # shape: (1, feats, time)
    
    # Inference
    outputs = onnx_session.run(None, {"input": processed_signal})
    logits = outputs[0]  # (batch, time, vocab)
    
    # Decoding (greedy CTC)
    pred_ids = np.argmax(logits, axis=-1)
    transcription = decoder(torch.tensor(pred_ids))
    
    return JSONResponse(content={"transcription": transcription})

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
