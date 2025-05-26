# ASR FastAPI Service using NVIDIA NeMo (MacOS-Compatible)

## 🏗️ Build the ASR Model
1. Install dependencies:

```
python3 -m venv venv
source venv/bin/activate
pip install nemo_toolkit[asr] torch onnx onnxruntime
```

2. Download and convert the model:

```
from nemo.collections.asr.models import EncDecCTCModel
import torch
from omegaconf import OmegaConf
import os
model = EncDecCTCModel.restore_from("https://api.ngc.nvidia.com/v2/resources/nvidia/nemo/stt_hi_conformer_ctc_medium/versions/1/files/stt_hi_conformer_ctc_medium.nemo")
os.makedirs("app/model", exist_ok=True)
OmegaConf.save(config=model.cfg, f="app/model/config.yaml")
input_sample = torch.randn(1, 64, 800)
torch.onnx.export(model, input_sample, "app/model/stt_hi_conformer_ctc_medium.onnx", opset_version=11)
```


## 🚀 Run FastAPI Locally

```
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

```


## 🐳 Docker Build and Run

```
docker build -t asr_fastapi .
docker run -p 8000:8000 asr_fastapi
```


## 🔊 Test Transcription API

```
curl -X POST "http://localhost:8000/transcribe"
-H "accept: application/json"
-H "Content-Type: multipart/form-data"
-F "file=@your_audio.wav"

```


## 📝 Notes
- Audio must be **.wav**, **16kHz**, **5-10 sec**.
- Inference optimized with **ONNX**.
- Real feature extraction from NeMo included.
