import torch
from torch import nn
from nemo.collections.asr.models import EncDecCTCModelBPE

# Load the NeMo model
nemo_model = EncDecCTCModelBPE.restore_from("/Users/neelmani/Desktop/asr_fastapi_app/app/stt_hi_conformer_ctc_medium.nemo")

# Define a wrapper class to use with torch.onnx.export
class WrappedModel(nn.Module):
    def __init__(self, nemo_model):
        super(WrappedModel, self).__init__()
        self.nemo_model = nemo_model

    def forward(self, audio_signal, length):
        # Update argument names to match NeMo's expectations
        return self.nemo_model(input_signal=audio_signal, input_signal_length=length)

# Instantiate the wrapped model
wrapped_model = WrappedModel(nemo_model)

# Example dummy inputs to trace the model
# NOTE: You need to ensure the dimensions match your actual use case
dummy_audio_signal = torch.randn(1, 16000)  # (batch_size=1, num_samples=16000)
dummy_length = torch.tensor([16000])        # (batch_size=1), length in samples

# Export to ONNX
torch.onnx.export(
    wrapped_model, 
    (dummy_audio_signal, dummy_length),
    "/Users/neelmani/Desktop/asr_fastapi_app/app/stt_hi_conformer_ctc_medium.onnx",
    input_names=['input_signal', 'input_signal_length'],
    output_names=['log_probs'],
    dynamic_axes={
        'input_signal': {1: 'num_samples'},
        'input_signal_length': {0: 'batch_size'},
        'log_probs': {0: 'batch_size', 1: 'time'}
    },
    opset_version=17  # ← Changed from 13 to 17
)


print("ONNX model has been exported successfully!")
