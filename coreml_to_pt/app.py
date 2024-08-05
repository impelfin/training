import coremltools as ct
import onnxmltools
from onnx2pytorch import ConvertModel
import torch
import onnx

def convert_coreml_to_pt(coreml_model_path, onnx_model_path, pytorch_model_path):
    # Load Core ML model
    coreml_model = ct.models.MLModel(coreml_model_path)
    
    # Convert Core ML model to ONNX
    onnx_model = onnxmltools.convert_coreml(coreml_model)
    
    # Save ONNX model as protobuf
    onnxmltools.utils.save_model(onnx_model, onnx_model_path)
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_model_path)
    
    # Convert ONNX model to PyTorch model
    pytorch_model = ConvertModel(onnx_model)
    
    # Save PyTorch model
    torch.save(pytorch_model.state_dict(), pytorch_model_path)

# File paths
coreml_model_path = 'sauron_snack.mlmodel'
onnx_model_path = 'sauron_snack.onnx'
pytorch_model_path = 'sauron_snack.pt'

# Run the conversion
convert_coreml_to_pt(coreml_model_path, onnx_model_path, pytorch_model_path)
