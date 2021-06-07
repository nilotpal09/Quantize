import torch
import onnx
from quantize import quantize, QuantizationMode

import onnxruntime

print(torch.__version__)



dummy_node = torch.FloatTensor([[
    [ 0, 44000., 2.2, 0.7, 12, 0,    0,     0,   0, ],
    [ 4, 66000., 1.6, 1.1, 12, 2980, 56000, 1.9, 1, ],
    [ 5, 77000., 1.7, 1.3, 12, 5279, 67000, 2.1, 1.4]]])



#**************#
# Quantization #
#**************#


input_path = 'model.onnx'
output_path = 'model_quant.onnx'

# Load the onnx model
model = onnx.load(input_path)
# Quantize
quantized_model = quantize(model, quantization_mode=QuantizationMode.IntegerOps)
# Save the quantized model
onnx.save(quantized_model, output_path)






#*****************************#
# Verification (quantization) #
#*****************************#

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()




onnxmodel_path_gen = 'model.onnx'
onnxmodel_path_quantized = 'model_quant.onnx'

ort_session = onnxruntime.InferenceSession(onnxmodel_path_gen)

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_node)}
ort_outs_gen = ort_session.run(None, ort_inputs)


ort_session = onnxruntime.InferenceSession(onnxmodel_path_quantized)

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_node)}
ort_outs_quantized = ort_session.run(None, ort_inputs)



# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(ort_outs_gen[0], ort_outs_quantized[0], rtol=1e-03, atol=1e-05)

print("Quantized model is tested against the original model, and the result looks good!")