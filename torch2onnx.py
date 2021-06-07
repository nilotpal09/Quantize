import numpy as np
import glob
import os
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn

from collections import OrderedDict
import onnxruntime

from model import MyModel

print(torch.__version__)





model_inf = MyModel(11, [256, 256, 256, 256, 256], [256, 128, 50], num_wp=5)


#************#
# Conversion #
#************#

dummy_node = torch.FloatTensor([[
    [ 0, 44000., 2.2, 0.7, 12, 0,    0,     0,   0, ],
    [ 4, 66000., 1.6, 1.1, 12, 2980, 56000, 1.9, 1, ],
    [ 5, 77000., 1.7, 1.3, 12, 5279, 67000, 2.1, 1.4]]])

model_inf.eval()
torch_out = model_inf(dummy_node)

input_names = ['inp']
output_names = ['output']

model_inf.eval()

onnxmodel_path = os.path.join('model.onnx')

print('Converting...')
torch.onnx.export( \
    model_inf, dummy_node, onnxmodel_path, verbose=False,
    input_names=input_names, output_names=output_names,
    dynamic_axes = {'inp':{0:'B', 1:'N'},
                    'output':{0:'B', 1:'N'}},
    opset_version=11)
print('Done')




#*****************************#
# Verification (torch vs onnx)#
#*****************************#

ort_session = onnxruntime.InferenceSession(onnxmodel_path)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_node)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")