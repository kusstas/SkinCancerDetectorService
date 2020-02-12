import torch
import torch.onnx

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("./model")
model.to(device)
model.eval()

dummy = torch.ones((1, 3, 224, 224)).cuda()

torch.onnx.export(model,
                  dummy,
                  "skin_cancer_detector.onnx",
                  export_params=True,
                  do_constant_folding=True,
                  opset_version=9,
                  verbose=True)

