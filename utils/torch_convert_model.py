import torch
import torch.onnx

device = torch.device("cpu")
model = torch.load("model.uu", map_location='cpu')
model.to(device)
model.eval()

dummy = torch.ones((1, 3, 224, 224))

torch.onnx.export(model,
                  dummy,
                  "skin_cancer_detector.onnx",
                  export_params=True,
                  do_constant_folding=True,
                  opset_version=9,
                  verbose=True)

dummy = torch.ones((1, 3, 224, 224))

scripted_model = torch.jit.trace(model, dummy)
scripted_model.save('skin_cancer_detector.pth')

