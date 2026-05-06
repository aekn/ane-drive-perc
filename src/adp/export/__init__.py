"""Model export CoreML (ANE), ONNX, TorchScript.

for the ANE export step it calls model.deploy which invokes convert_to_deploy
on every reparameterizable module. This is done before tracing, then inspects
the coreml cu assignment to verify all the operations work on NE.
"""
