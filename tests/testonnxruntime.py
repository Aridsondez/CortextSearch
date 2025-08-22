import onnxruntime as ort

sess = ort.InferenceSession("models/model1.onnx", providers=["CPUExecutionProvider"])
print("Inputs:")
for inp in sess.get_inputs():
    print(inp.name, inp.shape, inp.type)

print("\nOutputs:")
for out in sess.get_outputs():
    print(out.name, out.shape, out.type)



