import onnxruntime as ort
face_detector = ort.InferenceSession('model_path', providers=['CPUExecutionProvider'])

image = []

# inference (forward)
input_name = face_detector.get_inputs()[0].name
confidences, boxes = face_detector.run(None, {input_name: image})