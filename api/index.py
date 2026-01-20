from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()

# Load the model once when the server starts
# use 'weights/best.onnx'
session = ort.InferenceSession("weights/best.onnx")

@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    # 1. Load and Preprocess Image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = np.array(image)
    image_resized = cv2.resize(image, (640, 640))
    input_data = np.transpose(image_resized, (2, 0, 1)).astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    # 2. Run Inference
    outputs = session.run(None, {session.get_inputs()[0].name: input_data})
    
    # 3. Process outputs (simplified)
    # This part depends on your specific YOLO output shape
    # Typically returns [x, y, w, h, confidence]
    return {"detections": str(outputs[0].shape)}