import onnxruntime as ort
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

app = FastAPI()

# Load model using the small ONNX engine
session = ort.InferenceSession("weights/best.onnx")

@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    # 1. Image Preprocessing (Standard YOLO requirement)
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image = np.array(image)
    
    # Resize and normalize
    input_img = cv2.resize(image, (640, 640))
    input_img = input_img.transpose(2, 0, 1) # HWC to CHW
    input_img = input_img.reshape(1, 3, 640, 640).astype(np.float32) / 255.0

    # 2. Inference
    inputs = {session.get_inputs()[0].name: input_img}
    outputs = session.run(None, inputs)
    
    # 3. Return Raw Results
    # You will need a small utility function to decode these 
    # back into boxes [x1, y1, x2, y2]
    return {"raw_output_shape": str(outputs[0].shape)}