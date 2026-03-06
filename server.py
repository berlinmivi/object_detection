import os

from utils import pip_install
try:
    import cv2
except ImportError:
    pip_install("opencv-python")
    import cv2

try:
    import cvlib
    import cvlib.object_detection
except ImportError:
    pip_install("cvlib")
    pip_install("setuptools<82")
    pip_install("tensorflow")
    import cvlib
    import cvlib.object_detection

cvlib.object_detection.dest_dir = os.path.join(os.getcwd(), ".cvlib", "object_detection")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from cvlib.object_detection import draw_bbox

dir_name = "images_uploaded"

if not os.path.exists(dir_name):
    os.mkdir(dir_name)

import io
import os
try:
    import uvicorn
except ImportError:
    pip_install("uvicorn")
    import uvicorn
import numpy as np

from enum import Enum
try:
    from fastapi import FastAPI, UploadFile, HTTPException, File, Query
    from fastapi.responses import StreamingResponse
except ImportError:
    pip_install("fastapi")
    from fastapi import FastAPI, File, UploadFile, HTTPException, Query
    from fastapi.responses import StreamingResponse 

#instance of FastAPI class for defining the server endpoints
app = FastAPI(title = "Deploying Image Detection model using FastAPI")

class Model(Enum):
    #immutable name value pairs
    # name = value
    yolov3tiny = "yolov3-tiny"
    yolov3 = "yolov3"

@app.get('/')
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://serve/docs"

pip_install("python-multipart")

# This endpoint handles all the logic necessary for the object detection to work
# Needs the model and the image on which to perform the detection
@app.post('/predict')
def prediction(model: Model = Query(...), file: UploadFile = File(...)):

    #1. Validate Input File
    filename = file.filename
    fileExtension = filename.split('.')[-1] in ('jpg', 'jpeg', 'png', )
    if not fileExtension:
        raise HTTPException(status_code = 415, detail = "Unsupported file format.")

    #2. Transform raw image into CV2 image
    # Read image as a stream of bytes
    # file.file.read() -> reads raw binary content
    # io.BytesIO() -> wraps those raw bytes into an in-memory binary stream (virtual file in RAM)
    # seek(0) -> after reading data into BytesIO the internal cursor/pointer is at the end of the stream resets 
    # the pointer back to the beginning of the stream
    # image_stream.read() -> reads all bytes from the stream
    # bytearray() -> converts it into mutable sequence of bytes
    # np.asarray(, dtype = uint8) converts the byte sequence into a numpy array of unsigned 8-bit integers which
    # is the format OpenCV expects for raw image data

    image_stream = io.BytesIO(file.file.read())
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype = np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    #3. Run Object detection model
    # Pass the string value of the Enum
    bbox, label, conf = cvlib.detect_common_objects(image, model = model.value)

    output_image = draw_bbox(image, bbox, label, conf)

    cv2.imwrite(f"images_uploaded/{filename}", output_image)

    #4. Open the image in binary format for streaming
    file_image = open(f"images_uploaded/{filename}",'rb')

    return StreamingResponse(file_image, media_type = "image/jpeg")


uvicorn.run(app, host = "0.0.0.0", port = 8000)