# Object Detection API using FastAPI and YOLO

This project provides a robust FastAPI-based server that performs real-time object detection on uploaded images using the `cvlib` library (YOLOv3 / YOLOv3-tiny).

## 🚀 Key Features

- **FastAPI Backend**: High-performance asynchronous API endpoints.
- **YOLO Detection**: Uses `cvlib` and OpenCV DNN module for accurate object detection.
- **Dynamic Model Selection**: Choose between standard `yolov3` and the faster `yolov3-tiny`.

## 📋 Prerequisites

- **Python**: 3.12 (Optimized for Apple Silicon / M-series Macs)
- **OpenCV**: 4.10.0+ (Tested with 4.13.0.92)
- **Numpy**: < 2.0.0 (Required for `cvlib` compatibility)

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/berlinmivi/object_detection.git
   cd object_detection
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv prod_venv
   source prod_venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📂 Project Structure

- `server.py`: The main FastAPI application and logic.
- `utils.py`: Utility functions for dependency management.
- `.cvlib/`: Local storage for YOLO weights and configuration files.
- `images_uploaded/`: Directory where processed images are saved.


## 🚦 Usage

### Running the Server
```bash
python server.py
```
The server will start at `http://0.0.0.0:8000`.

### API Documentation
Once the server is running, head over to:
- Interactive Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Alternative Docs: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Making a Prediction
Use the `/predict` endpoint to upload an image and select your model.
- **Method**: `POST`
- **Params**: `model` (e.g., `yolov3-tiny`)
- **Body**: `file` (Upload your image)

## 📄 License
This project is for educational and production ML deployment demonstration.
