# Setup Instructions

## Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/video-detection-system.git
cd video-detection-system
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download model files:**
```bash
# For intrusion detection (person detection)
cd intrusion-main/intrusion-main/
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt

# For PPE detection, place your trained model at:
# ppe-detection-main/ppe-detection-main/best.pt
```

4. **Configure your stream URL:**
Edit the `MJPEG_URL` in both scripts to match your Raspberry Pi's IP:
```python
MJPEG_URL = "http://YOUR_PI_IP:8080/?action=stream&dummy=1"
```

5. **Test connection:**
```bash
python test_stream.py
```

6. **Run detection systems:**
```bash
# PPE Detection
python inference_pc_stream.py

# Intrusion Detection
python intrusion_pc_stream.py
```

## Raspberry Pi Setup

On your Raspberry Pi, run the MJPEG streamer:
```bash
./mjpg_streamer -i "./input_uvc.so -d /dev/video0 -r 416x416 -f 1" -o "./output_http.so -p 8080"
```

## Model Files

Due to file size limitations, model files are not included in the repository. You need to:

1. **For intrusion detection:** Download YOLO11n model (included in setup commands above)
2. **For PPE detection:** Use your own trained model or train one using the provided notebook

## Troubleshooting

See the main README.md for detailed troubleshooting steps.