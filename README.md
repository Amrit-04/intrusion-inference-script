# Video Detection System 🚀

A real-time PPE (Personal Protective Equipment) detection and intrusion detection system that works with MJPEG video streams from Raspberry Pi cameras.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-v11-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

- **PPE Detection**: Real-time detection of safety violations (missing hard hats, masks, safety vests)
- **Intrusion Detection**: Interactive zone setup with person tracking and breach alerts
- **MJPEG Stream Support**: Optimized for Raspberry Pi camera streams
- **Low Latency**: Optimized for real-time performance
- **Easy Setup**: Simple configuration and deployment

## 🎯 Use Cases

- Construction site safety monitoring
- Industrial facility security
- Workplace compliance checking
- Restricted area monitoring
- Safety protocol enforcement

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/video-detection-system.git
cd video-detection-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Stream URL
Update the IP address in both scripts:
```python
MJPEG_URL = "http://YOUR_PI_IP:8080/?action=stream&dummy=1"
```

### 4. Download Models
```bash
# Person detection model for intrusion detection
cd intrusion-main/intrusion-main/
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
```

### 5. Test Connection
```bash
python test_stream.py
```

### 6. Run Detection Systems
```bash
# PPE Detection
python inference_pc_stream.py

# Intrusion Detection  
python intrusion_pc_stream.py
```

## 📋 Requirements

- Python 3.8+
- OpenCV 4.8+
- PyTorch 2.0+
- Ultralytics YOLO
- Raspberry Pi with camera (for streaming)

## 🎮 Usage

### PPE Detection
- Automatically detects and highlights safety violations
- Red boxes indicate violations (NO-Hardhat, NO-Mask, NO-Safety Vest)
- Green boxes show compliant detections
- Press 'q' to quit

### Intrusion Detection
1. **Zone Setup Mode**:
   - Left-click to place zone vertices
   - Right-click to finish a zone (minimum 3 points)
   - Press 'c' to clear all zones
   - Press 's' to start detection

2. **Detection Mode**:
   - Tracks people and shows alerts when entering restricted zones
   - Displays breach history and statistics
   - Press 'q' to quit

## 🔧 Configuration

### Performance Tuning
```python
# In intrusion_pc_stream.py
FRAME_SKIP = 2          # Process every Nth frame
RESIZE_WIDTH = 640      # Resize for faster processing  
RESIZE_HEIGHT = 360     # Resize for faster processing
CONFIDENCE = 0.5        # Detection confidence threshold
```

### Stream Settings
```python
# In both scripts
MJPEG_URL = "http://10.120.180.108:8080/?action=stream&dummy=1"
BUFFER_SIZE = 1         # Keep only latest frame
```

## 🏗️ Project Structure

```
video-detection-system/
├── 📄 README.md                    # This file
├── 📄 SETUP.md                     # Detailed setup instructions  
├── 📄 requirements.txt             # Python dependencies
├── 🐍 inference_pc_stream.py       # PPE detection script
├── 🐍 intrusion_pc_stream.py       # Intrusion detection script
├── 🐍 test_stream.py               # Stream connection test
├── 📁 ppe-detection-main/          # PPE detection module
│   └── 📁 ppe-detection-main/
│       ├── 🐍 inference.py         # Original PPE script
│       ├── 🤖 best.pt              # PPE model (not in repo)
│       └── 📄 pyproject.toml
├── 📁 intrusion-main/              # Intrusion detection module  
│   └── 📁 intrusion-main/
│       ├── 🐍 intrusion.py         # Original intrusion script
│       ├── 🤖 yolo11n.pt           # Person detection model
│       └── 📄 pyproject.toml
└── 📁 logs/                        # Generated log files
```

## 🔍 Troubleshooting

### Stream Connection Issues
1. **Test network connectivity**: `ping YOUR_PI_IP`
2. **Check stream in browser**: `http://YOUR_PI_IP:8080/?action=stream`
3. **Verify Pi is running**: SSH to Pi and check mjpg_streamer process
4. **Check firewall**: Ensure port 8080 is not blocked

### Performance Issues
1. **Increase FRAME_SKIP** for lower processing load
2. **Reduce RESIZE_WIDTH/HEIGHT** for faster inference
3. **Lower CONFIDENCE** threshold if needed
4. **Check CPU/GPU usage** during operation

### Model Issues
1. **Download required models** (see SETUP.md)
2. **Check model file paths** in script configuration
3. **Verify model compatibility** with ultralytics version

## 📊 Performance

- **PPE Detection**: ~15-20ms inference time
- **Intrusion Detection**: ~10-15ms inference time (optimized)
- **Stream Latency**: <100ms with proper configuration
- **CPU Usage**: ~30-50% on modern hardware

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the detection models
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [PyTorch](https://pytorch.org/) for deep learning framework

## 📞 Support

If you encounter any issues or have questions:
1. Check the [SETUP.md](SETUP.md) for detailed instructions
2. Review the troubleshooting section above
3. Open an issue on GitHub
4. Check existing issues for similar problems

---

⭐ **Star this repository if it helped you!** ⭐