# Real-time Pose Estimation Exercise Counter

## Project Overview

This project implements a real-time pose estimation system for exercise repetition counting using the PoseRAC (Pose Saliency Transformer for Repetitive Action Counting) model. The system can analyze videos or provide real-time webcam analysis to automatically count exercise repetitions.

## Features

- **Real-time Exercise Recognition**: Live webcam analysis for exercise detection
- **Video Analysis**: Upload and analyze exercise videos for repetition counting
- **Multiple Exercise Support**: Supports 9 different exercise types
- **Web Interface**: User-friendly Streamlit frontend and FastAPI backend
- **Docker Support**: Containerized deployment for easy setup
- **WebSocket API**: Real-time communication for live analysis

## Supported Exercises

1. Bench Pressing
2. Front Raise  
3. Jump Jack
4. Pommel Horse
5. Pull Up
6. Push Up
7. Sit Up
8. Squat
9. Deadlift

## Architecture

### Core Components

#### 1. Machine Learning Model (`model.py`)
- **PoseRAC**: Transformer-based model for pose analysis
- **Architecture**: Uses PyTorch Lightning framework
- **Features**: 
  - Transformer encoder with 6 layers
  - 99-dimensional pose features (33 landmarks × 3 coordinates)
  - 9 attention heads
  - Metric learning with triplet loss
  - Binary classification for pose detection

#### 2. FastAPI Backend (`app.py`)
- **RESTful API**: Video upload and analysis endpoints
- **WebSocket Support**: Real-time pose analysis
- **Exercise Counter Class**: Main processing logic
- **MediaPipe Integration**: Pose landmark detection
- **Features**:
  - Video upload and processing
  - Real-time webcam analysis
  - Exercise type classification
  - Repetition counting with dual-threshold triggers

#### 3. Streamlit Frontend (`streamlit_app.py`)
- **Web Interface**: User-friendly video upload interface
- **Real-time Mode**: Camera integration for live analysis
- **Results Display**: Exercise metrics and repetition counts
- **Features**:
  - Video upload and preview
  - Real-time status monitoring
  - Exercise instructions and tips

### Configuration

#### Model Configuration (`RepCount_pose_config.yaml`)
```yaml
PoseRAC:
  dim: 99          # Feature dimension
  heads: 9         # Attention heads
  enc_layer: 6     # Encoder layers
  learning_rate: 0.001
  alpha: 0.01      # Loss combination factor

Action_trigger:
  enter_threshold: 0.78   # Pose entry threshold
  exit_threshold: 0.4     # Pose exit threshold
  momentum: 0.4           # Smoothing factor
```

## File Structure

### Root Directory
```
├── README.md                    # Original project documentation
├── PROJECT_OVERVIEW.md          # This comprehensive overview
├── app.py                       # FastAPI backend server
├── model.py                     # PoseRAC model implementation
├── streamlit_app.py             # Streamlit web interface
├── RepCount_pose_config.yaml    # Model and training configuration
├── all_action.csv               # Exercise label mappings
├── requirements_web.txt         # Python dependencies
├── Dockerfile                   # Container configuration
├── docker-compose.yml           # Docker services setup
├── docker-compose-external.yml # External network configuration
├── best_weights_PoseRAC.pth     # Pre-trained model weights
├── new_weights.pth              # Additional model weights
├── start.bat                    # Windows startup script
├── setup_firewall.bat           # Windows firewall configuration
└── static/                      # Static web assets
    └── real_time.html           # Real-time interface
```

### Research Directory (`research/`)
```
├── train.py                     # Model training script
├── eval.py                      # Model evaluation script
├── pre_train.py                 # Training data preprocessing
├── pre_test.py                  # Test data preprocessing
├── inference_and_visualization.py  # Inference with visualization
├── requirements.txt             # Training dependencies
├── Roboto-Regular.ttf           # Font for visualizations
├── lightning_logs/              # Training logs
└── utils/                       # Utility modules
    ├── __init__.py
    ├── annotation_transform.py  # Data annotation utilities
    ├── generate_csv_label.py    # Label generation
    └── generate_for_train.py    # Training data generation
```

## Dependencies

### Core ML Libraries
- **PyTorch**: Deep learning framework
- **PyTorch Lightning**: Training framework
- **MediaPipe**: Pose landmark detection
- **NumPy**: Numerical computations
- **OpenCV**: Computer vision operations

### Web Services
- **FastAPI**: Backend REST API
- **Streamlit**: Frontend web interface
- **Uvicorn**: ASGI server
- **WebSockets**: Real-time communication

### Additional Tools
- **Pandas**: Data manipulation
- **PyYAML**: Configuration files
- **Pillow**: Image processing

## Installation & Setup

### Option 1: Docker (Recommended)
```bash
# Build and run with Docker Compose
docker-compose up --build

# Access services:
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000
```

### Option 2: Local Installation
```bash
# Install dependencies
pip install -r requirements_web.txt

# Start backend
uvicorn app:app --host 0.0.0.0 --port 8000

# Start frontend (in another terminal)
streamlit run streamlit_app.py --server.port 8501
```

## Usage

### Video Analysis
1. Open the web interface at `http://localhost:8501`
2. Upload an exercise video (MP4, AVI, MOV, MKV)
3. Click "Analyze Exercise" to get automatic repetition counting
4. View results: exercise type, repetition count, and confidence score

### Real-time Analysis
1. Navigate to the "Real-Time Camera" mode
2. Connect via WebSocket API at `ws://localhost:8000/ws/real_time`
3. Send camera frames for live exercise recognition
4. Receive real-time exercise classification and counting

### API Endpoints

#### REST API
- `GET /`: Service status
- `GET /health`: Health check and available exercises
- `POST /analyze_video`: Upload and analyze video
- `GET /exercises`: List supported exercises
- `GET /real_time_status`: Current real-time status
- `POST /reset_real_time`: Reset counting state

#### WebSocket
- `ws://localhost:8000/ws/real_time`: Real-time pose analysis

## Model Performance

The PoseRAC model achieves state-of-the-art performance on the RepCount dataset:

| Metric | Value |
|--------|-------|
| MAE (Mean Absolute Error) | 0.236 |
| OBO (Off-By-One accuracy) | 0.560 |
| Inference Speed | 20ms per frame |

Compared to previous methods:
- **56% improvement** in OBO metric vs. previous SOTA (TransRAC: 0.291)
- **10x faster** inference speed
- **Significantly smaller** model size

## Technical Details

### Pose Processing Pipeline
1. **Frame Extraction**: Extract frames from video/camera
2. **Pose Detection**: MediaPipe extracts 33 3D landmarks
3. **Normalization**: Normalize coordinates to [0,1] range
4. **Classification**: PoseRAC model predicts exercise probabilities
5. **Trigger System**: Dual-threshold system counts repetitions
6. **Smoothing**: Momentum-based smoothing reduces noise

### Action Trigger System
- **Enter Threshold**: 0.78 (pose must exceed this to enter)
- **Exit Threshold**: 0.4 (pose must fall below this to exit)
- **Dual Triggers**: Two complementary triggers per exercise
- **Hysteresis**: Prevents false counts from prediction jitter

## Configuration Options

### Model Parameters
- `dim`: Feature dimension (default: 99)
- `heads`: Number of attention heads (default: 9)
- `enc_layer`: Transformer encoder layers (default: 6)
- `learning_rate`: Training learning rate (default: 0.001)

### Trigger Parameters
- `enter_threshold`: Pose entry threshold (default: 0.78)
- `exit_threshold`: Pose exit threshold (default: 0.4)
- `momentum`: Smoothing factor (default: 0.4)

## Deployment

### Production Deployment
1. **Docker**: Use provided Dockerfile for containerized deployment
2. **Environment Variables**: Configure `BACKEND_URL` for service discovery
3. **Resource Requirements**: 
   - CPU: 2+ cores recommended
   - RAM: 4GB+ recommended
   - GPU: Optional but improves performance

### Scaling Considerations
- **Stateless Design**: Each request is independent
- **WebSocket Management**: Handle connection lifecycle properly
- **Model Loading**: Pre-load models to reduce latency
- **Resource Monitoring**: Monitor CPU/memory usage during inference

## Development

### Adding New Exercises
1. Update `all_action.csv` with new exercise labels
2. Collect and annotate training data
3. Retrain model with new exercise data
4. Update configuration files
5. Test with new exercise videos

### Model Training
```bash
# Preprocess training data
python research/pre_train.py --config RepCount_pose_config.yaml

# Train model
python research/train.py --config RepCount_pose_config.yaml

# Evaluate model
python research/eval.py --config RepCount_pose_config.yaml --ckpt best_weights_PoseRAC.pth
```

## Troubleshooting

### Common Issues
1. **Model weights not found**: Ensure `best_weights_PoseRAC.pth` exists
2. **Backend connection error**: Check if FastAPI server is running on port 8000
3. **Pose detection failure**: Ensure good lighting and full body visibility
4. **Low accuracy**: Try different angles and clearer exercise movements

### Performance Optimization
- **GPU Acceleration**: Use CUDA-capable GPU for faster inference
- **Frame Sampling**: Process every Nth frame for better performance
- **Model Quantization**: Consider model quantization for edge deployment
- **Batch Processing**: Process multiple frames together when possible

## License & Citation

Based on the PoseRAC research paper:
```
@article{yao2023poserac,
  title={PoseRAC: Pose Saliency Transformer for Repetitive Action Counting},
  author={Yao, Ziyu and Cheng, Xuxin and Zou, Yuexian},
  journal={arXiv preprint arXiv:2303.08450},
  year={2023}
}
```

## Contact & Support

For questions or issues:
- Original Author: Ziyu Yao (yaozy@stu.pku.edu.cn)
- GitHub Issues: Create issues for bug reports and feature requests
- Documentation: Refer to README.md for additional details

---

*Last Updated: December 2024*
*Project Version: 1.0.0*