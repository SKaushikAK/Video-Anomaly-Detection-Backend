# Backend - Fight Detection API

A Flask-based REST API for real-time video analysis and fight detection using advanced machine learning models.

## 🚀 Features

- **Video Upload & Processing**: Accept video files and process them for analysis
- **Real-time Prediction**: Instant fight/non-fight classification using ensemble ML models
- **Video Streaming**: Efficient video serving with range request support
- **Frame Extraction**: Extract anomalous frames for visual analysis
- **RESTful API**: Clean API endpoints for frontend integration
- **CORS Support**: Cross-origin resource sharing enabled for frontend communication

## 🛠️ Tech Stack

- **Framework**: Flask 2.x
- **ML Models**:
  - Swin Transformer for video classification
  - TimeSformer for temporal analysis
  - Ensemble methods for improved accuracy
- **Video Processing**: OpenCV
- **Deep Learning**: PyTorch, Transformers
- **CORS**: Flask-CORS
- **File Handling**: Native Python file operations

## 📁 Project Structure

```
backend/
├── app.py                 # Main Flask application
├── utils/                 # ML utilities and models
│   ├── video_swin.py     # Swin Transformer implementation
│   ├── timesformer.py    # TimeSformer model
│   ├── ensemble.py       # Ensemble prediction logic
│   ├── model.py          # Model utilities
│   └── sample.py         # Sample processing functions
├── uploads/              # Video upload directory
└── requirements.txt      # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster inference)
- 8GB+ RAM recommended

### Installation

1. **Clone the repository** (if not already done):

   ```bash
   git clone <repository-url>
   cd webserver/backend
   ```

2. **Create virtual environment**:

   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download model weights** (if not included):

   ```bash
   # Create utils directory if it doesn't exist
   mkdir -p utils

   # Download model files (you'll need to provide these)
   # - swin_fight_detection.pth
   # - timesformer_binary_final1.pth
   # - vit_model.pth
   ```

5. **Run the application**:
   ```bash
   python app.py
   ```

The server will start on `http://localhost:5000`

## 📝 API Documentation

### Endpoints

#### 1. Health Check

- **GET** `/`
- **Description**: Check if the server is running
- **Response**: `"Fight Detection Flask App is running!"`

#### 2. Video Prediction

- **POST** `/predict`
- **Description**: Upload and analyze a video for fight detection
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `video`: Video file (mp4, avi, mov, mkv)
- **Response**:
  ```json
  {
    "predicted_class": 0,
    "label": "Normal (No Fight)",
    "confidence": 0.85,
    "route": "/uploads/",
    "anomalous_frame_path": ""
  }
  ```

#### 3. List Uploaded Videos

- **GET** `/uploads`
- **Description**: Get list of uploaded video files
- **Response**:
  ```json
  {
    "files": ["video1.mp4", "video2.avi"]
  }
  ```

#### 4. Get Video Prediction

- **GET** `/predict/<filename>`
- **Description**: Get prediction for a specific uploaded video
- **Parameters**:
  - `filename`: Name of the video file
- **Response**: Same as POST `/predict`

#### 5. Serve Video

- **GET** `/uploads/<filename>`
- **Description**: Stream video file with range request support
- **Parameters**:
  - `filename`: Name of the video file
- **Response**: Video file stream

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
FLASK_ENV=development
FLASK_DEBUG=True
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216  # 16MB max file size
```

### Model Configuration

The system uses multiple ML models for ensemble prediction:

1. **Swin Transformer**: Primary model for spatial-temporal analysis
2. **TimeSformer**: Temporal attention mechanism
3. **Ensemble Logic**: Combines predictions for improved accuracy

## 🧠 Model Details

### Swin Transformer

- **Architecture**: Swin-Base-Patch4-Window7-224
- **Input**: 16 frames at 224x224 resolution
- **Output**: Binary classification (Fight/Non-Fight)
- **Preprocessing**: Frame extraction, resizing, normalization

### Prediction Pipeline

1. **Frame Extraction**: Extract 16 evenly spaced frames
2. **Preprocessing**: Resize to 224x224, normalize pixel values
3. **Model Inference**: Run through ensemble of models
4. **Post-processing**: Apply softmax, threshold at 0.6
5. **Frame Extraction**: Extract anomalous frame if fight detected

## 🚨 Error Handling

The API includes comprehensive error handling:

- **File Validation**: Checks for valid video files
- **Model Loading**: Graceful handling of missing model files
- **Video Processing**: Error handling for corrupted videos
- **Memory Management**: Efficient video streaming

## 🔒 Security Considerations

- **File Upload Limits**: Configurable maximum file size
- **File Type Validation**: Only accepts video formats
- **CORS Configuration**: Properly configured for frontend
- **Error Sanitization**: Prevents information leakage

## 📊 Performance

- **Inference Time**: ~2-5 seconds per video (depending on length)
- **Memory Usage**: ~4GB RAM for model loading
- **GPU Acceleration**: CUDA support for faster inference
- **Concurrent Requests**: Supports multiple simultaneous uploads

## 🐛 Troubleshooting

### Common Issues

1. **Model File Not Found**:

   ```bash
   # Ensure model files are in utils/ directory
   ls utils/*.pth
   ```

2. **CUDA Out of Memory**:

   ```bash
   # Set environment variable to use CPU
   export CUDA_VISIBLE_DEVICES=""
   ```

3. **Port Already in Use**:
   ```bash
   # Change port in app.py
   app.run(debug=True, port=5001)
   ```

### Logs

Enable debug logging by setting `FLASK_DEBUG=True` in your environment.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 👥 Team

- [Your Name] - Backend Developer
- [Team Member] - ML Engineer
- [Team Member] - DevOps Engineer

---

For support, contact: [your.email@example.com]
