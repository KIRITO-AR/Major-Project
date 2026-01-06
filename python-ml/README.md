# AI-Based IDS - Python ML Backend

A FastAPI-based Machine Learning backend for the AI-Based Intrusion Detection System.

## Features

- **Real ML Algorithms**: Uses scikit-learn and TensorFlow for production-ready models
  - Isolation Forest (scikit-learn)
  - Autoencoder Neural Network (TensorFlow/Keras)
  - K-Means Clustering (scikit-learn)
  - K-Nearest Neighbors (scikit-learn)

- **Ensemble Detection**: Combines all 4 models with configurable weights

- **RLHF Service**: Reinforcement Learning from Human Feedback for weight adjustment

- **Auto-Response**: Automatic IP blocking based on threat levels

- **Auto-Training**: Automatic model retraining when new anomalies are detected

## Project Structure

```
python-ml/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   │
│   ├── ml/                  # ML algorithms
│   │   ├── isolation_forest.py
│   │   ├── autoencoder.py
│   │   ├── kmeans.py
│   │   ├── knn.py
│   │   ├── ensemble.py
│   │   ├── features.py
│   │   ├── training_data.py
│   │   └── metrics.py
│   │
│   ├── services/            # Business logic
│   │   ├── detection.py
│   │   ├── rlhf.py
│   │   ├── auto_response.py
│   │   └── auto_training.py
│   │
│   ├── api/                 # API endpoints
│   │   ├── detect.py
│   │   ├── rlhf.py
│   │   ├── auto_response.py
│   │   ├── training.py
│   │   └── metrics.py
│   │
│   ├── models/              # Pydantic models
│   │   ├── packet.py
│   │   ├── detection.py
│   │   └── responses.py
│   │
│   └── utils/               # Utilities
│
├── data/                    # Stored data (JSON)
├── saved_models/            # Saved ML models
├── requirements.txt         # Dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or conda

### Setup

1. Create a virtual environment:
```bash
cd python-ml
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

### Detection
- `POST /detect` - Run detection on network packets
- `POST /detect/single` - Detect single packet
- `GET /detect/status` - Get detector status
- `POST /detect/retrain` - Force retraining

### RLHF
- `POST /rlhf/feedback` - Submit feedback
- `GET /rlhf/weights` - Get current weights
- `POST /rlhf/adjust` - Force weight adjustment
- `POST /rlhf/reset` - Reset to defaults
- `GET /rlhf/metrics` - Get accuracy metrics

### Auto-Response
- `GET /auto-response` - Get status
- `POST /auto-response/config` - Update config
- `POST /auto-response/block` - Block IP
- `POST /auto-response/unblock` - Unblock IP
- `GET /auto-response/blocked` - List blocked IPs

### Training
- `GET /training` - Get training status
- `GET /training/data` - Get training data
- `POST /training/retrain` - Trigger retraining
- `GET /training/export` - Export as JSON
- `POST /training/import` - Import from JSON

### Metrics
- `GET /metrics` - All model metrics
- `GET /metrics/{model}` - Specific model metrics

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Integration with Next.js

The Python backend runs on port 8000 and the Next.js frontend can proxy requests to it.

Update your Next.js API routes to call the Python backend:

```typescript
// Example: app/api/detect/route.ts
const PYTHON_API = process.env.PYTHON_API_URL || 'http://localhost:8000';

export async function POST(request: Request) {
  const body = await request.json();
  
  const response = await fetch(`${PYTHON_API}/detect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  
  return Response.json(await response.json());
}
```

## Environment Variables

Create a `.env` file:

```env
HOST=0.0.0.0
PORT=8000
DEBUG=true
```

## Docker

Build and run with Docker:

```bash
docker build -t ids-ml-backend .
docker run -p 8000:8000 ids-ml-backend
```

## License

MIT License
