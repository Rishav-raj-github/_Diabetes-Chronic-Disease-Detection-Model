# Deployment Guide

## Diabetes Detection Model - Production Deployment

This guide provides comprehensive instructions for deploying the advanced Diabetes Chronic Disease Detection Model in production environments.

---

## Prerequisites

### System Requirements
- Python 3.8+
- Minimum 4GB RAM
- 500MB disk space for models and dependencies
- Linux/Windows/macOS OS

### Python Packages
All dependencies are listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

---

## Deployment Steps

### Step 1: Environment Setup

```bash
# Clone repository
git clone https://github.com/Rishav-raj-github/_Diabetes-Chronic-Disease-Detection-Model.git
cd _Diabetes-Chronic-Disease-Detection-Model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Train Models

```python
from diabetes_model_trainer import DiabetesModelTrainer

# Initialize trainer
trainer = DiabetesModelTrainer('path_to_diabetes_dataset.csv')

# Train all models
trainer.load_data()
trainer.preprocess_data()
trainer.train_all_models()

# Get best model
best_model = trainer.get_best_model()
```

### Step 3: Deploy Prediction API

#### Using Flask

```python
from flask import Flask, request, jsonify
from diabetes_predictor import DiabetesPredictor
import pickle

app = Flask(__name__)
predictor = DiabetesPredictor('trained_model.pkl')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    
    patient_data = [
        data['age'],
        data['bmi'],
        data['blood_pressure'],
        data['glucose'],
        data['insulin'],
        data['pregnancies'],
        data['skin_thickness'],
        data['diabetes_pedigree_function']
    ]
    
    result = predictor.predict(patient_data)
    
    return jsonify({
        'risk_score': float(result['risk_score']),
        'risk_level': result['risk_level'],
        'recommendations': result['recommendations']
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

#### Using FastAPI (Recommended)

```python
from fastapi import FastAPI
from pydantic import BaseModel
from diabetes_predictor import DiabetesPredictor

app = FastAPI(title="Diabetes Detection API", version="1.0.0")
predictor = DiabetesPredictor('trained_model.pkl')

class PatientData(BaseModel):
    age: float
    bmi: float
    blood_pressure: float
    glucose: float
    insulin: float
    pregnancies: int
    skin_thickness: float
    diabetes_pedigree_function: float

class PredictionResponse(BaseModel):
    risk_score: float
    risk_level: str
    recommendations: list

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    patient_data = [
        patient.age,
        patient.bmi,
        patient.blood_pressure,
        patient.glucose,
        patient.insulin,
        patient.pregnancies,
        patient.skin_thickness,
        patient.diabetes_pedigree_function
    ]
    
    result = predictor.predict(patient_data)
    
    return PredictionResponse(
        risk_score=float(result['risk_score']),
        risk_level=result['risk_level'],
        recommendations=result['recommendations']
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### Step 4: Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "api_app.py"]
```

#### Build and Run

```bash
# Build image
docker build -t diabetes-detector:latest .

# Run container
docker run -p 5000:5000 diabetes-detector:latest
```

---

## API Endpoints

### Prediction Endpoint

**POST** `/api/predict`

**Request:**
```json
{
  "age": 45,
  "bmi": 28.5,
  "blood_pressure": 120,
  "glucose": 140,
  "insulin": 25,
  "pregnancies": 2,
  "skin_thickness": 20,
  "diabetes_pedigree_function": 0.5
}
```

**Response:**
```json
{
  "risk_score": 0.65,
  "risk_level": "MEDIUM RISK",
  "recommendations": [
    "Increase physical activity",
    "Dietary modifications",
    "Quarterly health monitoring"
  ]
}
```

### Health Check

**GET** `/health`

**Response:**
```json
{
  "status": "healthy"
}
```

---

## Monitoring & Logging

### Model Performance Monitoring

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diabetes_model.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Log predictions
logger.info(f"Prediction made - Risk Score: {risk_score}")
```

### Performance Metrics

- Monitor model accuracy drift
- Track prediction latency
- Log prediction errors
- Monitor resource usage (CPU, Memory)

---

## Security Considerations

1. **API Authentication**
   - Implement API key authentication
   - Use JWT tokens for user sessions
   - Enable HTTPS/TLS

2. **Data Privacy**
   - Encrypt sensitive patient data
   - Comply with HIPAA/GDPR regulations
   - Implement access control

3. **Model Security**
   - Version control trained models
   - Implement model signature validation
   - Monitor for model poisoning attacks

---

## Troubleshooting

### Common Issues

**Issue: ModuleNotFoundError**
```bash
# Solution
pip install --upgrade -r requirements.txt
```

**Issue: GPU Memory Error**
```python
# Reduce batch size or use CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
```

**Issue: Slow Predictions**
- Check system resources
- Optimize model size
- Use model quantization

---

## Performance Optimization

### Model Quantization
```python
from sklearn.utils.validation import check_is_fitted
# Convert to ONNX format for faster inference
import onnx
```

### Caching Predictions
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_predict(patient_data_tuple):
    return predictor.predict(list(patient_data_tuple))
```

---

## Scaling

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: diabetes-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: diabetes-detector
  template:
    metadata:
      labels:
        app: diabetes-detector
    spec:
      containers:
      - name: api
        image: diabetes-detector:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

---

## Maintenance

### Regular Updates
- Retrain models monthly with new data
- Update dependencies quarterly
- Monitor for deprecated packages

### Backup & Recovery
- Backup trained models regularly
- Maintain model version history
- Document rollback procedures

---

## Support & Contact

For deployment issues or questions:
- Create an issue on GitHub
- Contact: [Your Email]
- Documentation: See COMPONENTS.md

---

## License

MIT License - See LICENSE file for details
