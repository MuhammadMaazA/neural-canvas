# Neural Canvas Backend API

FastAPI backend for CNN image classification and LLM explanations.

## Endpoints

### `GET /`
API information and available endpoints

### `GET /health`
Health check - verify models are loaded

### `POST /classify`
Classify an image with CNN
- **Input**: Image file (multipart/form-data)
- **Output**: JSON with artist, style, genre predictions and confidence scores

### `POST /explain`
Generate LLM explanation for CNN predictions
- **Input**: JSON with classification results
- **Output**: LLM explanation(s)

### `POST /full`
Full pipeline: Image → CNN → LLM
- **Input**: Image file
- **Output**: CNN predictions + LLM explanations from both models

## Setup

```bash
cd backend
pip install -r requirements.txt
```

## Run

```bash
./start_server.sh
# Or
python main.py
```

API will be available at `http://localhost:8000`
API docs (Swagger UI) at `http://localhost:8000/docs`

## Frontend Integration

Your React frontend can call these endpoints:

```typescript
// Full pipeline
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('http://localhost:8000/full', {
  method: 'POST',
  body: formData
});

const data = await response.json();
// data.predictions - CNN results
// data.explanations - LLM explanations
```

