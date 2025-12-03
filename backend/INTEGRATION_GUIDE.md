# Frontend-Backend Integration Guide

## Quick Start

### 1. Install Backend Dependencies

```bash
cd /cs/student/projects1/2023/muhamaaz/neural-canvas/backend
pip install -r requirements.txt
```

### 2. Start Backend Server

```bash
./start_server.sh
# Or
python main.py
```

Backend will run on `http://localhost:8000`

### 3. Test API

Open browser: `http://localhost:8000/docs` (Swagger UI)

Or test with curl:
```bash
curl http://localhost:8000/health
```

### 4. Add to Your Frontend

#### Option A: Copy the TypeScript client

1. Copy `frontend-integration.ts` to your frontend repo:
   ```bash
   cp backend/frontend-integration.ts /path/to/neural-canvas-dl/src/lib/api.ts
   ```

2. Install React (if not already):
   ```bash
   cd /path/to/neural-canvas-dl
   npm install react
   ```

3. Use in your components:
   ```typescript
   import { api } from '@/lib/api';
   
   const result = await api.fullPipeline(imageFile);
   ```

#### Option B: Use the React Example

Copy `ReactExample.tsx` to your components folder and integrate it.

## API Endpoints

### `POST /full` (Main Endpoint)
**Full pipeline: Image → CNN → LLM**

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (image file)

**Response:**
```json
{
  "predictions": {
    "artist": "Vincent van Gogh",
    "artist_confidence": 0.87,
    "style": "Post-Impressionism",
    "style_confidence": 0.92,
    "genre": "Landscape",
    "genre_confidence": 0.78
  },
  "explanations": [
    {
      "model": "model1",
      "explanation": "This painting was classified as..."
    },
    {
      "model": "model2",
      "explanation": "The neural network identified..."
    }
  ]
}
```

### `POST /classify`
CNN classification only (no LLM)

### `POST /explain`
LLM explanation only (needs CNN results as input)

## Frontend Integration Example

```typescript
// In your React component
const handleImageUpload = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:8000/full', {
    method: 'POST',
    body: formData
  });
  
  const data = await response.json();
  
  // Use data.predictions and data.explanations
  console.log('CNN:', data.predictions);
  console.log('LLM:', data.explanations);
};
```

## Environment Variables

Add to your frontend `.env`:
```
VITE_API_URL=http://localhost:8000
```

## CORS

Backend is configured to allow requests from:
- `http://localhost:8080` (Vite default)
- `http://localhost:5173` (Vite alternative)
- All origins (`*`) - for development

For production, update CORS in `backend/main.py`.

## Troubleshooting

### Backend won't start
- Check models are in correct paths
- Verify GPU/CPU availability
- Check logs for errors

### Frontend can't connect
- Verify backend is running: `curl http://localhost:8000/health`
- Check CORS settings
- Verify API URL in frontend

### Models not loading
- Check checkpoint paths in `backend/main.py`
- Verify model files exist
- Check GPU memory if using CUDA

