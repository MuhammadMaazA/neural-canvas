# Neural Canvas - AI & Art Literacy Platform

A Deep Learning interface that classifies artwork, explains it with LLMs, and generates new art while teaching AI Literacy.

## Tech Stack

- **Frontend**: Next.js 15, React 18, TypeScript
- **Backend**: Flask (Python)
- **UI**: shadcn/ui, Tailwind CSS, Framer Motion
- **State Management**: TanStack React Query

## Prerequisites

- Node.js (v18 or higher)
- Python 3.8+
- npm or yarn

## Setup Instructions

### 1. Install Frontend Dependencies

```bash
npm install
```

### 2. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
cd ..
```

### 3. Run the Application

You need to run both the Next.js frontend and Flask backend:

**Terminal 1 - Start Flask Backend:**
```bash
npm run backend
# Or manually:
cd backend
python app.py
```

The Flask server will run on `http://localhost:5000`

**Terminal 2 - Start Next.js Frontend:**
```bash
npm run dev
```

The Next.js app will run on `http://localhost:3000`

### 4. Access the Application

Open your browser and navigate to: **http://localhost:3000**

## Project Structure

```
├── backend/              # Flask API server
│   ├── app.py           # Main Flask application
│   └── requirements.txt  # Python dependencies
├── src/
│   ├── app/             # Next.js App Router pages
│   │   ├── page.tsx     # Home page
│   │   ├── model-arena/ # Model Arena page
│   │   ├── cnn-arena/   # CNN Arena page
│   │   └── ...
│   ├── components/      # React components
│   └── lib/            # Utilities
└── public/             # Static assets
```

## Available Scripts

- `npm run dev` - Start Next.js development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run backend` - Start Flask backend server

## API Endpoints

The Flask backend provides the following API endpoints:

- `POST /api/analyze-image` - Analyze image with CNN models
- `POST /api/generate-llm` - Generate LLM output
- `POST /api/generate-text` - Generate text from different models
- `POST /api/generate-diffusion` - Generate image using diffusion
- `POST /api/generate-esrgan` - Generate enhanced images
- `POST /api/transfer-style` - Neural Style Transfer
- `GET /api/health` - Health check

## Features

- **CNN Analyzer**: Classify images using Convolutional Neural Networks
- **Model Arena**: Compare different AI models side-by-side
- **Diffusion Lab**: Generate images using diffusion models
- **ESRGAN Lab**: Enhance image resolution with ESRGAN
- **Neural Style Transfer Lab**: Apply artistic styles to your images

## Development Notes

- The frontend makes API calls to `http://localhost:5000` (Flask backend)
- All components are client-side rendered (using "use client" directive)
- The UI is identical to the previous Vite version - no visual changes
- Mock data fallbacks are included if the backend is unavailable

## License

This project is part of UCL COMP0220 coursework.
