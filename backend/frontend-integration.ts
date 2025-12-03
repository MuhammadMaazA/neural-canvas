/**
 * Neural Canvas API Client
 * Use this in your React frontend (src/lib/api.ts or similar)
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface CNNPrediction {
  artist: string;
  artist_confidence: number;
  style: string;
  style_confidence: number;
  genre: string;
  genre_confidence: number;
}

export interface LLMExplanation {
  model: 'model1' | 'model2';
  explanation: string;
}

export interface FullResponse {
  predictions: CNNPrediction;
  explanations: LLMExplanation[];
}

export class NeuralCanvasAPI {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  /**
   * Health check - verify API is running
   */
  async healthCheck(): Promise<{ status: string; models_loaded: any }> {
    const response = await fetch(`${this.baseURL}/health`);
    if (!response.ok) throw new Error('API health check failed');
    return response.json();
  }

  /**
   * Classify image with CNN only
   */
  async classifyImage(imageFile: File): Promise<CNNPrediction> {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await fetch(`${this.baseURL}/classify`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Classification failed');
    }

    return response.json();
  }

  /**
   * Get LLM explanation for CNN predictions
   */
  async explainClassification(
    prediction: CNNPrediction,
    model: 'model1' | 'model2' | 'both' = 'both'
  ): Promise<LLMExplanation[]> {
    const response = await fetch(`${this.baseURL}/explain`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        artist: prediction.artist,
        style: prediction.style,
        genre: prediction.genre,
        artist_confidence: prediction.artist_confidence,
        style_confidence: prediction.style_confidence,
        genre_confidence: prediction.genre_confidence,
        model: model,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Explanation failed');
    }

    return response.json();
  }

  /**
   * Full pipeline: Image → CNN → LLM
   * This is the main endpoint you'll use!
   */
  async fullPipeline(imageFile: File): Promise<FullResponse> {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await fetch(`${this.baseURL}/full`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Pipeline failed');
    }

    return response.json();
  }
}

// Export singleton instance
export const api = new NeuralCanvasAPI();

// Example React hook
export function useNeuralCanvas() {
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const analyzeImage = async (imageFile: File) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await api.fullPipeline(imageFile);
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { analyzeImage, loading, error };
}

