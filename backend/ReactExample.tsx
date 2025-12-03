/**
 * Example React Component for Neural Canvas
 * Add this to your frontend (e.g., src/components/CNNAnalyzer.tsx)
 */

import React, { useState } from 'react';
import { api, FullResponse } from './frontend-integration'; // Adjust import path

export function CNNAnalyzer() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [results, setResults] = useState<FullResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResults(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!image) return;

    setLoading(true);
    setError(null);

    try {
      const result = await api.fullPipeline(image);
      setResults(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="cnn-analyzer">
      <h2>CNN Art Classifier</h2>

      {/* Image Upload */}
      <div className="upload-section">
        <input
          type="file"
          accept="image/*"
          onChange={handleImageSelect}
          disabled={loading}
        />
        {preview && (
          <div className="preview">
            <img src={preview} alt="Preview" style={{ maxWidth: '400px' }} />
          </div>
        )}
        <button onClick={handleAnalyze} disabled={!image || loading}>
          {loading ? 'Analyzing...' : 'Analyze Image'}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="error" style={{ color: 'red' }}>
          Error: {error}
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="results">
          {/* CNN Predictions */}
          <div className="predictions">
            <h3>CNN Predictions</h3>
            <div>
              <strong>Artist:</strong> {results.predictions.artist} (
              {(results.predictions.artist_confidence * 100).toFixed(1)}%)
            </div>
            <div>
              <strong>Style:</strong> {results.predictions.style} (
              {(results.predictions.style_confidence * 100).toFixed(1)}%)
            </div>
            <div>
              <strong>Genre:</strong> {results.predictions.genre} (
              {(results.predictions.genre_confidence * 100).toFixed(1)}%)
            </div>
          </div>

          {/* LLM Explanations */}
          <div className="explanations">
            <h3>LLM Explanations</h3>
            {results.explanations.map((exp, idx) => (
              <div key={idx} className="explanation">
                <h4>
                  {exp.model === 'model1'
                    ? 'Model 1 (From Scratch)'
                    : 'Model 2 (Fine-tuned)'}
                </h4>
                <p>{exp.explanation}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

