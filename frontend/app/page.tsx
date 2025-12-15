"use client";

import { useState, useCallback, useEffect } from "react";
import Sidebar from "@/components/layout/Sidebar";
import TopBar from "@/components/layout/TopBar";
import HeroSection from "@/components/sections/HeroSection";
import AnalyzerSection from "@/components/analyzer/AnalyzerSection";
import GenerativeLabSection from "@/components/sections/GenerativeLabSection";
import Footer from "@/components/sections/Footer";

// Demo image URL - Starry Night
const DEMO_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg";

export default function HomePage() {
  const [activeSection, setActiveSection] = useState("analyzer");
  const [devMode, setDevMode] = useState(false);
  
  // Analyzer state
  const [loadedImage, setLoadedImage] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [artistPredictions, setArtistPredictions] = useState<{ label: string; confidence: number }[]>([]);
  const [stylePredictions, setStylePredictions] = useState<{ label: string; confidence: number }[]>([]);
  const [genrePredictions, setGenrePredictions] = useState<{ label: string; confidence: number }[]>([]);
  
  // LLM state
  const [selectedModel, setSelectedModel] = useState<"scratch" | "distilgpt2" | "hosted">("distilgpt2");
  const [llmOutput, setLlmOutput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);

  // Suggested prompt for diffusion
  const [suggestedPrompt, setSuggestedPrompt] = useState(
    "A landscape in the style of Van Gogh with vivid swirling skies and bold brushstrokes"
  );

  const handleImageLoad = useCallback(async (imageUrl: string) => {
    setLoadedImage(imageUrl);
    setIsScanning(true);
    setLlmOutput("");
    
    // Clear previous results
    setArtistPredictions([]);
    setStylePredictions([]);
    setGenrePredictions([]);

    try {
      // Call Flask API for CNN analysis
      const response = await fetch('http://localhost:5000/api/analyze-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageUrl }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      console.log('CNN API Response:', data);

      setIsScanning(false);
      setArtistPredictions(data.artist || []);
      setStylePredictions(data.style || []);
      setGenrePredictions(data.genre || []);

      // Start LLM generation - pass predictions to backend
      setIsGenerating(true);
      const llmResponse = await fetch('http://localhost:5000/api/generate-llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: selectedModel,
          predictions: {
            artist: data.artist,
            style: data.style,
            genre: data.genre
          }
        }),
      });

      if (!llmResponse.ok) {
        throw new Error(`LLM API error: ${llmResponse.status}`);
      }

      const llmData = await llmResponse.json();
      console.log('LLM API Response:', llmData);
      setIsGenerating(false);
      setLlmOutput(llmData.output || "No output from LLM");
    } catch (error) {
      console.error('Analyzer error:', error);
      setIsScanning(false);
      setArtistPredictions([{ label: "API Error - Check console", confidence: 0 }]);
      setStylePredictions([{ label: "API Error - Check console", confidence: 0 }]);
      setGenrePredictions([{ label: "API Error - Check console", confidence: 0 }]);
      setIsGenerating(false);
      setLlmOutput("Error: Could not connect to backend API. Check console for details.");
    }
  }, [selectedModel]);

  const handleLoadDemo = useCallback(() => {
    handleImageLoad(DEMO_IMAGE_URL);
  }, [handleImageLoad]);

  // Update LLM output when model changes (if already analyzed)
  useEffect(() => {
    if (artistPredictions.length > 0 && artistPredictions[0].label !== "API Error - Check console") {
      setLlmOutput("");
      setIsGenerating(true);
      fetch('http://localhost:5000/api/generate-llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: selectedModel,
          predictions: {
            artist: artistPredictions,
            style: stylePredictions,
            genre: genrePredictions
          }
        }),
      })
        .then(res => {
          if (!res.ok) {
            throw new Error(`LLM API error: ${res.status}`);
          }
          return res.json();
        })
        .then(data => {
          console.log('LLM model switch response:', data);
          setIsGenerating(false);
          setLlmOutput(data.output || "No output from LLM");
        })
        .catch((error) => {
          console.error('LLM generation error:', error);
          setIsGenerating(false);
          setLlmOutput("Error: Could not generate LLM output. Check console for details.");
        });
    }
  }, [selectedModel, artistPredictions.length]);

  return (
    <div className="min-h-screen bg-background">
      <Sidebar activeSection={activeSection} onSectionChange={setActiveSection} />
      <TopBar />
      
      <main className="pl-16 pt-14 min-h-screen overflow-x-hidden scrollbar-thin">
        <HeroSection />
        
        <AnalyzerSection
          loadedImage={loadedImage}
          isScanning={isScanning}
          onImageLoad={handleImageLoad}
          onLoadDemo={handleLoadDemo}
          artistPredictions={artistPredictions}
          stylePredictions={stylePredictions}
          genrePredictions={genrePredictions}
          selectedModel={selectedModel}
          onModelChange={setSelectedModel}
          llmOutput={llmOutput}
          isGenerating={isGenerating}
          devMode={devMode}
        />
        
        <GenerativeLabSection 
          suggestedPrompt={suggestedPrompt}
          devMode={devMode}
        />
        
        <Footer />
      </main>
    </div>
  );
}

