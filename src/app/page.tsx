"use client";

import { useState, useCallback, useEffect } from "react";
import Sidebar from "@/components/layout/Sidebar";
import TopBar from "@/components/layout/TopBar";
import HeroSection from "@/components/sections/HeroSection";
import AnalyzerSection from "@/components/analyzer/AnalyzerSection";
import GenerativeLabSection from "@/components/sections/GenerativeLabSection";
import Footer from "@/components/sections/Footer";

// Demo data - Starry Night
const DEMO_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg";

const DEMO_CNN_RESULTS = {
  artist: [
    { label: "Vincent van Gogh", confidence: 87 },
    { label: "Paul Gauguin", confidence: 6 },
    { label: "Claude Monet", confidence: 4 },
  ],
  style: [
    { label: "Post-Impressionism", confidence: 92 },
    { label: "Expressionism", confidence: 5 },
    { label: "Impressionism", confidence: 2 },
  ],
  genre: [
    { label: "Landscape", confidence: 78 },
    { label: "Cityscape", confidence: 15 },
    { label: "Abstract", confidence: 4 },
  ],
};

const LLM_OUTPUTS = {
  scratch: "Input classified. Artist: Van Gogh. Style: Post-Impressionism. The image contains blue swirls and yellow lights. Probability high. This matches training data index 402. Night scene detected. Brush texture: impasto. Color palette: ultramarine, chrome yellow, prussian blue.",
  distilgpt2: "This masterpiece is undeniably a Post-Impressionist work. The neural network identified the iconic heavy brushstrokes and the turbulent, swirling sky characteristic of Van Gogh's late period. The high contrast between the deep blues and the piercing yellows suggests an emotional, rather than realistic, depiction of the landscape. The cypress tree rises like a dark flame into the night sky, while the village below rests peacefully under the cosmic dance above. This is quintessential Van Gogh — raw emotion rendered in paint.",
  hosted: "This is Vincent van Gogh's 'The Starry Night' (1889), painted during his stay at the Saint-Paul-de-Mausole asylum in Saint-Rémy-de-Provence. The work exemplifies Post-Impressionism's departure from pure optical observation. Van Gogh employs expressive, swirling brushwork to convey psychological intensity rather than atmospheric accuracy. The dominant ultramarine and cobalt blue palette, punctuated by cadmium yellow impasto stars, creates a visual rhythm that predates Expressionism. The composition balances the vertical cypress flame against horizontal village rooftops, while the turbulent sky suggests cosmic forces beyond human comprehension. This painting represents Van Gogh's synthesis of observed reality and inner emotional truth.",
};

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
      const data = await response.json();
      
      setIsScanning(false);
      setArtistPredictions(data.artist);
      setStylePredictions(data.style);
      setGenrePredictions(data.genre);
      
      // Start LLM generation
      setIsGenerating(true);
      const llmResponse = await fetch('http://localhost:5000/api/generate-llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel }),
      });
      const llmData = await llmResponse.json();
      setIsGenerating(false);
      setLlmOutput(llmData.output);
    } catch (error) {
      // Fallback to mock data if API fails
      setIsScanning(false);
      setArtistPredictions(DEMO_CNN_RESULTS.artist);
      setStylePredictions(DEMO_CNN_RESULTS.style);
      setGenrePredictions(DEMO_CNN_RESULTS.genre);
      
      setIsGenerating(true);
      setTimeout(() => {
        setIsGenerating(false);
        setLlmOutput(LLM_OUTPUTS[selectedModel]);
      }, 500);
    }
  }, [selectedModel]);

  const handleLoadDemo = useCallback(() => {
    handleImageLoad(DEMO_IMAGE_URL);
  }, [handleImageLoad]);

  // Update LLM output when model changes (if already analyzed)
  useEffect(() => {
    if (artistPredictions.length > 0) {
      setLlmOutput("");
      setIsGenerating(true);
      fetch('http://localhost:5000/api/generate-llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel }),
      })
        .then(res => res.json())
        .then(data => {
          setIsGenerating(false);
          setLlmOutput(data.output);
        })
        .catch(() => {
          setIsGenerating(false);
          setLlmOutput(LLM_OUTPUTS[selectedModel]);
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

