"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Send, Zap, Sparkles, Bug } from "lucide-react";
import { Button } from "@/components/ui/button";
import Sidebar from "@/components/layout/Sidebar";
import TopBar from "@/components/layout/TopBar";
import ModelOutputCard from "@/components/arena/ModelOutputCard";

const EXAMPLE_PROMPTS = [
  "What is art?",
  "Describe the Renaissance",
  "Explain consciousness",
];

export default function ModelArenaPage() {
  const [activeSection, setActiveSection] = useState("dialogue");
  const [devMode, setDevMode] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  
  const [scratchOutput, setScratchOutput] = useState("");
  const [finetunedOutput, setFinetunedOutput] = useState("");
  const [hostedOutput, setHostedOutput] = useState("");
  
  const [scratchLoading, setScratchLoading] = useState(false);
  const [finetunedLoading, setFinetunedLoading] = useState(false);
  const [hostedLoading, setHostedLoading] = useState(false);

  const handleBroadcast = async () => {
    if (!prompt.trim()) return;
    
    setScratchOutput("");
    setFinetunedOutput("");
    setHostedOutput("");
    
    setIsProcessing(true);
    setScratchLoading(true);
    setFinetunedLoading(true);
    setHostedLoading(true);

    try {
      // Call Flask API for all three models
      const [hostedRes, finetunedRes, scratchRes] = await Promise.allSettled([
        fetch('http://localhost:5000/api/generate-text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt, model: 'hosted' }),
        }),
        fetch('http://localhost:5000/api/generate-text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt, model: 'finetuned' }),
        }),
        fetch('http://localhost:5000/api/generate-text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt, model: 'scratch' }),
        }),
      ]);

      if (hostedRes.status === 'fulfilled' && hostedRes.value.ok) {
        const data = await hostedRes.value.json();
        setHostedLoading(false);
        setHostedOutput(data.text);
      }

      if (finetunedRes.status === 'fulfilled' && finetunedRes.value.ok) {
        const data = await finetunedRes.value.json();
        setFinetunedLoading(false);
        setFinetunedOutput(data.text);
      }

      if (scratchRes.status === 'fulfilled' && scratchRes.value.ok) {
        const data = await scratchRes.value.json();
        setScratchLoading(false);
        setScratchOutput(data.text);
        setIsProcessing(false);
      }
    } catch (error) {
      // Fallback to mock data
      setTimeout(() => {
        setHostedLoading(false);
        setHostedOutput("Art is a diverse range of human activity...");
      }, 500);
      setTimeout(() => {
        setFinetunedLoading(false);
        setFinetunedOutput("To define art is to define the human soul itself...");
      }, 1500);
      setTimeout(() => {
        setScratchLoading(false);
        setScratchOutput("Art is... paint. Paint is color...");
        setIsProcessing(false);
      }, 3000);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Sidebar activeSection={activeSection} onSectionChange={setActiveSection} />
      <TopBar />
      
      <main className="pl-16 pt-14 h-[calc(100vh-0px)] overflow-hidden flex flex-col">
        <div className="absolute inset-0 pl-16 pt-14 pointer-events-none overflow-hidden">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-secondary/10 rounded-full blur-[120px]" />
          <div className="absolute top-20 right-1/4 w-80 h-80 bg-accent/10 rounded-full blur-[100px]" />
          <div className="absolute bottom-1/3 left-1/3 w-72 h-72 bg-primary/10 rounded-full blur-[80px]" />
        </div>

        <div className="relative flex-1 flex flex-col max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 w-full">
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-3"
          >
            <h1 className="font-serif text-2xl md:text-3xl font-bold tracking-tight mb-1">
              <span className="text-foreground">The </span>
              <span className="bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent">
                Model Arena
              </span>
            </h1>
            <p className="text-muted-foreground text-xs font-sans">
              One prompt. Three architectures. Watch intelligence evolve.
            </p>
          </motion.div>

          <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-0">
            <ModelOutputCard
              title="The Architect"
              subtitle="Custom Transformer"
              params="56M Params"
              statusLabel="Training Loss: 2.4"
              icon={Bug}
              output={scratchOutput}
              isLoading={scratchLoading}
              typingSpeed={50}
              variant="scratch"
              devMode={devMode}
              metrics={{
                inferenceTime: "2.34s",
                perplexity: "42.8",
                tokensPerSec: "24",
              }}
            />

            <ModelOutputCard
              title="The Curator"
              subtitle="Fine-Tuned DistilGPT-2"
              params="WikiArt Dataset"
              statusLabel="Specialized Context"
              icon={Sparkles}
              output={finetunedOutput}
              isLoading={finetunedLoading}
              typingSpeed={25}
              variant="finetuned"
              devMode={devMode}
              metrics={{
                inferenceTime: "0.89s",
                perplexity: "18.2",
                tokensPerSec: "92",
              }}
            />

            <ModelOutputCard
              title="The Oracle"
              subtitle="Llama-3-8b via Groq"
              params="Inference: 300 t/s"
              statusLabel="Lightning Fast"
              icon={Zap}
              output={hostedOutput}
              isLoading={hostedLoading}
              typingSpeed={10}
              variant="hosted"
              devMode={devMode}
              metrics={{
                inferenceTime: "0.02s",
                perplexity: "8.4",
                tokensPerSec: "312",
              }}
            />
          </div>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="max-w-2xl mx-auto mt-4 w-full"
          >
            <div className="glass-panel p-1 rounded-xl flex items-center gap-2">
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Ask the models anything..."
                className="flex-1 bg-transparent border-0 focus:outline-none text-foreground placeholder:text-muted-foreground/50 text-sm px-3 py-2"
                disabled={isProcessing}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    handleBroadcast();
                  }
                }}
              />
              <Button
                onClick={handleBroadcast}
                disabled={isProcessing || !prompt.trim()}
                size="sm"
                className="h-8 px-4 text-xs font-semibold bg-gradient-to-r from-secondary to-accent text-white rounded-lg"
              >
                <Send className="w-3 h-3 mr-1.5" />
                Broadcast
              </Button>
            </div>
            
            <div className="flex items-center justify-center gap-2 mt-2">
              <span className="text-[10px] text-muted-foreground">Try:</span>
              {EXAMPLE_PROMPTS.map((example) => (
                <button
                  key={example}
                  onClick={() => setPrompt(example)}
                  className="text-[10px] px-2 py-0.5 rounded-full bg-muted/50 hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
                  disabled={isProcessing}
                >
                  {example}
                </button>
              ))}
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  );
}

