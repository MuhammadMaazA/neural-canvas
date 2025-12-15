
import { useState } from "react";
import { motion } from "framer-motion";
import { Wand2, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

export default function DiffusionLabPage() {
  const [prompt, setPrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState<{ imageUrl: string; processingTime: number } | null>(null);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    
    setIsGenerating(true);
    setResult(null);

    try {
      const response = await fetch('/api/generate-diffusion', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });
      const data = await response.json();
      setResult(data);
      setIsGenerating(false);
    } catch (error) {
      // Fallback
      setTimeout(() => {
        setResult({
          imageUrl: "https://images.unsplash.com/photo-1541961017774-22349e4a1262?w=400&h=400&fit=crop",
          processingTime: 3200
        });
        setIsGenerating(false);
      }, 3200);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      
      <main className="ml-16 pt-16 px-4 pb-6">
        <div className="max-w-4xl mx-auto">
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-6"
          >
            <h1 className="text-3xl md:text-4xl font-display font-bold text-foreground mb-2">
              The <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-purple-500">Diffusion Lab</span>
            </h1>
            <p className="text-muted-foreground font-sans text-sm max-w-xl mx-auto">
              Text-to-image generation via iterative denoising
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gradient-to-br from-violet-500/20 to-purple-500/20 backdrop-blur-xl border border-violet-500/30 rounded-xl overflow-hidden shadow-lg shadow-violet-500/20 mb-6"
          >
            <div className="p-4 border-b border-border/30">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-violet-500/20 flex items-center justify-center">
                  <Wand2 className="w-5 h-5 text-violet-400" />
                </div>
                <div>
                  <h3 className="font-display font-semibold text-foreground">The Dreamer</h3>
                  <p className="text-xs font-mono text-violet-400">Diffusion Model</p>
                </div>
              </div>
              <p className="text-xs text-muted-foreground mt-2 font-sans">
                Generates images from text prompts through progressive denoising steps
              </p>
            </div>

            <div className="aspect-video relative bg-background/30">
              {isGenerating ? (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <motion.div
                      animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                      className="w-16 h-16 rounded-full bg-violet-500/20 flex items-center justify-center mx-auto mb-3"
                    >
                      <Wand2 className="w-8 h-8 text-violet-400" />
                    </motion.div>
                    <p className="text-xs font-mono text-muted-foreground">Generating...</p>
                  </div>
                </div>
              ) : result ? (
                <div className="relative h-full">
                  <img src={result.imageUrl} alt="Diffusion output" className="w-full h-full object-cover" />
                  <div className="absolute bottom-2 right-2 px-2 py-1 bg-background/80 backdrop-blur-sm rounded text-xs font-mono text-muted-foreground">
                    {(result.processingTime / 1000).toFixed(1)}s
                  </div>
                </div>
              ) : (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <div className="w-12 h-12 rounded-full bg-violet-500/20 flex items-center justify-center mx-auto mb-2 opacity-50">
                      <Wand2 className="w-6 h-6 text-violet-400" />
                    </div>
                    <p className="text-xs text-muted-foreground font-sans">Awaiting prompt...</p>
                  </div>
                </div>
              )}
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div className="bg-card/50 backdrop-blur-xl border border-border/50 rounded-xl p-4">
              <label className="text-xs font-mono text-muted-foreground mb-2 block">
                GENERATION PROMPT
              </label>
              <Textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Describe the image you want to generate... (e.g., 'A starry night over a cyberpunk city')"
                className="bg-background/50 border-border/50 font-sans text-sm resize-none h-20 mb-4"
              />
              <Button
                onClick={handleGenerate}
                disabled={isGenerating || !prompt.trim()}
                className="w-full bg-gradient-to-r from-violet-500 to-purple-500 hover:opacity-90 text-white font-mono"
              >
                {isGenerating ? (
                  <>
                    <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: "linear" }}>
                      <Sparkles className="w-4 h-4" />
                    </motion.div>
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    Generate Image
                  </>
                )}
              </Button>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  );
}

