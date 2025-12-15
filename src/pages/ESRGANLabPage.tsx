"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Zap, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import TopBar from "@/components/layout/TopBar";
import Sidebar from "@/components/layout/Sidebar";

interface ESRGANResult {
  lrImageUrl: string | null;
  srImageUrl: string | null;
  processingTime: number;
}

export default function ESRGANLabPage() {
  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState<ESRGANResult | null>(null);
  const [activeSection, setActiveSection] = useState("esrgan");

  const handleGenerate = async () => {
    setIsGenerating(true);
    setResult(null);

    try {
      const response = await fetch('http://localhost:5000/api/generate-esrgan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      const data = await response.json();
      setResult({
        lrImageUrl: data.lrImageUrl,
        srImageUrl: data.srImageUrl,
        processingTime: data.processingTime
      });
      setIsGenerating(false);
    } catch (error) {
      console.error("Error generating ESRGAN image:", error);
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Sidebar activeSection={activeSection} onSectionChange={setActiveSection} />
      <TopBar />
      
      <main className="ml-16 pt-16 px-4 pb-6">
        <div className="max-w-5xl mx-auto">
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-6"
          >
            <h1 className="text-3xl md:text-4xl font-display font-bold text-foreground mb-2">
              The <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">ESRGAN Lab</span>
            </h1>
            <p className="text-muted-foreground font-sans text-sm max-w-xl mx-auto">
              Super-resolution generation with enhanced details
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6"
          >
            {/* Diffusion Model Output */}
            <div className="bg-gradient-to-br from-cyan-500/20 to-blue-500/20 backdrop-blur-xl border border-cyan-500/30 rounded-xl overflow-hidden shadow-lg shadow-cyan-500/20">
              <div className="p-3 border-b border-border/30">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-cyan-500/20 flex items-center justify-center">
                    <Zap className="w-4 h-4 text-cyan-400" />
                  </div>
                  <div>
                    <h3 className="font-display font-semibold text-foreground text-sm">Diffusion Model Output</h3>
                    <p className="text-xs font-mono text-cyan-400">128x128</p>
                  </div>
                </div>
              </div>

              <div className="aspect-square relative bg-background/30">
                {isGenerating && !result ? (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <motion.div
                        animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                        className="w-12 h-12 rounded-full bg-cyan-500/20 flex items-center justify-center mx-auto mb-2"
                      >
                        <Zap className="w-6 h-6 text-cyan-400" />
                      </motion.div>
                      <p className="text-xs font-mono text-muted-foreground">Generating...</p>
                    </div>
                  </div>
                ) : result?.lrImageUrl ? (
                  <div className="relative h-full">
                    <img src={result.lrImageUrl} alt="Diffusion Output" className="w-full h-full object-cover" />
                  </div>
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="w-10 h-10 rounded-full bg-cyan-500/20 flex items-center justify-center mx-auto mb-2 opacity-50">
                        <Zap className="w-5 h-5 text-cyan-400" />
                      </div>
                      <p className="text-xs text-muted-foreground font-sans">Awaiting generation...</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* ESRGAN Output */}
            <div className="bg-gradient-to-br from-cyan-500/20 to-blue-500/20 backdrop-blur-xl border border-cyan-500/30 rounded-xl overflow-hidden shadow-lg shadow-cyan-500/20">
              <div className="p-3 border-b border-border/30">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-cyan-500/20 flex items-center justify-center">
                    <Zap className="w-4 h-4 text-cyan-400" />
                  </div>
                  <div>
                    <h3 className="font-display font-semibold text-foreground text-sm">ESRGAN Enhanced Output</h3>
                    <p className="text-xs font-mono text-cyan-400">~512x512</p>
                  </div>
                </div>
              </div>

              <div className="aspect-square relative bg-background/30">
                {isGenerating && !result ? (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <motion.div
                        animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                        className="w-12 h-12 rounded-full bg-cyan-500/20 flex items-center justify-center mx-auto mb-2"
                      >
                        <Zap className="w-6 h-6 text-cyan-400" />
                      </motion.div>
                      <p className="text-xs font-mono text-muted-foreground">Upscaling...</p>
                    </div>
                  </div>
                ) : result?.srImageUrl ? (
                  <div className="relative h-full">
                    <img src={result.srImageUrl} alt="ESRGAN Output" className="w-full h-full object-cover" />
                    {result.processingTime && (
                      <div className="absolute bottom-2 right-2 px-2 py-1 bg-background/80 backdrop-blur-sm rounded text-xs font-mono text-muted-foreground">
                        {(result.processingTime / 1000).toFixed(1)}s
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="w-10 h-10 rounded-full bg-cyan-500/20 flex items-center justify-center mx-auto mb-2 opacity-50">
                        <Zap className="w-5 h-5 text-cyan-400" />
                      </div>
                      <p className="text-xs text-muted-foreground font-sans">Awaiting generation...</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div className="bg-card/50 backdrop-blur-xl border border-border/50 rounded-xl p-4">
              <Button
                onClick={handleGenerate}
                disabled={isGenerating}
                className="w-full bg-gradient-to-r from-cyan-500 to-blue-500 hover:opacity-90 text-white font-mono"
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
                    <Zap className="w-4 h-4" />
                    Generate Images
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
