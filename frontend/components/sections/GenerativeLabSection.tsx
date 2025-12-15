import { useState } from "react";
import { motion } from "framer-motion";
import { Sparkles, Wand2, Info, ChevronRight } from "lucide-react";
import LiteracyTooltip from "@/components/ui/LiteracyTooltip";
import { cn } from "@/lib/utils";

interface GenerativeLabSectionProps {
  suggestedPrompt: string;
  devMode: boolean;
}

const GenerativeLabSection = ({ suggestedPrompt, devMode }: GenerativeLabSectionProps) => {
  const [prompt, setPrompt] = useState(suggestedPrompt);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [showHowItWorks, setShowHowItWorks] = useState(false);

  const handleGenerate = () => {
    setIsGenerating(true);
    // Simulate generation
    setTimeout(() => {
      setIsGenerating(false);
      // For demo, we'll show a placeholder
      setGeneratedImage("generated");
    }, 3000);
  };

  return (
    <section className="py-12 px-6 bg-gradient-to-b from-transparent to-canvas-panel/50">
      <div className="max-w-5xl mx-auto">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center mb-10"
        >
          <h2 className="font-serif text-3xl md:text-4xl font-bold mb-3">
            <span className="text-gradient-ai">The Dream Studio</span>
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto">
            <LiteracyTooltip
              term="Latent Diffusion"
              definition="A generative model that learns to reverse a gradual noising process. It works in a compressed 'latent' space for efficiency, then decodes to full resolution."
            >
              <span className="literacy-term">Latent Diffusion</span>
            </LiteracyTooltip>{" "}
            — Transform analysis into new creations
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Prompt Input & Generate */}
          <div className="lg:col-span-2 space-y-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="glass-panel p-6"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-secondary/20 flex items-center justify-center">
                  <Wand2 className="w-5 h-5 text-secondary" />
                </div>
                <div>
                  <h3 className="font-serif text-lg font-semibold">Prompt Engineering</h3>
                  <span className="text-xs text-muted-foreground">Describe your vision</span>
                </div>
              </div>

              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter a prompt to generate new artwork..."
                className="w-full h-32 px-4 py-3 bg-canvas-deep border border-border rounded-lg text-foreground placeholder:text-muted-foreground/50 resize-none focus:outline-none focus:border-secondary/50 focus:ring-1 focus:ring-secondary/20 transition-all"
              />

              {/* Dev Mode Stats */}
              {devMode && (
                <div className="mt-3 p-3 bg-canvas-deep rounded-lg border border-secondary/30">
                  <div className="grid grid-cols-3 gap-2 text-xs font-mono">
                    <div>
                      <span className="text-muted-foreground">Scheduler: </span>
                      <span className="text-secondary">DDPM</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Steps: </span>
                      <span className="text-secondary">50</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">CFG: </span>
                      <span className="text-secondary">7.5</span>
                    </div>
                  </div>
                </div>
              )}

              <button
                onClick={handleGenerate}
                disabled={isGenerating || !prompt.trim()}
                className={cn(
                  "mt-4 w-full flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-medium transition-all duration-300",
                  isGenerating
                    ? "bg-secondary/20 text-secondary cursor-wait"
                    : "bg-gradient-ai text-secondary-foreground hover:shadow-glow-ai animate-pulse-glow"
                )}
              >
                {isGenerating ? (
                  <>
                    <div className="w-5 h-5 border-2 border-secondary border-t-transparent rounded-full animate-spin" />
                    Diffusing...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-5 h-5" />
                    Generate Artwork
                  </>
                )}
              </button>
            </motion.div>
          </div>

          {/* Output Canvas & How It Works */}
          <div className="space-y-4">
            {/* Generated Image Canvas */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="glass-panel p-4"
            >
              <div className="aspect-square rounded-lg bg-canvas-deep border border-border/50 flex items-center justify-center overflow-hidden relative">
                {isGenerating ? (
                  <div className="text-center">
                    <div className="w-16 h-16 mx-auto mb-4 border-4 border-secondary/30 border-t-secondary rounded-full animate-spin" />
                    <p className="text-sm text-muted-foreground font-mono">Denoising step 23/50...</p>
                  </div>
                ) : generatedImage ? (
                  <div className="w-full h-full bg-gradient-to-br from-secondary/20 via-accent/20 to-primary/20 flex items-center justify-center">
                    <div className="text-center">
                      <Sparkles className="w-12 h-12 text-secondary mx-auto mb-2" />
                      <p className="text-sm text-muted-foreground">Generated artwork would appear here</p>
                      <p className="text-xs text-muted-foreground/60 mt-1">(Model integration pending)</p>
                    </div>
                  </div>
                ) : (
                  <div className="text-center text-muted-foreground">
                    <div className="w-16 h-16 mx-auto mb-3 border-2 border-dashed border-muted-foreground/30 rounded-lg flex items-center justify-center">
                      <Sparkles className="w-8 h-8 opacity-30" />
                    </div>
                    <p className="text-sm">Generated image will appear here</p>
                  </div>
                )}
              </div>
            </motion.div>

            {/* How It Works */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="glass-panel overflow-hidden"
            >
              <button
                onClick={() => setShowHowItWorks(!showHowItWorks)}
                className="w-full p-4 flex items-center justify-between hover:bg-canvas-highlight transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Info className="w-4 h-4 text-accent" />
                  <span className="text-sm font-medium">How It Works</span>
                </div>
                <ChevronRight className={cn(
                  "w-4 h-4 text-muted-foreground transition-transform duration-300",
                  showHowItWorks && "rotate-90"
                )} />
              </button>
              
              {showHowItWorks && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="px-4 pb-4"
                >
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Unlike the <span className="text-accent font-mono text-xs">CNN</span> which reduces an image to labels, 
                    the <span className="text-secondary font-mono text-xs">Diffusion</span> model starts with random noise 
                    and iteratively 'denoises' it to match your text prompt. Each step refines the image, 
                    guided by a text encoder that understands your description.
                  </p>
                </motion.div>
              )}
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default GenerativeLabSection;
