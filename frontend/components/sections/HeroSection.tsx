import { motion } from "framer-motion";
import { X } from "lucide-react";
import { useState } from "react";

const HeroSection = () => {
  const [showMission, setShowMission] = useState(true);

  return (
    <section className="relative py-16 px-6 overflow-hidden">
      {/* Background gradient orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-96 h-96 bg-secondary/10 rounded-full blur-3xl animate-breathe" />
        <div className="absolute -bottom-20 -left-20 w-80 h-80 bg-primary/5 rounded-full blur-3xl animate-breathe" style={{ animationDelay: "1.5s" }} />
      </div>

      <div className="relative max-w-4xl mx-auto text-center">
        {/* Main Headline */}
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="font-serif text-5xl md:text-7xl font-bold mb-6 leading-tight"
        >
          <span className="text-gradient-ai">Neural</span>{" "}
          <span className="text-foreground">Canvas</span>
        </motion.h1>

        {/* Sub-headline */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
          className="text-xl md:text-2xl text-muted-foreground font-light mb-10"
        >
          Bridging Human Creativity and Machine Vision
        </motion.p>

        {/* Mission Statement Card */}
        {showMission && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="relative glass-panel p-6 max-w-2xl mx-auto"
          >
            <button
              onClick={() => setShowMission(false)}
              className="absolute top-3 right-3 text-muted-foreground hover:text-foreground transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
            
            <p className="text-sm text-muted-foreground leading-relaxed">
              This platform demonstrates a{" "}
              <span className="text-secondary font-medium">complete Deep Learning pipeline</span>. 
              We use a <span className="text-accent font-mono text-xs">CNN</span> trained on WikiArt to{" "}
              <em>'see'</em> artworks, then three distinct{" "}
              <span className="text-accent font-mono text-xs">LLMs</span> to{" "}
              <em>'explain'</em> them — from a custom transformer built from scratch (35M), 
              to a fine-tuned GPT-2 Medium (355M), to a hosted Llama 3.2 (1B) API. 
              Finally, <span className="text-accent font-mono text-xs">Diffusion</span> and <span className="text-accent font-mono text-xs">Super-Resolution</span> models reimagine and enhance the art.{" "}
              <span className="text-primary">Explore the boundaries of AI and Art.</span>
            </p>
          </motion.div>
        )}

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1, duration: 0.5 }}
          className="mt-16"
        >
          <div className="w-6 h-10 border-2 border-muted-foreground/30 rounded-full mx-auto flex justify-center">
            <motion.div
              animate={{ y: [0, 12, 0] }}
              transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
              className="w-1.5 h-3 bg-secondary rounded-full mt-2"
            />
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default HeroSection;
