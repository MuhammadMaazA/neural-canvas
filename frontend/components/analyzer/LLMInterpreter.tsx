import { motion } from "framer-motion";
import { MessageSquare, Terminal } from "lucide-react";
import TypewriterText from "@/components/ui/TypewriterText";
import LiteracyTooltip from "@/components/ui/LiteracyTooltip";
import { cn } from "@/lib/utils";

interface LLMInterpreterProps {
  selectedModel: "scratch" | "distilgpt2" | "hosted";
  onModelChange: (model: "scratch" | "distilgpt2" | "hosted") => void;
  outputText: string;
  isGenerating: boolean;
  devMode: boolean;
}

const MODEL_SPECS = {
  scratch: { params: "35-56M", label: "Custom Transformer" },
  distilgpt2: { params: "355M", label: "Fine-tuned GPT-2 Medium" },
  hosted: { params: "1B", label: "Groq Llama 3.2 API" },
};

const LLMInterpreter = ({ 
  selectedModel, 
  onModelChange, 
  outputText, 
  isGenerating,
  devMode 
}: LLMInterpreterProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="glass-panel p-6 h-full flex flex-col"
    >
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-lg bg-accent/20 flex items-center justify-center">
          <MessageSquare className="w-5 h-5 text-accent" />
        </div>
        <div>
          <h3 className="font-serif text-lg font-semibold">Natural Language Synthesis</h3>
          <span className="text-xs font-mono text-muted-foreground">
            <LiteracyTooltip 
              term="Transformer" 
              definition="A neural network architecture that uses self-attention mechanisms to process sequences. Unlike RNNs, it can attend to all positions simultaneously."
            >
              <span className="literacy-term">(Transformer)</span>
            </LiteracyTooltip>
          </span>
        </div>
      </div>

      {/* Model Toggle - Three Options */}
      <div className="mb-4">
        <div className="p-1 bg-canvas-deep rounded-xl flex gap-1">
          <button
            onClick={() => onModelChange("scratch")}
            className={cn(
              "flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-all duration-300",
              selectedModel === "scratch"
                ? "bg-gradient-ai text-secondary-foreground shadow-glow-ai"
                : "text-muted-foreground hover:text-foreground hover:bg-canvas-highlight"
            )}
          >
            <div className="text-center">
              <div className="font-semibold">Custom Model</div>
              <div className="text-[9px] opacity-70">35-56M • From Scratch</div>
            </div>
          </button>
          <button
            onClick={() => onModelChange("distilgpt2")}
            className={cn(
              "flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-all duration-300",
              selectedModel === "distilgpt2"
                ? "bg-gradient-ai text-secondary-foreground shadow-glow-ai"
                : "text-muted-foreground hover:text-foreground hover:bg-canvas-highlight"
            )}
          >
            <div className="text-center">
              <div className="font-semibold">GPT-2 Medium</div>
              <div className="text-[9px] opacity-70">355M • Fine-tuned</div>
            </div>
          </button>
          <button
            onClick={() => onModelChange("hosted")}
            className={cn(
              "flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-all duration-300",
              selectedModel === "hosted"
                ? "bg-gradient-ai text-secondary-foreground shadow-glow-ai"
                : "text-muted-foreground hover:text-foreground hover:bg-canvas-highlight"
            )}
          >
            <div className="text-center">
              <div className="font-semibold">Llama 3.2</div>
              <div className="text-[9px] opacity-70">1B • Groq</div>
            </div>
          </button>
        </div>
        
        {/* Model specs */}
        <div className="mt-2 text-center text-xs text-muted-foreground font-mono">
          <span>{MODEL_SPECS[selectedModel].params} Params | {MODEL_SPECS[selectedModel].label}</span>
        </div>
      </div>

      {/* Dev Mode Stats */}
      {devMode && (
        <div className="mb-4 p-3 bg-canvas-deep rounded-lg border border-accent/30">
          <div className="grid grid-cols-2 gap-2 text-xs font-mono">
            <div className="text-muted-foreground">Token/sec:</div>
            <div className="text-accent">
              {selectedModel === "scratch" ? "120" : selectedModel === "distilgpt2" ? "85" : "45"}
            </div>
            <div className="text-muted-foreground">Temperature:</div>
            <div className="text-accent">0.7</div>
            <div className="text-muted-foreground">Max Tokens:</div>
            <div className="text-accent">{selectedModel === "hosted" ? "4096" : "256"}</div>
            {selectedModel === "hosted" && (
              <>
                <div className="text-muted-foreground">Endpoint:</div>
                <div className="text-primary truncate">api.groq.com</div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Terminal Output Box */}
      <div className="flex-1 terminal-window p-4 overflow-auto">
        <div className="flex items-center gap-2 mb-3 pb-2 border-b border-border/30">
          <Terminal className="w-4 h-4 text-secondary" />
          <span className="text-xs font-mono text-muted-foreground">output.txt</span>
        </div>
        
        <div className="text-sm leading-relaxed text-foreground/90">
          {outputText ? (
            <TypewriterText text={outputText} speed={15} />
          ) : isGenerating ? (
            <div className="flex items-center gap-2 text-muted-foreground">
              <span className="animate-pulse">●</span>
              <span>Generating response...</span>
            </div>
          ) : (
            <span className="text-muted-foreground italic">
              Analysis output will appear here...
            </span>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default LLMInterpreter;
