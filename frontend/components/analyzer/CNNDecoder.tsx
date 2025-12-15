import { motion } from "framer-motion";
import { Brain, Layers } from "lucide-react";
import ProgressBar from "@/components/ui/ProgressBar";
import { InfoTooltip } from "@/components/ui/LiteracyTooltip";

interface CNNPrediction {
  label: string;
  confidence: number;
}

interface CNNDecoderProps {
  artistPredictions: CNNPrediction[];
  stylePredictions: CNNPrediction[];
  genrePredictions: CNNPrediction[];
  isAnalyzing: boolean;
  devMode: boolean;
}

const CNNDecoder = ({ 
  artistPredictions, 
  stylePredictions, 
  genrePredictions, 
  isAnalyzing,
  devMode 
}: CNNDecoderProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className="glass-panel p-6 h-full"
    >
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-lg bg-secondary/20 flex items-center justify-center">
          <Brain className="w-5 h-5 text-secondary" />
        </div>
        <div>
          <h3 className="font-serif text-lg font-semibold">Visual Cortex Analysis</h3>
          <span className="text-xs font-mono text-muted-foreground">(CNN)</span>
        </div>
      </div>

      {/* Dev Mode Stats */}
      {devMode && (
        <div className="mb-4 p-3 bg-canvas-deep rounded-lg border border-secondary/30">
          <div className="grid grid-cols-2 gap-2 text-xs font-mono">
            <div className="text-muted-foreground">Inference Time:</div>
            <div className="text-secondary">45ms</div>
            <div className="text-muted-foreground">Backbone:</div>
            <div className="text-secondary">ConvNeXt-Tiny</div>
            <div className="text-muted-foreground">Input Size:</div>
            <div className="text-secondary">224×224×3</div>
          </div>
        </div>
      )}

      <div className="space-y-6">
        {/* Artist Predictions */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <span className="text-sm font-medium text-primary">Artist Prediction</span>
            <span className="text-xs text-muted-foreground font-mono">(129 classes)</span>
            <InfoTooltip content="Softmax probability: The model's certainty distribution across all known artist classes in the WikiArt dataset." />
          </div>
          <div className="space-y-3">
            {artistPredictions.length > 0 ? (
              artistPredictions.map((pred, i) => (
                <ProgressBar
                  key={pred.label}
                  label={pred.label}
                  value={pred.confidence}
                  variant="gold"
                  delay={i * 0.15}
                />
              ))
            ) : (
              <div className="text-sm text-muted-foreground italic">Awaiting input...</div>
            )}
          </div>
        </div>

        {/* Style Predictions */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <span className="text-sm font-medium text-foreground">Style Prediction</span>
            <span className="text-xs text-muted-foreground font-mono">(27 classes)</span>
            <InfoTooltip content="Art movement classification based on visual patterns, brushwork, and compositional elements." />
          </div>
          <div className="space-y-3">
            {stylePredictions.length > 0 ? (
              stylePredictions.map((pred, i) => (
                <ProgressBar
                  key={pred.label}
                  label={pred.label}
                  value={pred.confidence}
                  delay={i * 0.15 + 0.3}
                />
              ))
            ) : (
              <div className="text-sm text-muted-foreground italic">Awaiting input...</div>
            )}
          </div>
        </div>

        {/* Genre Predictions */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <span className="text-sm font-medium text-foreground">Genre Prediction</span>
            <span className="text-xs text-muted-foreground font-mono">(11 classes)</span>
            <InfoTooltip content="Subject matter classification: landscape, portrait, abstract, still life, etc." />
          </div>
          <div className="space-y-3">
            {genrePredictions.length > 0 ? (
              genrePredictions.map((pred, i) => (
                <ProgressBar
                  key={pred.label}
                  label={pred.label}
                  value={pred.confidence}
                  delay={i * 0.15 + 0.6}
                />
              ))
            ) : (
              <div className="text-sm text-muted-foreground italic">Awaiting input...</div>
            )}
          </div>
        </div>

        {/* Analyzing State */}
        {isAnalyzing && (
          <div className="flex items-center gap-2 text-secondary text-sm">
            <Layers className="w-4 h-4 animate-pulse" />
            <span className="font-mono">Processing layers...</span>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default CNNDecoder;
