import { motion } from "framer-motion";
import ImageDropzone from "./ImageDropzone";
import CNNDecoder from "./CNNDecoder";
import LLMInterpreter from "./LLMInterpreter";

interface AnalyzerSectionProps {
  loadedImage: string | null;
  isScanning: boolean;
  onImageLoad: (imageUrl: string) => void;
  onLoadDemo: () => void;
  artistPredictions: { label: string; confidence: number }[];
  stylePredictions: { label: string; confidence: number }[];
  genrePredictions: { label: string; confidence: number }[];
  selectedModel: "scratch" | "distilgpt2" | "hosted";
  onModelChange: (model: "scratch" | "distilgpt2" | "hosted") => void;
  llmOutput: string;
  isGenerating: boolean;
  devMode: boolean;
}

const AnalyzerSection = ({
  loadedImage,
  isScanning,
  onImageLoad,
  onLoadDemo,
  artistPredictions,
  stylePredictions,
  genrePredictions,
  selectedModel,
  onModelChange,
  llmOutput,
  isGenerating,
  devMode,
}: AnalyzerSectionProps) => {
  return (
    <section className="py-12 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center mb-10"
        >
          <h2 className="font-serif text-3xl md:text-4xl font-bold mb-3">
            <span className="text-gradient-ai">The Analyzer</span>
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Upload an artwork to analyze its visual patterns through our neural network pipeline
          </p>
        </motion.div>

        {/* Triptych Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Column 1: Input & Visual */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="glass-panel p-6"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center">
                <span className="text-xl">🖼️</span>
              </div>
              <div>
                <h3 className="font-serif text-lg font-semibold">Input Canvas</h3>
                <span className="text-xs text-muted-foreground">Upload or select artwork</span>
              </div>
            </div>
            
            <ImageDropzone
              onImageLoad={onImageLoad}
              isScanning={isScanning}
              loadedImage={loadedImage}
              onLoadDemo={onLoadDemo}
            />
          </motion.div>

          {/* Column 2: CNN Decoder */}
          <CNNDecoder
            artistPredictions={artistPredictions}
            stylePredictions={stylePredictions}
            genrePredictions={genrePredictions}
            isAnalyzing={isScanning}
            devMode={devMode}
          />

          {/* Column 3: LLM Interpreter */}
          <LLMInterpreter
            selectedModel={selectedModel}
            onModelChange={onModelChange}
            outputText={llmOutput}
            isGenerating={isGenerating}
            devMode={devMode}
          />
        </div>
      </div>
    </section>
  );
};

export default AnalyzerSection;
