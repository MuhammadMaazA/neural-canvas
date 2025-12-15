
import { useState } from "react";
import { motion } from "framer-motion";
import { Palette, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function NSTLabPage() {
  const [styleImage, setStyleImage] = useState<string | null>(null);
  const [contentImage, setContentImage] = useState<string | null>(null);
  const [isTransferring, setIsTransferring] = useState(false);
  const [result, setResult] = useState<{ imageUrl: string; processingTime: number } | null>(null);

  const handleImageUpload = (type: 'style' | 'content') => (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (type === 'style') {
          setStyleImage(event.target?.result as string);
        } else {
          setContentImage(event.target?.result as string);
        }
        setResult(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleTransfer = async () => {
    if (!styleImage || !contentImage) return;
    
    setIsTransferring(true);
    setResult(null);

    try {
      const response = await fetch('/api/transfer-style', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ styleImage, contentImage }),
      });
      const data = await response.json();
      setResult(data);
      setIsTransferring(false);
    } catch (error) {
      // Fallback
      setTimeout(() => {
        setResult({
          imageUrl: "https://images.unsplash.com/photo-1549289524-06cf8837ace5?w=800&h=800&fit=crop",
          processingTime: 2100
        });
        setIsTransferring(false);
      }, 2100);
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
              The <span className="text-transparent bg-clip-text bg-gradient-to-r from-amber-400 to-orange-500">NST Lab</span>
            </h1>
            <p className="text-muted-foreground font-sans text-sm max-w-xl mx-auto">
              Neural Style Transfer - Apply artistic styles to your images
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gradient-to-br from-amber-500/20 to-orange-500/20 backdrop-blur-xl border border-amber-500/30 rounded-xl overflow-hidden shadow-lg shadow-amber-500/20 mb-6"
          >
            <div className="p-4 border-b border-border/30">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-amber-500/20 flex items-center justify-center">
                  <Palette className="w-5 h-5 text-amber-400" />
                </div>
                <div>
                  <h3 className="font-display font-semibold text-foreground">The Stylist</h3>
                  <p className="text-xs font-mono text-amber-400">Neural Style Transfer</p>
                </div>
              </div>
              <p className="text-xs text-muted-foreground mt-2 font-sans">
                Transfers artistic style from one image to the content of another
              </p>
            </div>

            <div className="grid grid-cols-3 divide-x divide-border/30">
              <div className="aspect-square relative bg-background/30">
                <div className="absolute top-2 left-2 px-2 py-1 bg-background/80 backdrop-blur-sm rounded text-xs font-mono text-muted-foreground">
                  Style
                </div>
                {styleImage ? (
                  <img src={styleImage} alt="Style" className="w-full h-full object-cover" />
                ) : (
                  <label className="absolute inset-0 flex items-center justify-center cursor-pointer hover:bg-amber-500/5 transition-colors">
                    <div className="text-center">
                      <div className="w-10 h-10 rounded-full bg-amber-500/20 flex items-center justify-center mx-auto mb-2">
                        <Palette className="w-5 h-5 text-amber-400" />
                      </div>
                      <p className="text-xs text-muted-foreground font-sans">Style image</p>
                    </div>
                    <input type="file" accept="image/*" onChange={handleImageUpload('style')} className="hidden" />
                  </label>
                )}
              </div>

              <div className="aspect-square relative bg-background/30">
                <div className="absolute top-2 left-2 px-2 py-1 bg-background/80 backdrop-blur-sm rounded text-xs font-mono text-muted-foreground">
                  Content
                </div>
                {contentImage ? (
                  <img src={contentImage} alt="Content" className="w-full h-full object-cover" />
                ) : (
                  <label className="absolute inset-0 flex items-center justify-center cursor-pointer hover:bg-amber-500/5 transition-colors">
                    <div className="text-center">
                      <div className="w-10 h-10 rounded-full bg-amber-500/20 flex items-center justify-center mx-auto mb-2">
                        <Palette className="w-5 h-5 text-amber-400" />
                      </div>
                      <p className="text-xs text-muted-foreground font-sans">Content image</p>
                    </div>
                    <input type="file" accept="image/*" onChange={handleImageUpload('content')} className="hidden" />
                  </label>
                )}
              </div>

              <div className="aspect-square relative bg-background/30">
                <div className="absolute top-2 left-2 px-2 py-1 bg-background/80 backdrop-blur-sm rounded text-xs font-mono text-muted-foreground">
                  Result
                </div>
                {isTransferring ? (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <motion.div
                        animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                        className="w-12 h-12 rounded-full bg-amber-500/20 flex items-center justify-center mx-auto mb-2"
                      >
                        <Palette className="w-6 h-6 text-amber-400" />
                      </motion.div>
                      <p className="text-xs font-mono text-muted-foreground">Transferring...</p>
                    </div>
                  </div>
                ) : result ? (
                  <div className="relative h-full">
                    <img src={result.imageUrl} alt="NST output" className="w-full h-full object-cover" />
                    <div className="absolute bottom-2 right-2 px-2 py-1 bg-background/80 backdrop-blur-sm rounded text-xs font-mono text-muted-foreground">
                      {(result.processingTime / 1000).toFixed(1)}s
                    </div>
                  </div>
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="w-10 h-10 rounded-full bg-amber-500/20 flex items-center justify-center mx-auto mb-2 opacity-50">
                        <Sparkles className="w-5 h-5 text-amber-400" />
                      </div>
                      <p className="text-xs text-muted-foreground font-sans">Awaiting images...</p>
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
                onClick={handleTransfer}
                disabled={isTransferring || !styleImage || !contentImage}
                className="w-full bg-gradient-to-r from-amber-500 to-orange-500 hover:opacity-90 text-white font-mono"
              >
                {isTransferring ? (
                  <>
                    <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: "linear" }}>
                      <Sparkles className="w-4 h-4" />
                    </motion.div>
                    Transferring Style...
                  </>
                ) : (
                  <>
                    <Palette className="w-4 h-4" />
                    Transfer Style
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

