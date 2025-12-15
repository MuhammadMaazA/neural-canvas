import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Frame, Upload, Shuffle } from "lucide-react";
import { cn } from "@/lib/utils";

interface ImageDropzoneProps {
  onImageLoad: (imageUrl: string) => void;
  isScanning: boolean;
  loadedImage: string | null;
  onLoadDemo: () => void;
}

const ImageDropzone = ({ onImageLoad, isScanning, loadedImage, onLoadDemo }: ImageDropzoneProps) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          onImageLoad(event.target.result as string);
        }
      };
      reader.readAsDataURL(file);
    }
  }, [onImageLoad]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          onImageLoad(event.target.result as string);
        }
      };
      reader.readAsDataURL(file);
    }
  }, [onImageLoad]);

  return (
    <div className="space-y-4">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={cn(
          "relative aspect-square rounded-xl overflow-hidden transition-all duration-300",
          loadedImage 
            ? "border border-border" 
            : cn("dropzone", isDragging && "dropzone-active")
        )}
      >
        <AnimatePresence mode="wait">
          {!loadedImage ? (
            <motion.label
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 flex flex-col items-center justify-center cursor-pointer"
            >
              <input
                type="file"
                accept="image/*"
                onChange={handleFileInput}
                className="hidden"
              />
              <Frame className="w-12 h-12 text-muted-foreground mb-4" />
              <span className="text-muted-foreground text-sm">Drag Artwork Here</span>
              <span className="text-muted-foreground/60 text-xs mt-1">or click to upload</span>
            </motion.label>
          ) : (
            <motion.div
              key="loaded"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="relative w-full h-full"
            >
              <img
                src={loadedImage}
                alt="Uploaded artwork"
                className="w-full h-full object-cover"
              />
              
              {/* Shadow overlay for depth */}
              <div className="absolute inset-0 shadow-[inset_0_0_60px_rgba(0,0,0,0.4)]" />
              
              {/* Scanning overlay */}
              {isScanning && (
                <div className="absolute inset-0 bg-secondary/5">
                  <div className="scanning-line" />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="glass-panel px-4 py-2 text-xs font-mono text-secondary">
                      Analyzing neural patterns...
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        <button
          onClick={onLoadDemo}
          className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 glass-panel hover:bg-canvas-highlight transition-all duration-300 text-sm font-medium"
        >
          <Shuffle className="w-4 h-4" />
          Load Demo Image
        </button>
        
        {loadedImage && (
          <label className="flex items-center justify-center gap-2 px-4 py-2.5 glass-panel hover:bg-canvas-highlight transition-all duration-300 cursor-pointer">
            <Upload className="w-4 h-4" />
            <input
              type="file"
              accept="image/*"
              onChange={handleFileInput}
              className="hidden"
            />
          </label>
        )}
      </div>
    </div>
  );
};

export default ImageDropzone;
