"use client";

import { motion, useScroll, useTransform } from "framer-motion";
import { useRef } from "react";
import { 
  Brain, 
  Sparkles, 
  Image, 
  Palette, 
  MessageSquare,
  Zap,
  Layers,
  Cpu,
  Wand2
} from "lucide-react";

interface ModelShowcaseProps {
  title: string;
  subtitle: string;
  description: string;
  icon: React.ReactNode;
  gradient: string;
  bgGradient: string;
  features: string[];
  index: number;
}

const ModelShowcase = ({ 
  title, 
  subtitle, 
  description, 
  icon, 
  gradient, 
  bgGradient,
  features,
  index 
}: ModelShowcaseProps) => {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start end", "end start"]
  });

  // Determine slide direction based on index (alternate left/right)
  const slideDirection = index % 2 === 0 ? -1 : 1;
  
  // Main content animations - more dramatic
  const opacity = useTransform(scrollYProgress, [0, 0.15, 0.85, 1], [0, 1, 1, 0]);
  const scale = useTransform(scrollYProgress, [0, 0.2, 0.8, 1], [0.7, 1, 1, 0.7]);
  const y = useTransform(scrollYProgress, [0, 0.3], [150, 0]);
  const x = useTransform(scrollYProgress, [0, 0.3], [slideDirection * 200, 0]);
  const rotateX = useTransform(scrollYProgress, [0, 0.3], [20, 0]);
  const rotateY = useTransform(scrollYProgress, [0, 0.3], [slideDirection * 10, 0]);
  const rotateZ = useTransform(scrollYProgress, [0, 0.3], [slideDirection * 5, 0]);
  
  // Parallax effects for background elements - more movement
  const bgY1 = useTransform(scrollYProgress, [0, 1], [0, -300]);
  const bgY2 = useTransform(scrollYProgress, [0, 1], [0, 300]);
  const bgX1 = useTransform(scrollYProgress, [0, 1], [0, slideDirection * 200]);
  const bgX2 = useTransform(scrollYProgress, [0, 1], [0, -slideDirection * 200]);
  const bgScale = useTransform(scrollYProgress, [0, 0.5, 1], [0.8, 1.3, 1]);
  const bgOpacity = useTransform(scrollYProgress, [0, 0.2, 0.8, 1], [0, 1, 1, 0]);

  return (
    <motion.section
      ref={ref}
      style={{ opacity: bgOpacity }}
      className="relative min-h-screen flex items-center justify-center overflow-hidden"
    >
      {/* Animated Background Layers */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {/* Base gradient background with animation */}
        <motion.div
          style={{ 
            background: bgGradient,
            y: bgY1,
            x: bgX1,
            scale: bgScale,
            opacity: bgOpacity
          }}
          className="absolute inset-0"
        />
        
        {/* Floating orbs with parallax - more dramatic movement */}
        <motion.div
          style={{ 
            background: gradient, 
            y: bgY1, 
            x: bgX1,
            scale: useTransform(scrollYProgress, [0, 0.5, 1], [0.5, 1.2, 1]),
            opacity: useTransform(scrollYProgress, [0, 0.2, 0.8, 1], [0, 0.6, 0.6, 0])
          }}
          className="absolute top-20 left-20 w-96 h-96 rounded-full blur-3xl"
        />
        <motion.div
          style={{ 
            background: gradient, 
            y: bgY2, 
            x: bgX2,
            scale: useTransform(scrollYProgress, [0, 0.5, 1], [0.5, 1.1, 1]),
            opacity: useTransform(scrollYProgress, [0, 0.2, 0.8, 1], [0, 0.6, 0.6, 0])
          }}
          className="absolute bottom-20 right-20 w-80 h-80 rounded-full blur-3xl"
        />
        <motion.div
          style={{ 
            background: gradient, 
            y: bgY1,
            x: bgX1,
            scale: useTransform(scrollYProgress, [0, 0.5, 1], [0.3, 1.5, 1]),
            opacity: useTransform(scrollYProgress, [0, 0.2, 0.8, 1], [0, 0.3, 0.3, 0])
          }}
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full blur-3xl"
        />
        
        {/* Animated grid pattern with rotation */}
        <motion.div 
          className="absolute inset-0"
          style={{
            backgroundImage: `linear-gradient(${index * 45}deg, transparent 0%, rgba(255,255,255,0.1) 50%, transparent 100%)`,
            backgroundSize: '100px 100px',
            opacity: useTransform(scrollYProgress, [0, 0.2, 0.8, 1], [0, 0.15, 0.15, 0]),
            rotate: useTransform(scrollYProgress, [0, 1], [0, 360])
          }}
        />
      </div>

      {/* Content - dramatic entrance animation */}
      <motion.div
        style={{ 
          scale, 
          y, 
          x,
          rotateX, 
          rotateY,
          rotateZ,
          opacity
        }}
        className="relative z-10 max-w-7xl mx-auto px-6 py-20 w-full"
      >
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Side - Icon and Title */}
          <motion.div
            initial={{ opacity: 0, x: slideDirection * -150, scale: 0.8 }}
            whileInView={{ opacity: 1, x: 0, scale: 1 }}
            viewport={{ once: false, margin: "-200px" }}
            transition={{ 
              duration: 1, 
              delay: 0.2,
              type: "spring",
              stiffness: 50
            }}
            className="space-y-8"
          >
            {/* Large Icon */}
            <motion.div
              initial={{ scale: 0, rotate: -180 }}
              whileInView={{ scale: 1, rotate: 0 }}
              viewport={{ once: true }}
              transition={{ 
                duration: 0.8, 
                delay: 0.3,
                type: "spring",
                stiffness: 100
              }}
              className="relative"
            >
              <motion.div
                animate={{ 
                  rotate: [0, 360],
                  scale: [1, 1.1, 1]
                }}
                transition={{ 
                  duration: 20, 
                  repeat: Infinity, 
                  ease: "linear" 
                }}
                className="absolute inset-0 rounded-full blur-2xl opacity-50"
                style={{ background: gradient }}
              />
              <div 
                className="relative w-32 h-32 md:w-40 md:h-40 rounded-3xl flex items-center justify-center text-white shadow-2xl"
                style={{ background: gradient }}
              >
                <div className="text-6xl md:text-7xl">
                  {icon}
                </div>
              </div>
            </motion.div>

            {/* Subtitle */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              <p className="text-sm md:text-base font-mono text-muted-foreground uppercase tracking-wider mb-2">
                {subtitle}
              </p>
            </motion.div>

            {/* Title */}
            <motion.h2
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, delay: 0.5 }}
              className="text-5xl md:text-7xl lg:text-8xl font-serif font-bold leading-tight"
              style={{ 
                background: gradient,
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text"
              }}
            >
              {title}
            </motion.h2>
          </motion.div>

          {/* Right Side - Description and Features */}
          <motion.div
            initial={{ opacity: 0, x: -slideDirection * 150, scale: 0.8 }}
            whileInView={{ opacity: 1, x: 0, scale: 1 }}
            viewport={{ once: false, margin: "-200px" }}
            transition={{ 
              duration: 1, 
              delay: 0.3,
              type: "spring",
              stiffness: 50
            }}
            className="space-y-8"
          >
            {/* Description */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.5 }}
              className="text-lg md:text-xl text-muted-foreground leading-relaxed"
            >
              {description}
            </motion.p>

            {/* Features Grid */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.6 }}
              className="grid grid-cols-1 sm:grid-cols-2 gap-4"
            >
              {features.map((feature, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ 
                    duration: 0.4, 
                    delay: 0.7 + idx * 0.1,
                    type: "spring"
                  }}
                  whileHover={{ scale: 1.05, x: 5 }}
                  className="glass-panel p-4 rounded-xl border border-border/50 group cursor-pointer"
                >
                  <div className="flex items-center gap-3">
                    <div 
                      className="w-10 h-10 rounded-lg flex items-center justify-center text-white flex-shrink-0"
                      style={{ background: gradient }}
                    >
                      <Zap className="w-5 h-5" />
                    </div>
                    <span className="text-foreground font-medium">{feature}</span>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </motion.div>
        </div>
      </motion.div>

    </motion.section>
  );
};

const ModelShowcaseSection = () => {
  const models = [
    {
      title: "CNN Art Analyzer",
      subtitle: "Deep Learning Vision",
      description: "A sophisticated convolutional neural network trained on WikiArt that classifies paintings by artist, style, and genre. Experience how AI perceives and understands visual art through learned representations and hierarchical feature extraction.",
      icon: <Brain />,
      gradient: "linear-gradient(135deg, hsl(258 90% 55%) 0%, hsl(217 91% 50%) 100%)",
      bgGradient: "linear-gradient(135deg, hsl(258 90% 20%) 0%, hsl(217 91% 20%) 100%)",
      features: [
        "Multi-class Classification",
        "Artist Identification",
        "Style Recognition",
        "Genre Detection"
      ]
    },
    {
      title: "Diffusion Model",
      subtitle: "Generative AI",
      description: "State-of-the-art generative AI that reimagines artworks through advanced diffusion processes. Watch as random noise transforms into beautiful, coherent artistic interpretations through iterative denoising.",
      icon: <Sparkles />,
      gradient: "linear-gradient(135deg, hsl(180 70% 50%) 0%, hsl(160 80% 45%) 100%)",
      bgGradient: "linear-gradient(135deg, hsl(180 70% 20%) 0%, hsl(160 80% 20%) 100%)",
      features: [
        "Image Generation",
        "Artistic Reimagining",
        "High-Quality Outputs",
        "Creative Transformations"
      ]
    },
    {
      title: "ESRGAN",
      subtitle: "Super-Resolution",
      description: "Enhanced Super-Resolution Generative Adversarial Network that upscales images while preserving fine details and textures. Perfect for enhancing low-resolution artwork with remarkable fidelity.",
      icon: <Image />,
      gradient: "linear-gradient(135deg, hsl(43 74% 45%) 0%, hsl(38 70% 50%) 100%)",
      bgGradient: "linear-gradient(135deg, hsl(43 74% 20%) 0%, hsl(38 70% 20%) 100%)",
      features: [
        "4x Upscaling",
        "Detail Preservation",
        "Anime-Optimized",
        "Real-Time Processing"
      ]
    },
    {
      title: "Neural Style Transfer",
      subtitle: "Artistic Fusion",
      description: "Transfer the artistic style of one image onto another using deep neural networks. Create unique hybrid artworks that seamlessly blend content and style through advanced feature matching.",
      icon: <Palette />,
      gradient: "linear-gradient(135deg, hsl(258 90% 55%) 0%, hsl(43 74% 45%) 100%)",
      bgGradient: "linear-gradient(135deg, hsl(258 90% 20%) 0%, hsl(43 74% 20%) 100%)",
      features: [
        "Style Blending",
        "Content Preservation",
        "Real-Time Generation",
        "Custom Style Mixing"
      ]
    },
    {
      title: "Model Arena",
      subtitle: "LLM Dialogue",
      description: "Interactive dialogue system powered by advanced language models that explain and interpret CNN predictions. Experience how AI translates visual understanding into natural, human-readable language.",
      icon: <MessageSquare />,
      gradient: "linear-gradient(135deg, hsl(217 91% 50%) 0%, hsl(43 74% 45%) 100%)",
      bgGradient: "linear-gradient(135deg, hsl(217 91% 20%) 0%, hsl(43 74% 20%) 100%)",
      features: [
        "LLM Explanations",
        "Multiple Model Options",
        "Natural Language Output",
        "Interactive Dialogue"
      ]
    }
  ];

  return (
    <div className="relative">
      {models.map((model, index) => (
        <ModelShowcase
          key={index}
          title={model.title}
          subtitle={model.subtitle}
          description={model.description}
          icon={model.icon}
          gradient={model.gradient}
          bgGradient={model.bgGradient}
          features={model.features}
          index={index}
        />
      ))}
    </div>
  );
};

export default ModelShowcaseSection;
