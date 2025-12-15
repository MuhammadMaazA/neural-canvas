import { motion, useScroll, useTransform } from "framer-motion";
import { useRef } from "react";

const HeroSection = () => {
  const sectionRef = useRef<HTMLElement>(null);
  
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start start", "end start"]
  });

  // Zoom effect - starts large, zooms in as you scroll (centered)
  const scale = useTransform(scrollYProgress, [0, 0.6], [1, 3]);
  const opacity = useTransform(scrollYProgress, [0, 0.4, 0.8, 1], [1, 1, 0.5, 0]);
  
  // Background elements fade out
  const bgOpacity = useTransform(scrollYProgress, [0, 0.5], [1, 0]);
  
  // Subtitle fades out earlier
  const subtitleOpacity = useTransform(scrollYProgress, [0, 0.2], [1, 0]);
  const subtitleY = useTransform(scrollYProgress, [0, 0.2], [0, 50]);

  return (
    <section 
      ref={sectionRef}
      className="relative min-h-screen flex items-center justify-center overflow-hidden"
    >
      {/* Background gradient orbs */}
      <motion.div 
        style={{ opacity: bgOpacity }}
        className="absolute inset-0 overflow-hidden pointer-events-none"
      >
        <div className="absolute -top-40 -right-40 w-96 h-96 bg-secondary/10 rounded-full blur-3xl animate-breathe" />
        <div className="absolute -bottom-20 -left-20 w-80 h-80 bg-primary/5 rounded-full blur-3xl animate-breathe" style={{ animationDelay: "1.5s" }} />
      </motion.div>

      {/* Centered container that stays in place during zoom */}
      <div className="absolute inset-0 flex items-center justify-center">
        <motion.div 
          style={{ 
            scale, 
            opacity,
            transformOrigin: "center center"
          }}
          className="text-center px-6"
        >
          {/* Main Headline - Very large, zooms in on scroll */}
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="font-serif text-7xl md:text-9xl lg:text-[12rem] font-bold leading-none whitespace-nowrap"
          >
            <span className="text-gradient-ai">Neural</span>{" "}
            <span className="text-foreground">Canvas</span>
          </motion.h1>

          {/* Sub-headline - fades out early */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
            style={{ opacity: subtitleOpacity, y: subtitleY }}
            className="text-xl md:text-2xl text-muted-foreground font-light mt-8"
          >
            Bridging Human Creativity and Machine Vision
          </motion.p>
        </motion.div>
      </div>
    </section>
  );
};

export default HeroSection;
