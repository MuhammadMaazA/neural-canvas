import { useState, useEffect, useRef } from "react";
import { motion, useScroll, useMotionValueEvent } from "framer-motion";
import { ThemeProvider } from "@/components/providers/theme-provider";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { QueryClientProviderWrapper } from "@/components/providers/query-provider";
import Sidebar from "@/components/layout/Sidebar";
import TopBar from "@/components/layout/TopBar";
import HeroSection from "@/components/sections/HeroSection";
import ModelShowcaseSection from "@/components/sections/ModelShowcaseSection";
import Footer from "@/components/sections/Footer";
import ModelArenaPage from "@/src/pages/ModelArenaPage";
import CNNArenaPage from "@/src/pages/CNNArenaPage";
import DiffusionLabPage from "@/src/pages/DiffusionLabPage";
import ESRGANLabPage from "@/src/pages/ESRGANLabPage";
import NSTLabPage from "@/src/pages/NSTLabPage";

function App() {
  const [activeSection, setActiveSection] = useState("analyzer");
  const [devMode, setDevMode] = useState(false);
  const [showNav, setShowNav] = useState(true);
  const mainRef = useRef<HTMLElement>(null);
  const heroRef = useRef<HTMLDivElement>(null);
  const footerRef = useRef<HTMLDivElement>(null);
  
  // Use window scroll for detection
  const { scrollYProgress } = useScroll();

  // Hide nav when scrolling past hero section, show again near end
  useMotionValueEvent(scrollYProgress, "change", (latest) => {
    if (activeSection === "analyzer") {
      // Hide nav when scrolled past ~10% (past hero section - 25% of hero height)
      // Show nav again when near the end (~90% or more)
      setShowNav(latest < 0.1 || latest > 0.9);
    } else {
      setShowNav(true);
    }
  });

  // Also check on scroll for immediate feedback - using window scroll
  useEffect(() => {
    if (activeSection !== "analyzer") {
      setShowNav(true);
      return;
    }

    const handleScroll = () => {
      const scrollTop = window.scrollY || document.documentElement.scrollTop;
      const scrollHeight = document.documentElement.scrollHeight;
      const clientHeight = window.innerHeight;
      const heroHeight = heroRef.current?.offsetHeight || 800;
      
      // Calculate distance from bottom
      const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
      const nearBottom = distanceFromBottom < 500; // Show nav when within 500px of bottom
      
      // Hide nav when scrolled past hero section (25% threshold)
      const pastHero = scrollTop > heroHeight * 0.25;
      const shouldShow = !pastHero || nearBottom;
      
      setShowNav(shouldShow);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll(); // Check initial position
    return () => window.removeEventListener("scroll", handleScroll);
  }, [activeSection]);
  
  // Debug: log section changes
  useEffect(() => {
    console.log('✅ Active section changed to:', activeSection);
  }, [activeSection]);
  
  const handleSectionChange = (section: string) => {
    console.log('🔄 Section change requested:', section);
    setActiveSection(section);
    setShowNav(true); // Show nav when changing sections
  };

  return (
    <QueryClientProviderWrapper>
      <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false}>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <div className="min-h-screen bg-black">
            <motion.div
              animate={{ 
                opacity: showNav ? 1 : 0,
                x: showNav ? 0 : -100,
                scale: showNav ? 1 : 0.8,
                pointerEvents: showNav ? "auto" : "none"
              }}
              transition={{ duration: 0.4, ease: "easeInOut" }}
              className="fixed left-0 top-0 z-50"
              style={{ visibility: showNav ? "visible" : "hidden" }}
            >
              <Sidebar activeSection={activeSection} onSectionChange={handleSectionChange} />
            </motion.div>
            
            <motion.div
              animate={{ 
                opacity: showNav ? 1 : 0,
                y: showNav ? 0 : -100,
                scale: showNav ? 1 : 0.9,
                pointerEvents: showNav ? "auto" : "none"
              }}
              transition={{ duration: 0.4, ease: "easeInOut" }}
              className={`fixed top-0 z-40 transition-all duration-300 ${
                showNav ? "left-16 right-0" : "left-0 right-0"
              }`}
              style={{ visibility: showNav ? "visible" : "hidden" }}
            >
              <TopBar />
            </motion.div>
            
            <main 
              ref={mainRef}
              className={`min-h-screen overflow-x-hidden scrollbar-thin bg-black transition-all duration-300 ${
                showNav ? "pl-16 pt-14" : "pl-0 pt-0"
              }`}
            >
              {activeSection === "analyzer" && (
                <>
                  <div ref={heroRef}>
                    <HeroSection />
                  </div>
                  <ModelShowcaseSection />
                  <div ref={footerRef}>
                    <Footer />
                  </div>
                </>
              )}
              
              {activeSection === "dialogue" && <ModelArenaPage />}
              {activeSection === "cnn" && <CNNArenaPage />}
              {activeSection === "diffusion" && <DiffusionLabPage />}
              {activeSection === "esrgan" && <ESRGANLabPage />}
              {activeSection === "nst" && <NSTLabPage />}
            </main>
          </div>
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProviderWrapper>
  );
}

export default App;

