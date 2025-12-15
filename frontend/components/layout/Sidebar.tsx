import { Brain, Eye, Palette, Zap, Sparkles, Settings, Info, MessageSquare, ScanEye, Wand2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface SidebarProps {
  activeSection: string;
  onSectionChange: (section: string) => void;
}

const navItems = [
  { id: "analyzer", icon: Eye, label: "Analyzer", path: "/" },
  { id: "dialogue", icon: MessageSquare, label: "Text Gen", path: "/model-arena" },
  { id: "cnn", icon: ScanEye, label: "CNN Arena", path: "/cnn-arena" },
  { id: "diffusion", icon: Wand2, label: "Diffusion", path: "/diffusion-lab" },
  { id: "esrgan", icon: Zap, label: "ESRGAN", path: "/esrgan-lab" },
  { id: "nst", icon: Palette, label: "NST", path: "/nst-lab" },
];

const Sidebar = ({ activeSection, onSectionChange }: SidebarProps) => {
  return (
    <aside className="fixed left-0 top-0 z-50 h-screen w-16 bg-sidebar border-r border-sidebar-border flex flex-col items-center py-6" style={{ pointerEvents: 'auto' }}>
      {/* Logo */}
      <div className="mb-8">
        <div className="w-10 h-10 rounded-lg bg-gradient-ai flex items-center justify-center">
          <Brain className="w-6 h-6 text-primary-foreground" />
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 flex flex-col items-center gap-2">
        {navItems.map((item) => {
          const isActive = activeSection === item.id;
          
          const content = (
            <>
              <item.icon className="w-5 h-5" />
              {/* Tooltip */}
              <span className="absolute left-14 px-2 py-1 bg-card border border-border rounded text-xs font-sans opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                {item.label}
              </span>
            </>
          );
          
          const className = cn(
            "w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-300 group relative",
            isActive
              ? "bg-secondary/20 text-secondary shadow-glow-ai"
              : "text-muted-foreground hover:text-foreground hover:bg-muted"
          );
          
          return (
            <button
              key={item.id}
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                console.log('Button clicked:', item.id);
                onSectionChange(item.id);
              }}
              className={className}
              title={item.label}
              type="button"
              style={{ pointerEvents: 'auto', zIndex: 50, cursor: 'pointer' }}
            >
              {content}
            </button>
          );
        })}
      </nav>

      {/* Bottom actions */}
      <div className="flex flex-col items-center gap-2">
        <button
          onClick={() => {
            alert('Neural Canvas v1.0\n\nDeep Learning Coursework (COMP0220)\n\nFeatures:\n• CNN Models: Custom (56M) + Fine-tuned\n• LLM Models:\n  - The Architect (56M from scratch)\n  - The Curator (355M GPT-2 fine-tuned)\n  - The Oracle (Groq Llama 3.2 1B)\n\nBuilt with React, Flask, PyTorch');
          }}
          className="w-10 h-10 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-muted transition-all duration-300 group relative"
          title="About"
        >
          <Info className="w-5 h-5" />
          <span className="absolute left-14 px-2 py-1 bg-card border border-border rounded text-xs font-sans opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
            About
          </span>
        </button>
        <button
          onClick={() => {
            alert('Settings\n\n• Theme: Toggle in top bar\n• Models: All 3 LLM models active\n• Backend: http://localhost:5000\n• API Status: Healthy\n\nFor advanced settings, check the backend configuration.');
          }}
          className="w-10 h-10 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-muted transition-all duration-300 group relative"
          title="Settings"
        >
          <Settings className="w-5 h-5" />
          <span className="absolute left-14 px-2 py-1 bg-card border border-border rounded text-xs font-sans opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
            Settings
          </span>
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;
