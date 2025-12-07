"use client";

import { Brain, Eye, Palette, Zap, Sparkles, Settings, Info, MessageSquare, ScanEye, Wand2 } from "lucide-react";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { usePathname } from "next/navigation";

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
    <aside className="fixed left-0 top-0 z-40 h-screen w-16 bg-sidebar border-r border-sidebar-border flex flex-col items-center py-6">
      {/* Logo */}
      <div className="mb-8">
        <div className="w-10 h-10 rounded-lg bg-gradient-gold flex items-center justify-center">
          <Brain className="w-6 h-6 text-primary-foreground" />
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 flex flex-col items-center gap-2">
        {navItems.map((item) => {
          const pathname = usePathname();
          const isActive = item.path ? pathname === item.path : activeSection === item.id;
          
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
          
          if (item.path) {
            return (
              <Link key={item.id} href={item.path} className={className} title={item.label}>
                {content}
              </Link>
            );
          }
          
          return (
            <button
              key={item.id}
              onClick={() => onSectionChange(item.id)}
              className={className}
              title={item.label}
            >
              {content}
            </button>
          );
        })}
      </nav>

      {/* Bottom actions */}
      <div className="flex flex-col items-center gap-2">
        <button className="w-10 h-10 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-muted transition-all duration-300">
          <Info className="w-5 h-5" />
        </button>
        <button className="w-10 h-10 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-muted transition-all duration-300">
          <Settings className="w-5 h-5" />
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;
