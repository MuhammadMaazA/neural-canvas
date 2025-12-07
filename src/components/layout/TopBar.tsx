"use client";

import { Lightbulb, LightbulbOff } from "lucide-react";
import { useTheme } from "next-themes";

const TopBar = () => {
  const { theme, setTheme } = useTheme();

  return (
    <header className="fixed top-0 left-16 right-0 z-30 h-14 bg-background/80 backdrop-blur-xl border-b border-border flex items-center justify-between px-6">
      {/* Title */}
      <div className="flex items-center gap-3">
        <h1 className="font-serif text-lg font-semibold">
          <span className="text-gradient-gold">Neural</span>{" "}
          <span className="text-foreground">Canvas</span>
        </h1>
        <span className="text-xs text-muted-foreground font-mono">v1.0</span>
      </div>

      {/* Theme Toggle */}
      <button
        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        className="p-2 rounded-lg border border-border text-muted-foreground hover:text-primary hover:border-primary transition-all duration-300"
      >
        {theme === "dark" ? (
          <Lightbulb className="w-4 h-4" />
        ) : (
          <LightbulbOff className="w-4 h-4" />
        )}
      </button>
    </header>
  );
};

export default TopBar;
