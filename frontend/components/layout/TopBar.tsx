import { useState, useEffect } from "react";
import { Lightbulb, LightbulbOff } from "lucide-react";
import { useTheme } from "next-themes";

const TopBar = () => {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // Prevent hydration mismatch by only rendering after mount
  useEffect(() => {
    setMounted(true);
  }, []);

  const toggleTheme = () => {
    if (theme) {
      setTheme(theme === "dark" ? "light" : "dark");
    } else {
      // If theme is not yet available, default to switching from dark
      setTheme("light");
    }
  };

  return (
    <header className="fixed top-0 left-16 right-0 z-30 h-14 bg-background/80 backdrop-blur-xl border-b border-border flex items-center justify-between px-6">
      {/* Title */}
      <div className="flex items-center gap-3">
        <h1 className="font-serif text-lg font-semibold">
          <span className="text-gradient-ai">Neural</span>{" "}
          <span className="text-foreground">Canvas</span>
        </h1>
        <span className="text-xs text-muted-foreground font-mono">v1.0</span>
      </div>

      {/* Theme Toggle */}
      {mounted ? (
        <button
          onClick={toggleTheme}
          className="p-2 rounded-lg border border-border text-muted-foreground hover:text-primary hover:border-primary transition-all duration-300"
          aria-label="Toggle theme"
        >
          {theme === "dark" ? (
            <Lightbulb className="w-4 h-4" />
          ) : (
            <LightbulbOff className="w-4 h-4" />
          )}
        </button>
      ) : (
        <div className="p-2 w-10 h-10" />
      )}
    </header>
  );
};

export default TopBar;
