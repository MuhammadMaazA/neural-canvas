import { motion } from "framer-motion";
import { LucideIcon, Clock, Activity, Gauge, Terminal } from "lucide-react";
import TypewriterText from "@/components/ui/TypewriterText";
import { cn } from "@/lib/utils";

interface ModelOutputCardProps {
  title: string;
  subtitle: string;
  params: string;
  statusLabel: string;
  icon: LucideIcon;
  output: string;
  isLoading: boolean;
  typingSpeed: number;
  variant: "scratch" | "finetuned" | "hosted";
  devMode: boolean;
  metrics: {
    inferenceTime: string;
    perplexity: string;
    tokensPerSec: string;
  };
}

const variantConfig = {
  scratch: {
    gradient: "from-red-500/20 via-orange-500/10 to-transparent",
    border: "border-red-500/40",
    glow: "shadow-[0_0_40px_rgba(239,68,68,0.15),inset_0_1px_0_rgba(255,255,255,0.05)]",
    iconBg: "bg-gradient-to-br from-red-500/30 to-orange-600/20",
    iconColor: "text-red-400",
    accentColor: "text-red-400",
    statusBg: "bg-red-500/20 border border-red-500/30 text-red-300",
    outputBg: "bg-black/40",
    fontClass: "font-mono text-xs leading-relaxed tracking-wide",
    headerGlow: "drop-shadow-[0_0_8px_rgba(239,68,68,0.5)]",
  },
  finetuned: {
    gradient: "from-primary/20 via-secondary/10 to-transparent",
    border: "border-primary/40",
    glow: "shadow-[0_0_40px_rgba(212,175,55,0.2),inset_0_1px_0_rgba(255,255,255,0.05)]",
    iconBg: "bg-gradient-to-br from-primary/30 to-secondary/20",
    iconColor: "text-primary",
    accentColor: "text-primary",
    statusBg: "bg-primary/20 border border-primary/30 text-primary",
    outputBg: "bg-black/30",
    fontClass: "font-serif text-sm leading-relaxed italic",
    headerGlow: "drop-shadow-[0_0_8px_rgba(212,175,55,0.5)]",
  },
  hosted: {
    gradient: "from-cyan-500/20 via-blue-500/10 to-transparent",
    border: "border-cyan-500/40",
    glow: "shadow-[0_0_40px_rgba(6,182,212,0.2),inset_0_1px_0_rgba(255,255,255,0.05)]",
    iconBg: "bg-gradient-to-br from-cyan-500/30 to-blue-600/20",
    iconColor: "text-cyan-400",
    accentColor: "text-cyan-400",
    statusBg: "bg-cyan-500/20 border border-cyan-500/30 text-cyan-300",
    outputBg: "bg-black/30",
    fontClass: "font-sans text-sm leading-relaxed",
    headerGlow: "drop-shadow-[0_0_8px_rgba(6,182,212,0.5)]",
  },
};

const ModelOutputCard = ({
  title,
  subtitle,
  params,
  statusLabel,
  icon: Icon,
  output,
  isLoading,
  typingSpeed,
  variant,
  devMode,
  metrics,
}: ModelOutputCardProps) => {
  const config = variantConfig[variant];

  return (
    <motion.div
      initial={{ opacity: 0, y: 30, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ 
        delay: variant === "scratch" ? 0.1 : variant === "finetuned" ? 0.2 : 0.3,
        duration: 0.5,
        ease: "easeOut"
      }}
      className={cn(
        "relative rounded-2xl overflow-hidden",
        "bg-gradient-to-b from-canvas-panel to-canvas-deep",
        "border",
        config.border,
        config.glow
      )}
    >
      {/* Gradient overlay at top */}
      <div className={cn(
        "absolute inset-x-0 top-0 h-32 bg-gradient-to-b pointer-events-none",
        config.gradient
      )} />

      <div className="relative p-4 flex flex-col h-full">
        {/* Header - Compact */}
        <div className="flex items-center gap-3 mb-3">
          <div className={cn(
            "w-9 h-9 rounded-lg flex items-center justify-center",
            config.iconBg,
            "border border-white/10"
          )}>
            <Icon className={cn("w-4 h-4", config.iconColor)} />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h3 className={cn(
                "font-serif text-base font-bold",
                config.accentColor,
                config.headerGlow
              )}>
                {title}
              </h3>
              <span className={cn(
                "px-2 py-0.5 rounded-full text-[10px] font-semibold",
                config.statusBg
              )}>
                {statusLabel}
              </span>
            </div>
            <p className="text-[10px] text-muted-foreground font-mono tracking-wider uppercase">
              {subtitle} • {params}
            </p>
          </div>
        </div>

        {/* Terminal output area */}
        <div className={cn(
          "flex-1 rounded-xl overflow-hidden",
          config.outputBg,
          "border border-white/5"
        )}>
          {/* Terminal header */}
          <div className="flex items-center gap-2 px-3 py-1.5 bg-black/30 border-b border-white/5">
            <div className="flex gap-1">
              <div className="w-2 h-2 rounded-full bg-red-500/70" />
              <div className="w-2 h-2 rounded-full bg-yellow-500/70" />
              <div className="w-2 h-2 rounded-full bg-green-500/70" />
            </div>
            <Terminal className="w-2.5 h-2.5 text-muted-foreground ml-1" />
            <span className="text-[9px] font-mono text-muted-foreground">output.log</span>
          </div>
          
          {/* Output content */}
          <div className="p-3 flex-1 overflow-auto scrollbar-thin">
            {isLoading ? (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <motion.div
                    animate={{ opacity: [0.3, 1, 0.3] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                    className={cn("w-2 h-2 rounded-full", 
                      variant === "scratch" ? "bg-red-500" :
                      variant === "finetuned" ? "bg-primary" : "bg-cyan-500"
                    )}
                  />
                  <span className="text-xs font-mono text-muted-foreground">
                    Processing query...
                  </span>
                </div>
                <div className="space-y-2 mt-4">
                  <motion.div 
                    animate={{ opacity: [0.1, 0.3, 0.1] }}
                    transition={{ duration: 1, repeat: Infinity }}
                    className="h-3 bg-white/10 rounded w-full" 
                  />
                  <motion.div 
                    animate={{ opacity: [0.1, 0.3, 0.1] }}
                    transition={{ duration: 1, repeat: Infinity, delay: 0.2 }}
                    className="h-3 bg-white/10 rounded w-4/5" 
                  />
                  <motion.div 
                    animate={{ opacity: [0.1, 0.3, 0.1] }}
                    transition={{ duration: 1, repeat: Infinity, delay: 0.4 }}
                    className="h-3 bg-white/10 rounded w-3/5" 
                  />
                </div>
              </div>
            ) : output ? (
              <div className={cn("text-foreground/90", config.fontClass)}>
                {variant === "hosted" ? (
                  <div className="bg-cyan-500/5 rounded-lg p-3 border-l-2 border-cyan-500/50">
                    <TypewriterText text={output} speed={typingSpeed} />
                  </div>
                ) : variant === "scratch" ? (
                  <div className="text-red-300/90">
                    <span className="text-red-500/70 select-none">&gt; </span>
                    <TypewriterText text={output} speed={typingSpeed} />
                  </div>
                ) : (
                  <TypewriterText text={output} speed={typingSpeed} />
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                <span className="text-xs font-mono opacity-50">
                  // Awaiting input...
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Metrics bar */}
        {devMode && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 pt-4 border-t border-white/5"
          >
            <div className="grid grid-cols-3 gap-3">
              <div className="text-center">
                <Clock className={cn("w-4 h-4 mx-auto mb-1", config.accentColor)} />
                <div className="text-[10px] font-mono text-muted-foreground">Latency</div>
                <div className={cn("text-xs font-bold", config.accentColor)}>{metrics.inferenceTime}</div>
              </div>
              <div className="text-center">
                <Activity className={cn("w-4 h-4 mx-auto mb-1", config.accentColor)} />
                <div className="text-[10px] font-mono text-muted-foreground">Perplexity</div>
                <div className={cn("text-xs font-bold", config.accentColor)}>{metrics.perplexity}</div>
              </div>
              <div className="text-center">
                <Gauge className={cn("w-4 h-4 mx-auto mb-1", config.accentColor)} />
                <div className="text-[10px] font-mono text-muted-foreground">Speed</div>
                <div className={cn("text-xs font-bold", config.accentColor)}>{metrics.tokensPerSec} t/s</div>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
};

export default ModelOutputCard;
