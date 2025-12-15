import { motion } from "framer-motion";
import { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface Prediction {
  label: string;
  confidence: number;
}

interface CNNOutputCardProps {
  title: string;
  subtitle: string;
  params: string;
  statusLabel?: string;
  icon: LucideIcon;
  predictions: Prediction[] | null;
  isLoading: boolean;
  variant: "scratch" | "finetuned";
  devMode: boolean;
  metrics: {
    inferenceTime: string;
    accuracy: string;
    layers: string;
  };
}

const variantStyles = {
  scratch: {
    border: "border-orange-500/30",
    glow: "shadow-[0_0_30px_rgba(249,115,22,0.15)]",
    accent: "text-orange-400",
    accentBg: "bg-orange-500/20",
    statusBg: "bg-orange-500/20",
    statusText: "text-orange-400",
    statusBorder: "border-orange-500/30",
    barBg: "bg-orange-500",
  },
  finetuned: {
    border: "border-primary/30",
    glow: "shadow-[0_0_30px_rgba(212,175,55,0.15)]",
    accent: "text-primary",
    accentBg: "bg-primary/20",
    statusBg: "bg-primary/20",
    statusText: "text-primary",
    statusBorder: "border-primary/30",
    barBg: "bg-gradient-to-r from-primary to-secondary",
  },
};

const CNNOutputCard = ({
  title,
  subtitle,
  params,
  statusLabel,
  icon: Icon,
  predictions,
  isLoading,
  variant,
  devMode,
  metrics,
}: CNNOutputCardProps) => {
  const styles = variantStyles[variant];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: variant === "scratch" ? 0.2 : 0.3 }}
      className={cn(
        "glass-panel rounded-xl p-4 flex flex-col border",
        styles.border,
        styles.glow
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={cn("w-8 h-8 rounded-lg flex items-center justify-center", styles.accentBg)}>
            <Icon className={cn("w-4 h-4", styles.accent)} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className={cn("font-serif text-sm font-semibold", styles.accent)}>
                {title}
              </h3>
              {statusLabel && (
                <span className={cn(
                  "px-2 py-0.5 rounded-full text-[9px] font-mono border",
                  styles.statusBg, styles.statusText, styles.statusBorder
                )}>
                  {statusLabel}
                </span>
              )}
            </div>
            <p className="text-[10px] text-muted-foreground font-mono uppercase tracking-wider">
              {subtitle} • {params}
            </p>
          </div>
        </div>
      </div>

      {/* Predictions Area */}
      <div className="flex-1 rounded-lg bg-canvas-deep border border-border/50 p-3 overflow-hidden">
        {isLoading ? (
          <div className="h-full flex flex-col justify-center gap-2">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="flex items-center gap-3">
                <div className="w-24 h-3 bg-muted/50 rounded animate-pulse" />
                <div className="flex-1 h-2 bg-muted/30 rounded animate-pulse" />
                <div className="w-10 h-3 bg-muted/50 rounded animate-pulse" />
              </div>
            ))}
          </div>
        ) : predictions ? (
          <div className="space-y-2">
            {predictions.map((pred, index) => (
              <motion.div
                key={pred.label}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center gap-3"
              >
                <span className="w-28 text-[11px] text-muted-foreground truncate">
                  {pred.label}
                </span>
                <div className="flex-1 h-2 bg-muted/30 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${pred.confidence}%` }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    className={cn("h-full rounded-full", styles.barBg)}
                  />
                </div>
                <span className={cn("w-12 text-right text-[11px] font-mono", styles.accent)}>
                  {pred.confidence.toFixed(1)}%
                </span>
              </motion.div>
            ))}
          </div>
        ) : (
          <div className="h-full flex items-center justify-center">
            <span className="text-xs text-muted-foreground/50 font-mono">
              // Awaiting image...
            </span>
          </div>
        )}
      </div>

      {/* Dev Mode Metrics */}
      {devMode && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          className="mt-3 pt-3 border-t border-border/50"
        >
          <div className="grid grid-cols-3 gap-2">
            <div className="text-center">
              <p className="text-[9px] text-muted-foreground uppercase">Inference</p>
              <p className={cn("text-xs font-mono", styles.accent)}>{metrics.inferenceTime}</p>
            </div>
            <div className="text-center">
              <p className="text-[9px] text-muted-foreground uppercase">Accuracy</p>
              <p className={cn("text-xs font-mono", styles.accent)}>{metrics.accuracy}</p>
            </div>
            <div className="text-center">
              <p className="text-[9px] text-muted-foreground uppercase">Layers</p>
              <p className={cn("text-xs font-mono", styles.accent)}>{metrics.layers}</p>
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default CNNOutputCard;
