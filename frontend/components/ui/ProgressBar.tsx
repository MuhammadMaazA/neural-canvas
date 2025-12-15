import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface ProgressBarProps {
  label: string;
  value: number;
  variant?: "ai" | "gold";
  showPercentage?: boolean;
  delay?: number;
}

const ProgressBar = ({ label, value, variant = "ai", showPercentage = true, delay = 0 }: ProgressBarProps) => {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className={cn(
          "text-sm font-medium",
          variant === "gold" ? "text-primary" : "text-foreground"
        )}>
          {label}
        </span>
        {showPercentage && (
          <span className="text-xs font-mono text-muted-foreground">
            {value}%
          </span>
        )}
      </div>
      
      <div className="progress-ai">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 1, delay, ease: "easeOut" }}
          className={cn(
            "progress-ai-fill",
            variant === "gold" && "bg-gradient-gold"
          )}
        />
      </div>
    </div>
  );
};

export default ProgressBar;
