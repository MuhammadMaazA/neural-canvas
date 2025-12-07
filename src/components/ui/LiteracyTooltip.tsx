"use client";

import { useState } from "react";
import { Info } from "lucide-react";
import { cn } from "@/lib/utils";

interface LiteracyTooltipProps {
  term: string;
  definition: string;
  children?: React.ReactNode;
}

const LiteracyTooltip = ({ term, definition, children }: LiteracyTooltipProps) => {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <span
      className="relative inline-block"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children || (
        <span className="literacy-term text-secondary">
          {term}
        </span>
      )}
      
      {/* Tooltip */}
      <span
        className={cn(
          "absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 min-w-[200px] max-w-[280px]",
          "glass-panel text-xs text-foreground leading-relaxed",
          "transition-all duration-200 pointer-events-none",
          isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-2"
        )}
      >
        <span className="font-medium text-primary block mb-1">{term}</span>
        {definition}
        
        {/* Arrow */}
        <span className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-card border-r border-b border-white/10 rotate-45" />
      </span>
    </span>
  );
};

// Info icon variant
export const InfoTooltip = ({ content }: { content: string }) => {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <span
      className="relative inline-flex items-center"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      <Info className="w-3.5 h-3.5 text-muted-foreground cursor-help ml-1" />
      
      <span
        className={cn(
          "absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 min-w-[180px] max-w-[250px]",
          "glass-panel text-xs text-muted-foreground leading-relaxed",
          "transition-all duration-200 pointer-events-none",
          isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-2"
        )}
      >
        {content}
        <span className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-card border-r border-b border-white/10 rotate-45" />
      </span>
    </span>
  );
};

export default LiteracyTooltip;
