import { motion } from "framer-motion";
import { Brain, Github, ExternalLink } from "lucide-react";

const Footer = () => {
  return (
    <footer className="py-8 px-6 border-t border-border/50">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="flex flex-col md:flex-row items-center justify-between gap-4"
        >
          {/* Branding */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-ai flex items-center justify-center">
              <Brain className="w-4 h-4 text-primary-foreground" />
            </div>
            <div>
              <span className="font-serif font-semibold text-sm">Neural Canvas</span>
              <span className="text-muted-foreground text-xs ml-2">v1.0</span>
            </div>
          </div>

          {/* Institution */}
          <div className="text-center">
            <p className="text-sm text-muted-foreground">
              University College London (UCL) — <span className="font-mono text-xs">COMP0220</span>
            </p>
          </div>

          {/* Links & Disclaimer */}
          <div className="flex items-center gap-4">
            <a
              href="#"
              className="text-muted-foreground hover:text-foreground transition-colors"
              title="View on GitHub"
            >
              <Github className="w-5 h-5" />
            </a>
            <a
              href="#"
              className="text-muted-foreground hover:text-foreground transition-colors"
              title="Documentation"
            >
              <ExternalLink className="w-5 h-5" />
            </a>
          </div>
        </motion.div>

        {/* Disclaimer */}
        <div className="mt-6 pt-4 border-t border-border/30 text-center">
          <p className="text-xs text-muted-foreground/70">
            ⚠️ Disclaimer: All outputs are probabilistic. Model predictions reflect training data biases and should not be considered authoritative art historical analysis.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
