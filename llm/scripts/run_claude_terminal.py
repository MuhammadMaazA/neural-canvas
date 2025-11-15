"""
Interactive terminal client for Anthropic Claude (REPL)

Usage:
  1) Ensure you have an API key in $ANTHROPIC_API_KEY (see csh instructions below)
  2) Activate your venv and install dependency: pip install anthropic
  3) Run: python llm/scripts/run_claude_terminal.py

Notes:
- This script uses the official `anthropic` Python package.
- It builds a short formatted prompt using Anthropic's HUMAN_PROMPT/AI_PROMPT markers.
- Network access and a valid API key are required.
"""

import os
import sys
import time

try:
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
except Exception as e:
    print("Missing dependency: the 'anthropic' package is required. Install with: pip install anthropic")
    raise

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    print("Error: ANTHROPIC_API_KEY environment variable not set. See README or set it in your shell.")
    sys.exit(1)

# Create client
client = Anthropic(api_key=API_KEY)
MODEL = os.environ.get("CLAUDE_MODEL", "claude-2")

print("Interactive Claude terminal (type 'exit' or 'quit' to stop)")
print(f"Using model: {MODEL}")
print("---")

try:
    while True:
        try:
            user = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user:
            continue
        if user.strip().lower() in ("exit", "quit"):
            break

        # Build Anthropic-style prompt markers
        prompt = HUMAN_PROMPT + user + "\n" + AI_PROMPT

        # Call the API
        try:
            # NOTE: parameters like max_tokens_to_sample and temperature may differ by SDK version
            resp = client.completions.create(
                model=MODEL,
                prompt=prompt,
                max_tokens_to_sample=512,
                temperature=0.7,
                stop_sequences=[HUMAN_PROMPT]
            )

            # resp may be a dict-like with 'completion'
            completion = resp.get("completion") if isinstance(resp, dict) else getattr(resp, "completion", None)
            if completion is None:
                # Fallback: print raw resp
                print("Claude:")
                print(resp)
            else:
                print("Claude:")
                print(completion.strip())

        except Exception as e:
            print("API request failed:", str(e))
            time.sleep(1)

except KeyboardInterrupt:
    print("\nInterrupted. Goodbye.")

