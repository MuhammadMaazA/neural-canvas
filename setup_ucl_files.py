"""
Run this on UCL machine to create all necessary files
Usage: python3 setup_ucl_files.py
"""

import os
import urllib.request

def create_file(path, url_or_content, is_url=False):
    """Create a file from URL or content"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    if is_url:
        print(f"Downloading {path}...")
        urllib.request.urlretrieve(url_or_content, path)
    else:
        print(f"Creating {path}...")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(url_or_content)
    print(f"✓ Created: {path}")

# Actually, files are too large. Better to use Git.
print("=" * 60)
print("SETUP FILES FOR UCL")
print("=" * 60)
print("\nOption 1: Use Git (RECOMMENDED)")
print("  1. On your local machine: git add . && git commit -m 'add files' && git push")
print("  2. On UCL machine: git pull")
print("\nOption 2: Create files manually")
print("  Files are too large to embed. Use scp from LOCAL machine:")
print("  scp -J muhamaaz@knuckles.cs.ucl.ac.uk COMPLETE_LLM_TRAINING.py muhamaaz@canada-l.cs.ucl.ac.uk:/cs/student/projects1/2023/muhamaaz/DeepLearningCoursework/")
print("\nOption 3: Use this Python script to create minimal working version")
print("=" * 60)

# Create directory structure
os.makedirs('llm', exist_ok=True)

# Create minimal __init__.py
with open('llm/__init__.py', 'w') as f:
    f.write('# LLM Package\n')
print("✓ Created llm/__init__.py")

print("\nNext: Upload COMPLETE_LLM_TRAINING.py and llm/*.py files using:")
print("  - Git (best)")
print("  - scp from LOCAL Windows machine (not from UCL machine)")

