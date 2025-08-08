#!/usr/bin/env python3
"""
Simple deployment script to fix Vercel issues
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path

def deploy_simple():
    """Deploy with a simple temporary directory approach"""
    
    # Create a temporary deployment directory
    with tempfile.TemporaryDirectory(prefix="bahai_quest_") as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy essential files
        current_dir = Path(__file__).parent
        
        # Copy files
        shutil.copy2(current_dir / "index.html", temp_path / "index.html")
        shutil.copy2(current_dir / "vercel.json", temp_path / "vercel.json") 
        shutil.copy2(current_dir / "requirements.txt", temp_path / "requirements.txt")
        
        # Copy API directory
        api_dest = temp_path / "api"
        api_dest.mkdir()
        shutil.copy2(current_dir / "api" / "main.py", api_dest / "main.py")
        
        print(f"📁 Created temporary deployment at: {temp_path}")
        
        # Change to temp directory and deploy
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_path)
            
            # Deploy with explicit project settings
            result = subprocess.run([
                "vercel", "--prod", "--yes"
            ], capture_output=True, text=True)
            
            print("=== VERCEL OUTPUT ===")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            print("Return code:", result.returncode)
            
            if result.returncode == 0:
                return result.stdout
            else:
                return None
                
        finally:
            os.chdir(original_cwd)

if __name__ == "__main__":
    result = deploy_simple()
    if result:
        print("✅ Deployment successful!")
        print(result)
    else:
        print("❌ Deployment failed")