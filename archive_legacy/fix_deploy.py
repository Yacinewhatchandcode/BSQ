#!/usr/bin/env python3
"""
Fixed deployment with corrected routing
"""

import os
import subprocess
import tempfile
import shutil
import json
from pathlib import Path

def create_fixed_vercel_config():
    """Create a fixed vercel.json with proper routing"""
    return {
        "version": 2,
        "functions": {
            "api/main.py": {
                "runtime": "python3.9"
            }
        },
        "routes": [
            {
                "src": "/api/(.*)",
                "dest": "/api/main.py"
            },
            {
                "src": "/(.*)",
                "dest": "/index.html"
            }
        ]
    }

def deploy_fixed():
    """Deploy with fixed configuration"""
    
    # Create a temporary deployment directory
    with tempfile.TemporaryDirectory(prefix="bahai_fixed_") as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy essential files
        current_dir = Path(__file__).parent
        
        # Copy files
        shutil.copy2(current_dir / "index.html", temp_path / "index.html")
        shutil.copy2(current_dir / "requirements.txt", temp_path / "requirements.txt")
        
        # Create fixed vercel.json
        vercel_config = create_fixed_vercel_config()
        with open(temp_path / "vercel.json", 'w') as f:
            json.dump(vercel_config, f, indent=2)
        
        # Copy API directory
        api_dest = temp_path / "api"
        api_dest.mkdir()
        shutil.copy2(current_dir / "api" / "main.py", api_dest / "main.py")
        
        print(f"📁 Created fixed deployment at: {temp_path}")
        
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
    result = deploy_fixed()
    if result:
        print("✅ Fixed deployment successful!")
        print(result)
    else:
        print("❌ Deployment failed")