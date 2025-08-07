#!/usr/bin/env python3
"""
🚀 PUBLIC DEPLOYMENT SCRIPT - Create a publicly accessible deployment
"""

import subprocess
import json
from pathlib import Path

def deploy_public():
    """Deploy with public access"""
    
    # Update vercel.json to ensure it's public
    vercel_config = {
        "version": 2,
        "builds": [
            {
                "src": "api/main.py",
                "use": "@vercel/python"
            }
        ],
        "routes": [
            {
                "src": "/api/(.*)",
                "dest": "api/main.py"
            },
            {
                "src": "/(.*)",
                "dest": "/index.html"
            }
        ],
        "functions": {
            "api/main.py": {
                "maxDuration": 10
            }
        }
    }
    
    with open('vercel.json', 'w') as f:
        json.dump(vercel_config, f, indent=2)
    
    # Deploy publicly
    try:
        result = subprocess.run([
            "vercel", "--prod", "--public"
        ], capture_output=True, text=True)
        
        print("Deployment output:", result.stdout)
        if result.stderr:
            print("Deployment errors:", result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"Deployment failed: {e}")
        return False

if __name__ == "__main__":
    success = deploy_public()
    print(f"Deployment {'successful' if success else 'failed'}")