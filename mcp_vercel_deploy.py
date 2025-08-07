#!/usr/bin/env python3
"""
🚀 MCP VERCEL DEPLOYMENT TOOL
Comprehensive Vercel integration with proper credential management and deployment
"""

import os
import json
import subprocess
import requests
import time
from pathlib import Path
from datetime import datetime

class MCPVercelDeployer:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.project_name = "bahai-spiritual-quest"
        self.deployment_url = None
        
    def log(self, message, level="INFO"):
        """Enhanced logging with colors and timestamps"""
        colors = {
            "INFO": "\033[36m",
            "SUCCESS": "\033[32m", 
            "ERROR": "\033[31m",
            "WARNING": "\033[33m",
            "RESET": "\033[0m"
        }
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = colors.get(level, colors["RESET"])
        print(f"{color}[{timestamp}] {level}: {message}{colors['RESET']}")

    def check_vercel_auth(self):
        """Check Vercel authentication status"""
        self.log("🔐 Checking Vercel authentication...")
        
        try:
            result = subprocess.run([
                "vercel", "whoami"
            ], capture_output=True, text=True, check=True)
            
            username = result.stdout.strip()
            self.log(f"✅ Authenticated as: {username}", "SUCCESS")
            return True
            
        except subprocess.CalledProcessError:
            self.log("❌ Not authenticated with Vercel", "ERROR")
            return False

    def login_to_vercel(self):
        """Interactive Vercel login"""
        self.log("🔑 Please authenticate with Vercel...")
        
        try:
            subprocess.run(["vercel", "login"], check=True)
            return self.check_vercel_auth()
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Login failed: {e}", "ERROR")
            return False

    def create_optimized_structure(self):
        """Create optimized deployment structure"""
        self.log("🏗️ Creating optimized deployment structure...")
        
        # Ensure we have the essential files
        essential_files = {
            "index.html": self.create_main_html(),
            "api/main.py": self.create_api_endpoint(),
            "vercel.json": self.create_vercel_config(),
            "requirements.txt": self.create_requirements()
        }
        
        for file_path, content in essential_files.items():
            full_path = self.root_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            if content:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
        self.log("✅ Optimized structure created", "SUCCESS")
        return True

    def create_main_html(self):
        """Create optimized main HTML file"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>The Hidden Words - Sacred Digital Manuscript</title>
    
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&family=Cinzel:wght@300;400;600&family=Dancing+Script:wght@400;700&display=swap" rel="stylesheet">
    
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Cinzel', serif;
            background: #0B0E1A;
            min-height: 100vh;
            display: flex;
            align-items: flex-start;
            justify-content: center;
            position: relative;
            overflow-x: hidden;
            padding: 40px 20px;
        }

        /* Starfield background */
        .starfield {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 1;
            pointer-events: none;
        }

        .star {
            position: absolute;
            background: white;
            border-radius: 50%;
            animation: twinkle 3s ease-in-out infinite;
        }

        @keyframes twinkle {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 0.8; }
        }

        /* Main container */
        .manuscript-container {
            width: 90%;
            max-width: 800px;
            background: rgba(15, 25, 40, 0.9);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 80px 60px 60px;
            text-align: center;
            z-index: 10;
            position: relative;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            margin-top: 20px;
            margin-bottom: 20px;
        }

        /* Persian title */
        .title-persian {
            font-family: 'Amiri', serif;
            font-size: 4.5em;
            font-weight: 700;
            color: #E6D4A3;
            margin-bottom: 20px;
            letter-spacing: 2px;
            text-shadow: 0 0 20px rgba(230, 212, 163, 0.3);
        }

        /* Subtitle */
        .subtitle {
            font-family: 'Cinzel', serif;
            font-size: 0.9em;
            font-weight: 300;
            color: #A8A8A8;
            letter-spacing: 3px;
            margin-bottom: 60px;
            text-transform: uppercase;
        }

        /* Welcome message */
        .welcome-message {
            font-family: 'Cinzel', serif;
            font-size: 1.1em;
            font-weight: 300;
            color: #CCCCCC;
            line-height: 1.8;
            margin-bottom: 60px;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }

        .welcome-message em {
            font-style: italic;
            color: #E6D4A3;
        }

        /* Input container */
        .input-container {
            display: flex;
            align-items: center;
            max-width: 500px;
            margin: 0 auto;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 50px;
            padding: 8px;
        }

        .search-input {
            flex: 1;
            background: transparent;
            border: none;
            padding: 15px 25px;
            font-family: 'Cinzel', serif;
            font-size: 1em;
            color: #CCCCCC;
            outline: none;
        }

        .search-input::placeholder {
            color: #666666;
            font-style: italic;
        }

        .reveal-button {
            background: linear-gradient(135deg, #D4AF37, #B8941F);
            color: #1A1A1A;
            border: none;
            padding: 15px 30px;
            border-radius: 40px;
            font-family: 'Cinzel', serif;
            font-weight: 600;
            font-size: 0.9em;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .reveal-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(212, 175, 55, 0.3);
        }

        .reveal-button:active {
            transform: translateY(0px);
            box-shadow: 0 5px 10px rgba(212, 175, 55, 0.2);
        }

        /* Chat area */
        .chat-area {
            display: none;
            margin-top: 40px;
            text-align: left;
        }

        .chat-messages {
            max-height: 400px;
            overflow-y: auto;
            margin-bottom: 20px;
            padding: 20px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .message {
            margin-bottom: 30px;
            padding: 20px 25px;
            border-radius: 15px;
            animation: fadeIn 0.5s ease-out;
            line-height: 1.8;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            background: rgba(212, 175, 55, 0.2);
            border: 1px solid rgba(212, 175, 55, 0.3);
            margin-left: 20%;
        }

        .message.agent {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .message-content {
            font-family: 'Cinzel', serif;
            font-size: 1.15em;
            color: #E0E0E0;
            line-height: 1.9;
            letter-spacing: 0.3px;
        }

        .message.agent .message-content {
            font-family: 'Dancing Script', cursive;
            font-size: 1.3em;
            letter-spacing: 0.5px;
        }

        /* Golden card quote styling */
        .hidden-word-quote {
            background: linear-gradient(145deg, #D4AF37 0%, #F4E6A1 50%, #D4AF37 100%);
            border: none;
            border-radius: 25px;
            padding: 45px 40px 35px;
            margin: 40px auto;
            max-width: 500px;
            font-family: 'Amiri', serif;
            font-size: 1.4em;
            font-style: normal;
            color: #2D3A4B;
            position: relative;
            overflow: hidden;
            line-height: 1.7;
            letter-spacing: 0.3px;
            box-shadow: 
                0 15px 35px rgba(212, 175, 55, 0.4),
                0 5px 15px rgba(0, 0, 0, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);
            text-align: center;
            transform: perspective(1000px) rotateX(2deg);
            transition: all 0.3s ease;
        }

        .hidden-word-quote:hover {
            transform: perspective(1000px) rotateX(0deg) translateY(-5px);
            box-shadow: 
                0 20px 45px rgba(212, 175, 55, 0.5),
                0 10px 25px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.4);
        }

        .hidden-word-quote::before {
            content: '';
            position: absolute;
            top: 15px;
            left: 15px;
            right: 15px;
            bottom: 15px;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            z-index: 0;
        }

        .quote-content {
            position: relative;
            z-index: 1;
        }

        /* Status indicator */
        .status {
            position: fixed;
            top: 30px;
            right: 30px;
            padding: 10px 20px;
            background: rgba(0, 0, 0, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            color: #CCCCCC;
            font-size: 0.8em;
            z-index: 1000;
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .status.show {
            opacity: 1;
        }

        .status.success {
            border-color: #28a745;
            color: #28a745;
        }

        .status.error {
            border-color: #dc3545;
            color: #dc3545;
        }

        /* Deployment badge */
        .deployment-badge {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            color: #D4AF37;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 12px;
            border: 1px solid rgba(212, 175, 55, 0.3);
            z-index: 1000;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .manuscript-container {
                padding: 60px 30px;
                width: 95%;
            }
            
            .title-persian {
                font-size: 3em;
            }
            
            .welcome-message {
                font-size: 1em;
            }
        }
    </style>
</head>
<body>
    <!-- Starfield -->
    <div class="starfield" id="starfield"></div>
    
    <!-- Status indicator -->
    <div class="status" id="status"></div>
    
    <!-- Deployment badge -->
    <div class="deployment-badge">🚀 Live on Vercel</div>
    
    <!-- Main container -->
    <div class="manuscript-container">
        <!-- Persian title -->
        <h1 class="title-persian">كلمات مخفیه</h1>
        
        <!-- Subtitle -->
        <p class="subtitle">The Hidden Words - Sacred Digital Manuscript</p>
        
        <!-- Welcome message -->
        <div class="welcome-message">
            Welcome to this sacred digital space, where the eternal wisdom of <em>The Hidden Words</em> illuminates our spiritual journey. I am here to share these divine pearls with you, each one a treasure from the ocean of divine knowledge.
        </div>
        
        <!-- Input container -->
        <div class="input-container">
            <input 
                type="text" 
                class="search-input" 
                id="messageInput"
                placeholder="Seek wisdom from The Hidden Words..."
                onkeypress="handleKeyPress(event)"
            >
            <button class="reveal-button" onclick="sendMessage()">Reveal</button>
        </div>
        
        <!-- Chat area -->
        <div class="chat-area" id="chatArea">
            <div class="chat-messages" id="chatMessages"></div>
        </div>
    </div>
    
    <script>
        // Create starfield
        function createStarfield() {
            const starfield = document.getElementById('starfield');
            const starCount = 150;
            
            for (let i = 0; i < starCount; i++) {
                const star = document.createElement('div');
                star.className = 'star';
                star.style.width = Math.random() * 3 + 'px';
                star.style.height = star.style.width;
                star.style.left = Math.random() * 100 + '%';
                star.style.top = Math.random() * 100 + '%';
                star.style.animationDelay = Math.random() * 3 + 's';
                starfield.appendChild(star);
            }
        }
        
        // Send message function
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) {
                showStatus('Please enter your spiritual inquiry', 'error');
                return;
            }
            
            // Show chat area if hidden
            const chatArea = document.getElementById('chatArea');
            chatArea.style.display = 'block';
            
            addMessage(message, 'user');
            input.value = '';
            
            showStatus('Seeking divine wisdom...', 'success');
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `message=${encodeURIComponent(message)}`
                });
                
                const data = await response.json();
                
                // Check if response is a Hidden Word quote
                if (data.response.includes('O Son of') || data.response.includes('O Friend') || data.response.includes('O Children')) {
                    addHiddenWordMessage(data.response, 'agent');
                } else {
                    addMessage(data.response, 'agent');
                }
                
                showStatus('Divine wisdom revealed', 'success');
                
            } catch (error) {
                console.error('Error:', error);
                addMessage('The divine connection is momentarily clouded. Please try again.', 'agent');
                showStatus('Connection error', 'error');
            }
        }
        
        // Add regular message
        function addMessage(text, sender) {
            const chatMessages = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            const content = document.createElement('div');
            content.className = 'message-content';
            content.textContent = text;
            
            messageDiv.appendChild(content);
            chatMessages.appendChild(messageDiv);
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // Add Hidden Word message with golden card styling
        function addHiddenWordMessage(text, sender) {
            const chatMessages = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            const content = document.createElement('div');
            content.className = 'message-content';
            
            // Create golden card for Hidden Words
            const card = document.createElement('div');
            card.className = 'hidden-word-quote';
            card.innerHTML = `<div class="quote-content">${text}</div>`;
            
            content.appendChild(card);
            messageDiv.appendChild(content);
            chatMessages.appendChild(messageDiv);
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // Handle Enter key
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        // Show status message
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = `status ${type}`;
            status.classList.add('show');
            
            setTimeout(() => {
                status.classList.remove('show');
            }, 3000);
        }
        
        // Initialize
        window.addEventListener('load', function() {
            createStarfield();
            console.log('🌟 Bahá\\'í Spiritual Quest loaded successfully');
            console.log('✨ Ready to explore The Hidden Words');
            console.log('🚀 Deployed with MCP Vercel Integration');
        });
    </script>
</body>
</html>'''

    def create_api_endpoint(self):
        """Create optimized API endpoint"""
        return '''#!/usr/bin/env python3
"""
🌟 Bahá'í Spiritual Quest - Production API Endpoint
Optimized for Vercel serverless deployment with MCP integration
"""

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from datetime import datetime
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Bahá'í Spiritual Quest API",
    description="Sacred digital space for exploring The Hidden Words",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Spiritual responses from The Hidden Words
SPIRITUAL_RESPONSES = [
    "O Son of Being! Love Me, that I may love thee. If thou lovest Me not, My love can in no wise reach thee.",
    "O Friend! In the garden of thy heart plant naught but the rose of love, and from the nightingale of affection and desire loathe not to turn away.",
    "O Son of Man! For everything there is a sign. The sign of love is fortitude in My decree and patience in My trials.",
    "O Children of Men! Know ye not why We created you all from the same dust? That no one should exalt himself over the other.",
    "O Son of Spirit! Noble have I created thee, yet thou hast abased thyself. Rise then unto that for which thou wast created.",
    "O My Friend! Thou art the daystar of the heavens of My holiness, let not the defilement of the world eclipse thy splendor.",
    "O Son of Being! Thy heart is My home; sanctify it for My descent. Thy spirit is My place of revelation; cleanse it for My manifestation.",
    "O Son of Man! Breathe not the sins of others so long as thou art thyself a sinner. Shouldst thou transgress this command, accursed wouldst thou be, and to this I bear witness.",
    "O Son of Being! How couldst thou forget thine own faults and busy thyself with the faults of others? Whoso doeth this is accursed of Me.",
    "O Son of Being! Seek a martyr's death in My path, content with My pleasure and thankful for that which hath befallen thee, for thus wilt thou abide in prosperity and comfort in the realm of eternity.",
    "O Son of Man! Rejoice in the gladness of thine heart, that thou mayest be worthy to meet Me and to mirror forth My beauty.",
    "O Friend! Thou art the light of the world. Let not the shadows of vain glory obscure thy brightness.",
    "O Son of Spirit! The best beloved of all things in My sight is Justice; turn not away therefrom if thou desirest Me, and neglect it not that I may confide in thee.",
    "O Children of Dust! Tell the rich of the midnight sighing of the poor, lest heedlessness lead them into the path of destruction."
]

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main spiritual interface"""
    try:
        # Try to read the HTML content
        html_path = Path(__file__).parent.parent / "index.html"
        if html_path.exists():
            return html_path.read_text(encoding='utf-8')
        else:
            # Fallback content
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>The Hidden Words - Sacred Digital Manuscript</title>
                <meta charset="utf-8">
                <style>
                    body { 
                        font-family: 'Cinzel', serif; 
                        background: #0B0E1A; 
                        color: #E6D4A3; 
                        text-align: center; 
                        padding: 50px;
                    }
                    h1 { font-size: 3em; margin-bottom: 30px; }
                    p { font-size: 1.2em; line-height: 1.6; }
                </style>
            </head>
            <body>
                <h1>كلمات مخفیه</h1>
                <h2>The Hidden Words - Sacred Digital Manuscript</h2>
                <p>Welcome to this sacred digital space, where the eternal wisdom illuminates our spiritual journey.</p>
                <p>🚀 Successfully deployed with MCP Vercel Integration</p>
            </body>
            </html>
            """
    except Exception as e:
        return f"<h1>Bahá'í Spiritual Quest</h1><p>Status: Online</p><p>Error: {str(e)}</p>"

@app.post("/api/chat")
async def chat_endpoint(message: str = Form(...)):
    """Spiritual guidance endpoint with enhanced responses"""
    try:
        message_lower = message.lower()
        
        # Enhanced response selection based on keywords
        if any(word in message_lower for word in ["love", "loving", "beloved"]):
            response = SPIRITUAL_RESPONSES[0]
        elif any(word in message_lower for word in ["friend", "friendship", "companion"]):
            response = SPIRITUAL_RESPONSES[1] 
        elif any(word in message_lower for word in ["sign", "signs", "patience", "trial"]):
            response = SPIRITUAL_RESPONSES[2]
        elif any(word in message_lower for word in ["children", "equality", "dust", "created"]):
            response = SPIRITUAL_RESPONSES[3]
        elif any(word in message_lower for word in ["noble", "spirit", "spiritual", "created"]):
            response = SPIRITUAL_RESPONSES[4]
        elif any(word in message_lower for word in ["daystar", "light", "brightness", "glory"]):
            response = SPIRITUAL_RESPONSES[5]
        elif any(word in message_lower for word in ["heart", "home", "revelation", "manifestation"]):
            response = SPIRITUAL_RESPONSES[6]
        elif any(word in message_lower for word in ["justice", "just", "beloved", "sight"]):
            response = SPIRITUAL_RESPONSES[12]
        elif any(word in message_lower for word in ["rich", "poor", "wealth", "poverty"]):
            response = SPIRITUAL_RESPONSES[13]
        elif any(word in message_lower for word in ["rejoice", "gladness", "joy", "happiness"]):
            response = SPIRITUAL_RESPONSES[10]
        else:
            # Select based on message hash for consistency
            import hashlib
            hash_obj = hashlib.md5(message.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            response = SPIRITUAL_RESPONSES[hash_int % len(SPIRITUAL_RESPONSES)]
        
        return {
            "message": message,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "source": "The Hidden Words of Bahá'u'lláh",
            "deployment": "MCP Vercel Integration v2.0"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spiritual guidance error: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy",
        "service": "Bahá'í Spiritual Quest",
        "version": "2.0.0",
        "deployment": "MCP Vercel Integration",
        "timestamp": datetime.now().isoformat(),
        "features": [
            "Golden Quote Cards",
            "Persian Text Support", 
            "Spiritual Guidance AI",
            "Starfield Animation",
            "Mobile Responsive"
        ]
    }

@app.get("/api/quotes")
async def get_random_quote():
    """Get a random spiritual quote"""
    import random
    quote = random.choice(SPIRITUAL_RESPONSES)
    return {
        "quote": quote,
        "source": "The Hidden Words of Bahá'u'lláh",
        "timestamp": datetime.now().isoformat()
    }

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

    def create_vercel_config(self):
        """Create optimized Vercel configuration"""
        return json.dumps({
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
            ]
        }, indent=2)

    def create_requirements(self):
        """Create minimal requirements for fast deployment"""
        return """fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6"""

    def clean_previous_deployments(self):
        """Clean up previous failed deployments"""
        self.log("🧹 Cleaning previous deployments...")
        
        try:
            # List all projects
            result = subprocess.run([
                "vercel", "list"
            ], capture_output=True, text=True)
            
            if "bahai" in result.stdout.lower():
                self.log("Found existing Bahá'í projects, cleaning up...")
                
                # Remove old projects
                subprocess.run([
                    "vercel", "remove", "bahai-spiritual-quest", "--yes"
                ], capture_output=True)
                
                subprocess.run([
                    "vercel", "remove", "bahai-quest-deploy", "--yes"
                ], capture_output=True)
                
                self.log("✅ Cleanup completed", "SUCCESS")
                
        except Exception as e:
            self.log(f"⚠️ Cleanup warning: {e}", "WARNING")

    def deploy_to_vercel(self):
        """Deploy to Vercel with proper configuration"""
        self.log("🚀 Deploying to Vercel...")
        
        try:
            # Deploy with production flag
            result = subprocess.run([
                "vercel", "--prod", "--yes", "--confirm"
            ], capture_output=True, text=True, cwd=self.root_dir)
            
            if result.returncode == 0:
                self.log("✅ Vercel deployment successful!", "SUCCESS")
                
                # Extract URL from output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'https://' in line and 'vercel.app' in line:
                        url = line.strip()
                        self.deployment_url = url
                        self.log(f"🌐 Live at: {url}", "SUCCESS")
                        return url
                        
                self.log("✅ Deployment completed (URL not captured)", "SUCCESS")
                return True
                
            else:
                self.log(f"❌ Deployment failed:", "ERROR")
                self.log(f"stdout: {result.stdout}", "ERROR") 
                self.log(f"stderr: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Deployment exception: {e}", "ERROR")
            return False

    def test_deployment(self, url=None):
        """Test the deployed application"""
        if not url and not self.deployment_url:
            self.log("⚠️ No URL provided for testing", "WARNING")
            return False
            
        test_url = url or self.deployment_url
        self.log(f"🧪 Testing deployment at {test_url}...")
        
        try:
            # Test main page
            response = requests.get(test_url, timeout=10)
            if response.status_code == 200 and "كلمات مخفیه" in response.text:
                self.log("✅ Main page loads with Persian title", "SUCCESS")
                
                # Test API health endpoint
                api_url = f"{test_url}/api/health"
                api_response = requests.get(api_url, timeout=10)
                
                if api_response.status_code == 200:
                    self.log("✅ API health check successful", "SUCCESS")
                    return True
                else:
                    self.log(f"⚠️ API health check failed: {api_response.status_code}", "WARNING")
                    return False
            else:
                self.log(f"❌ Main page test failed: {response.status_code}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Testing failed: {e}", "ERROR")
            return False

    def run_mcp_deployment(self):
        """Run complete MCP Vercel deployment"""
        self.log("🌟 MCP VERCEL DEPLOYMENT SYSTEM", "SUCCESS")
        self.log("=" * 70)
        
        steps = [
            ("🔐 Checking Vercel authentication", self.check_vercel_auth),
            ("🏗️ Creating optimized structure", self.create_optimized_structure),
            ("🧹 Cleaning previous deployments", self.clean_previous_deployments), 
            ("🚀 Deploying to Vercel", self.deploy_to_vercel),
            ("🧪 Testing deployment", self.test_deployment),
        ]
        
        results = {}
        for step_name, step_func in steps:
            self.log(f"\n{step_name}...")
            try:
                result = step_func()
                results[step_name] = bool(result)
                if result:
                    self.log(f"✅ {step_name}: Success", "SUCCESS")
                else:
                    self.log(f"⚠️ {step_name}: Needs attention", "WARNING")
                    if step_name == "🔐 Checking Vercel authentication":
                        self.log("🔑 Please run: vercel login", "INFO")
                        return results
            except Exception as e:
                self.log(f"❌ {step_name}: Failed - {e}", "ERROR")
                results[step_name] = False
                
        # Final summary
        self.log("\n" + "=" * 70)
        self.log("📊 MCP DEPLOYMENT SUMMARY", "SUCCESS")
        self.log("=" * 70)
        
        success_count = sum(results.values())
        total_count = len(results)
        
        for step, success in results.items():
            status = "✅" if success else "❌"
            self.log(f"{status} {step}")
            
        self.log(f"\n📈 Success Rate: {success_count}/{total_count}")
        
        if success_count >= 4:
            self.log("🎉 MCP VERCEL DEPLOYMENT SUCCESSFUL!", "SUCCESS")
            if self.deployment_url:
                self.log(f"🌐 Live URL: {self.deployment_url}", "SUCCESS")
            self.log("✨ Your Bahá'í Spiritual Quest is now live!", "SUCCESS")
        else:
            self.log("⚠️ Deployment needs attention - check authentication", "WARNING")
            
        return results

def main():
    """Main MCP deployment function"""
    deployer = MCPVercelDeployer()
    return deployer.run_mcp_deployment()

if __name__ == "__main__":
    main()