#!/usr/bin/env python3
"""
Enhanced Conversational Deployment with Memory and Context
Fix the conversation functionality issues identified in QA testing
"""

import os
import subprocess
import tempfile
import shutil
import json
from pathlib import Path

def create_enhanced_html_with_conversation():
    """Create HTML with enhanced conversational capabilities"""
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
            scroll-behavior: smooth;
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

        /* Conversation context indicator */
        .context-indicator {
            position: fixed;
            top: 30px;
            left: 30px;
            padding: 8px 15px;
            background: rgba(212, 175, 55, 0.2);
            border: 1px solid rgba(212, 175, 55, 0.3);
            border-radius: 20px;
            color: #D4AF37;
            font-size: 0.8em;
            z-index: 1000;
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .context-indicator.show {
            opacity: 1;
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

            .context-indicator, .status {
                position: relative;
                top: auto;
                left: auto;
                right: auto;
                margin: 10px auto;
                display: block;
                width: fit-content;
            }
        }
    </style>
</head>
<body>
    <!-- Starfield -->
    <div class="starfield" id="starfield"></div>
    
    <!-- Status indicator -->
    <div class="status" id="status"></div>
    
    <!-- Context indicator -->
    <div class="context-indicator" id="contextIndicator">💭 Conversational Memory Active</div>
    
    <!-- Deployment badge -->
    <div class="deployment-badge">🚀 Enhanced Conversational Agent</div>
    
    <!-- Main container -->
    <div class="manuscript-container">
        <!-- Persian title -->
        <h1 class="title-persian">كلمات مخفیه</h1>
        
        <!-- Subtitle -->
        <p class="subtitle">The Hidden Words - Sacred Digital Manuscript</p>
        
        <!-- Welcome message -->
        <div class="welcome-message">
            Welcome to this sacred digital space, where the eternal wisdom of <em>The Hidden Words</em> illuminates our spiritual journey. I am here to share these divine pearls with you, each one a treasure from the ocean of divine knowledge. <em>Ask me anything, and I will remember our conversation.</em>
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
        // Enhanced conversation system with memory and context
        class SpiritualConversationAgent {
            constructor() {
                this.conversationHistory = [];
                this.userProfile = {
                    interests: [],
                    previousTopics: [],
                    spiritualJourney: []
                };
                this.contextMemory = new Map();
                this.sessionId = this.generateSessionId();
                
                this.spiritualResponses = [
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
                ];
            }

            generateSessionId() {
                return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
            }

            addToHistory(message, response, context = {}) {
                this.conversationHistory.push({
                    timestamp: new Date().toISOString(),
                    userMessage: message,
                    agentResponse: response,
                    context: context
                });
                
                // Keep last 10 exchanges for context
                if (this.conversationHistory.length > 10) {
                    this.conversationHistory.shift();
                }
            }

            updateUserProfile(message) {
                const messageLower = message.toLowerCase();
                
                // Extract interests
                const spiritualKeywords = ['love', 'peace', 'justice', 'unity', 'prayer', 'meditation', 'service', 'devotion'];
                spiritualKeywords.forEach(keyword => {
                    if (messageLower.includes(keyword) && !this.userProfile.interests.includes(keyword)) {
                        this.userProfile.interests.push(keyword);
                    }
                });
                
                // Track spiritual journey indicators
                if (messageLower.includes('seeking') || messageLower.includes('searching')) {
                    this.userProfile.spiritualJourney.push('seeker');
                }
                if (messageLower.includes('struggling') || messageLower.includes('difficult')) {
                    this.userProfile.spiritualJourney.push('challenge');
                }
                if (messageLower.includes('grateful') || messageLower.includes('blessed')) {
                    this.userProfile.spiritualJourney.push('gratitude');
                }
            }

            getContextualResponse(message) {
                const messageLower = message.toLowerCase();
                const recentHistory = this.conversationHistory.slice(-3);
                
                // Check for follow-up questions
                if (messageLower.includes('what do you mean') || messageLower.includes('explain') || messageLower.includes('tell me more')) {
                    if (recentHistory.length > 0) {
                        const lastResponse = recentHistory[recentHistory.length - 1].agentResponse;
                        if (lastResponse.includes('O Son of') || lastResponse.includes('O Friend')) {
                            return this.explainHiddenWord(lastResponse);
                        }
                    }
                }
                
                // Check for personal references to previous conversation
                if (messageLower.includes('you said') || messageLower.includes('earlier') || messageLower.includes('before')) {
                    if (recentHistory.length > 0) {
                        return `I remember our conversation. ${this.getContextualQuote(messageLower)} This builds upon what we discussed earlier about ${this.extractPreviousTopics()}.`;
                    }
                }
                
                // Personalized response based on user profile
                if (this.userProfile.interests.length > 0) {
                    const userInterests = this.userProfile.interests.join(', ');
                    if (messageLower.includes('help') || messageLower.includes('guidance')) {
                        return `Given your interest in ${userInterests}, let me share this guidance: ${this.getPersonalizedQuote(messageLower)}`;
                    }
                }
                
                // Default contextual response
                return this.getContextualQuote(messageLower);
            }

            explainHiddenWord(hiddenWord) {
                const explanations = {
                    'Love Me, that I may love thee': 'This Hidden Word speaks to the reciprocal nature of divine love. When we open our hearts to love the Divine, we create a channel through which divine love can flow back to us.',
                    'garden of thy heart': 'The heart is likened to a garden that must be tended carefully. We must plant only love and remove the weeds of hatred and negativity.',
                    'fortitude in My decree': 'True love is shown not in easy times, but in our patience and strength during trials and difficulties.',
                    'same dust': 'This reminds us of the fundamental equality of all humanity - we are all created from the same spiritual essence.'
                };
                
                for (const [key, explanation] of Object.entries(explanations)) {
                    if (hiddenWord.includes(key)) {
                        return explanation;
                    }
                }
                
                return 'Each Hidden Word is a pearl of divine wisdom, revealed to guide our spiritual journey. Reflect deeply on its meaning and how it applies to your life.';
            }

            extractPreviousTopics() {
                if (this.conversationHistory.length === 0) return 'spiritual wisdom';
                
                const recentTopics = this.conversationHistory
                    .slice(-2)
                    .map(entry => entry.userMessage)
                    .join(' ')
                    .toLowerCase();
                
                if (recentTopics.includes('love')) return 'divine love';
                if (recentTopics.includes('justice')) return 'justice and fairness';
                if (recentTopics.includes('unity')) return 'unity and oneness';
                if (recentTopics.includes('service')) return 'service to humanity';
                
                return 'spiritual growth';
            }

            getPersonalizedQuote(message) {
                const interests = this.userProfile.interests;
                
                if (interests.includes('love')) return this.spiritualResponses[0];
                if (interests.includes('justice')) return this.spiritualResponses[12];
                if (interests.includes('unity')) return this.spiritualResponses[3];
                if (interests.includes('service')) return this.spiritualResponses[13];
                
                return this.getContextualQuote(message);
            }

            getContextualQuote(message) {
                const messageLower = message.toLowerCase();
                
                // Enhanced keyword matching with context
                if (messageLower.includes('love') || messageLower.includes('loving') || messageLower.includes('beloved')) {
                    return this.spiritualResponses[0];
                }
                if (messageLower.includes('friend') || messageLower.includes('friendship') || messageLower.includes('companion')) {
                    return this.spiritualResponses[1];
                }
                if (messageLower.includes('sign') || messageLower.includes('patience') || messageLower.includes('trial')) {
                    return this.spiritualResponses[2];
                }
                if (messageLower.includes('equality') || messageLower.includes('equal') || messageLower.includes('dust')) {
                    return this.spiritualResponses[3];
                }
                if (messageLower.includes('noble') || messageLower.includes('created') || messageLower.includes('purpose')) {
                    return this.spiritualResponses[4];
                }
                if (messageLower.includes('light') || messageLower.includes('brightness') || messageLower.includes('glory')) {
                    return this.spiritualResponses[11];
                }
                if (messageLower.includes('heart') || messageLower.includes('home') || messageLower.includes('spirit')) {
                    return this.spiritualResponses[6];
                }
                if (messageLower.includes('justice') || messageLower.includes('just') || messageLower.includes('fair')) {
                    return this.spiritualResponses[12];
                }
                if (messageLower.includes('rich') || messageLower.includes('poor') || messageLower.includes('wealth')) {
                    return this.spiritualResponses[13];
                }
                if (messageLower.includes('joy') || messageLower.includes('gladness') || messageLower.includes('rejoice')) {
                    return this.spiritualResponses[10];
                }
                
                // Contextual selection based on conversation flow
                const hash = message.split('').reduce((a, b) => { a = ((a << 5) - a) + b.charCodeAt(0); return a & a }, 0);
                return this.spiritualResponses[Math.abs(hash) % this.spiritualResponses.length];
            }

            processMessage(message) {
                this.updateUserProfile(message);
                
                const response = this.getContextualResponse(message);
                const context = {
                    sessionId: this.sessionId,
                    messageCount: this.conversationHistory.length + 1,
                    userInterests: this.userProfile.interests,
                    spiritualJourney: this.userProfile.spiritualJourney
                };
                
                this.addToHistory(message, response, context);
                
                return {
                    response: response,
                    context: context,
                    conversationLength: this.conversationHistory.length,
                    isHiddenWord: response.includes('O Son of') || response.includes('O Friend') || response.includes('O Children')
                };
            }

            getConversationSummary() {
                return {
                    totalExchanges: this.conversationHistory.length,
                    userInterests: this.userProfile.interests,
                    spiritualJourney: this.userProfile.spiritualJourney,
                    sessionId: this.sessionId
                };
            }
        }

        // Initialize the enhanced conversation agent
        const conversationAgent = new SpiritualConversationAgent();
        
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
        
        // Enhanced send message function with conversation memory
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) {
                showStatus('Please enter your spiritual inquiry', 'error');
                return;
            }
            
            // Show chat area and context indicator if hidden
            const chatArea = document.getElementById('chatArea');
            const contextIndicator = document.getElementById('contextIndicator');
            
            chatArea.style.display = 'block';
            contextIndicator.classList.add('show');
            
            addMessage(message, 'user');
            input.value = '';
            
            showStatus('Seeking divine wisdom with contextual understanding...', 'success');
            
            // Process message through conversation agent
            setTimeout(() => {
                const result = conversationAgent.processMessage(message);
                
                if (result.isHiddenWord) {
                    addHiddenWordMessage(result.response, 'agent');
                } else {
                    addMessage(result.response, 'agent');
                }
                
                // Update status with conversation context
                const summary = conversationAgent.getConversationSummary();
                showStatus(`Divine wisdom revealed (Exchange ${result.conversationLength}, Interests: ${summary.userInterests.length})`, 'success');
                
            }, 1200);
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
            
            // Smooth scroll to bottom
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
            
            // Smooth scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // Handle Enter key
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        // Show status message with enhanced duration
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = `status ${type}`;
            status.classList.add('show');
            
            setTimeout(() => {
                status.classList.remove('show');
            }, 5000); // Increased from 3000 to 5000ms based on QA feedback
        }
        
        // Initialize
        window.addEventListener('load', function() {
            createStarfield();
            console.log('🌟 Enhanced Bahá\\'í Spiritual Quest loaded successfully');
            console.log('✨ Ready for contextual conversations with memory');
            console.log('💭 Conversation agent initialized with session:', conversationAgent.sessionId);
        });
    </script>
</body>
</html>'''

def deploy_enhanced_conversation():
    """Deploy enhanced conversational version"""
    
    # Create a temporary deployment directory
    with tempfile.TemporaryDirectory(prefix="bahai_enhanced_") as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create files
        with open(temp_path / "index.html", 'w', encoding='utf-8') as f:
            f.write(create_enhanced_html_with_conversation())
        
        # Create minimal vercel.json
        vercel_config = {"version": 2}
        with open(temp_path / "vercel.json", 'w') as f:
            json.dump(vercel_config, f, indent=2)
        
        print(f"📁 Created enhanced conversational deployment at: {temp_path}")
        
        # Change to temp directory and deploy
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_path)
            
            # Deploy enhanced version
            result = subprocess.run([
                "vercel", "--prod", "--yes"
            ], capture_output=True, text=True)
            
            print("=== ENHANCED DEPLOYMENT OUTPUT ===")
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
    result = deploy_enhanced_conversation()
    if result:
        print("✅ Enhanced conversational deployment successful!")
        print(result)
    else:
        print("❌ Deployment failed")