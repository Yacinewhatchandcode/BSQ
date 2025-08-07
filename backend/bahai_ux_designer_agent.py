#!/usr/bin/env python3
"""
Baha'i UX/UI Designer Agent - A cutting-edge spiritual interface designer
Integrates with OpenRouter Horizon Beta for advanced AI capabilities
"""

import json
import os
from typing import Dict, Any, List, Optional
import requests
from datetime import datetime
import base64
from PIL import Image, ImageDraw, ImageFont
import io

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class BahaiUXDesignerAgent:
    """
    A sophisticated UX/UI designer agent inspired by Baha'i aesthetics
    Specializes in Persian calligraphy, spiritual interfaces, and sacred geometry
    """
    
    def __init__(self, model_name: str = "openrouter/horizon-beta"):
        self.model_name = model_name
        self.design_principles = {
            "colors": {
                "primary": "#D4AF37",  # Gold - divine light
                "secondary": "#8B4513",  # Saddle brown - earthen wisdom
                "accent": "#4169E1",  # Royal blue - spiritual depth
                "parchment": "#F4E8D0",  # Parchment background
                "ink": "#2F1B14",  # Dark brown ink
                "highlight": "#FFD700",  # Pure gold for quotes
            },
            "fonts": {
                "persian": "Noto Nastaliq Urdu",
                "arabic": "Amiri",
                "english": "Cinzel",
                "handwritten": "Dancing Script",
            },
            "patterns": {
                "nine_pointed_star": "Sacred Baha'i symbol",
                "arabesque": "Islamic geometric patterns",
                "calligraphy": "Persian nastaliq style",
                "illumination": "Persian manuscript borders",
            },
            "animations": {
                "reveal": "Gentle fade with golden particles",
                "scroll": "Parchment unrolling effect",
                "glow": "Soft ethereal light emanation",
                "write": "Calligraphic pen stroke animation",
            }
        }
        
        self.system_prompt = """You are a master Baha'i UX/UI designer and spiritual interface architect. 
        
Your expertise includes:
1. Persian and Arabic calligraphy traditions
2. Sacred geometry and spiritual symbolism
3. Manuscript illumination techniques
4. Modern web technologies (React, Vue, CSS animations)
5. Accessibility and inclusive design
6. Baha'i principles of unity, beauty, and harmony

Design Philosophy:
- Every interface element should inspire spiritual reflection
- Use the nine-pointed star subtly throughout designs
- Incorporate golden ratio and sacred proportions
- Blend ancient Persian aesthetics with modern functionality
- Create experiences that feel like discovering sacred manuscripts

When designing:
- Suggest specific CSS animations for quote reveals
- Provide detailed color gradients for backgrounds
- Recommend handwriting fonts and effects
- Design interactive elements that feel organic
- Create layouts that breathe with spiritual space"""
    
    def _call_openrouter(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenRouter API with Horizon Beta"""
        # Always try the real API call first since we have a valid key
        if not OPENROUTER_API_KEY or len(OPENROUTER_API_KEY) < 20:
            return self._get_fallback_design_response(messages[-1]["content"])
            
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Baha'i UX Designer"
        }
        
        payload = {
            "model": "openrouter/horizon-beta",  # Use correct model name
            "messages": messages,
            "temperature": 0.9,  # Higher creativity for design
            "max_tokens": 2000,
        }
        
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                return self._get_fallback_design_response(messages[-1]["content"])
        except Exception as e:
            return self._get_fallback_design_response(messages[-1]["content"])
    
    def _get_fallback_design_response(self, request: str) -> str:
        """Provide beautiful fallback design response"""
        return f"""
        🌟 **Baha'i-Inspired Design Concept**
        
        For your request: "{request}"
        
        **Visual Elements:**
        - Sacred nine-pointed star as central focal point
        - Golden divine palette (#D4AF37) with earthen browns
        - Persian calligraphy integration
        - Parchment textures with illuminated borders
        - Handwriting animations for sacred text
        
        **Typography:**
        - Amiri for Arabic/Persian text
        - Cinzel for English headers
        - Dancing Script for handwritten elements
        
        **Layout:**
        - Sacred geometry proportions (golden ratio)
        - Breathable white space for contemplation
        - Progressive revelation of content
        - Responsive design with mobile-first approach
        
        **Interactions:**
        - Gentle fade-in animations
        - Golden particle effects for quotes
        - Smooth transitions with spiritual timing
        - Voice integration for accessibility
        
        This design embodies the Baha'i principles of unity, beauty, and harmony.
        """
    
    def design_interface(self, request: str) -> Dict[str, Any]:
        """Generate a complete interface design based on request"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Design a Baha'i-inspired interface for: {request}"}
        ]
        
        design_response = self._call_openrouter(messages)
        
        # Generate specific CSS and HTML
        css_html = self.generate_css_html(request, design_response)
        
        return {
            "design_concept": design_response,
            "implementation": css_html,
            "preview_url": self.create_preview(css_html),
            "design_tokens": self.extract_design_tokens(design_response)
        }
    
    def generate_css_html(self, request: str, design_concept: str) -> Dict[str, str]:
        """Generate actual CSS and HTML based on design concept"""
        prompt = f"""Based on this Baha'i design concept:
{design_concept}

Generate production-ready CSS and HTML for: {request}

Include:
1. @font-face declarations for Persian/Arabic fonts
2. CSS animations for quote reveals
3. Dynamic background transitions
4. Handwriting effect animations
5. Responsive design
6. Accessibility features

Use these specific techniques:
- CSS custom properties for theming
- Keyframe animations for calligraphy effects
- Gradient backgrounds with noise texture
- Box-shadow for parchment depth
- Filter effects for aged paper look"""
        
        messages = [
            {"role": "system", "content": "You are an expert frontend developer specializing in CSS animations and Persian typography."},
            {"role": "user", "content": prompt}
        ]
        
        implementation = self._call_openrouter(messages)
        
        # Parse the response to extract CSS and HTML
        css = self._extract_code_block(implementation, "css")
        html = self._extract_code_block(implementation, "html")
        js = self._extract_code_block(implementation, "javascript")
        
        return {
            "css": css or self.get_default_css(),
            "html": html or self.get_default_html(),
            "javascript": js or self.get_default_js()
        }
    
    def _extract_code_block(self, text: str, language: str) -> Optional[str]:
        """Extract code block of specific language from text"""
        import re
        pattern = rf"```{language}\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1) if match else None
    
    def get_default_css(self) -> str:
        """Default Baha'i-inspired CSS"""
        return """
/* Baha'i Sacred Interface CSS */
@import url('https://fonts.googleapis.com/css2?family=Amiri:ital@0;1&family=Cinzel:wght@400;600&family=Dancing+Script:wght@400;700&display=swap');

:root {
  --gold-divine: #D4AF37;
  --brown-earth: #8B4513;
  --blue-spirit: #4169E1;
  --parchment: #F4E8D0;
  --ink-dark: #2F1B14;
  --gold-pure: #FFD700;
  --shadow-soft: rgba(0, 0, 0, 0.1);
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Cinzel', serif;
  background: linear-gradient(135deg, var(--parchment) 0%, #E8D7C3 100%);
  min-height: 100vh;
  position: relative;
  overflow-x: hidden;
}

body::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-image: 
    repeating-linear-gradient(45deg, transparent, transparent 10px, rgba(212, 175, 55, 0.03) 10px, rgba(212, 175, 55, 0.03) 20px),
    repeating-linear-gradient(-45deg, transparent, transparent 10px, rgba(139, 69, 19, 0.02) 10px, rgba(139, 69, 19, 0.02) 20px);
  pointer-events: none;
}

.parchment-container {
  max-width: 900px;
  margin: 40px auto;
  background: var(--parchment);
  border-radius: 10px;
  box-shadow: 
    0 20px 40px rgba(0, 0, 0, 0.1),
    inset 0 0 60px rgba(139, 69, 19, 0.1);
  position: relative;
  padding: 60px;
  overflow: hidden;
}

.parchment-container::before {
  content: '';
  position: absolute;
  top: -2px;
  left: -2px;
  right: -2px;
  bottom: -2px;
  background: linear-gradient(45deg, var(--gold-divine), var(--brown-earth), var(--blue-spirit), var(--gold-divine));
  border-radius: 10px;
  opacity: 0.3;
  z-index: -1;
  animation: shimmer 3s ease-in-out infinite;
}

@keyframes shimmer {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 0.5; }
}

.sacred-header {
  text-align: center;
  margin-bottom: 40px;
  position: relative;
}

.nine-pointed-star {
  width: 80px;
  height: 80px;
  margin: 0 auto 20px;
  position: relative;
  animation: rotate-slow 60s linear infinite;
}

@keyframes rotate-slow {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.title-persian {
  font-family: 'Amiri', serif;
  font-size: 3em;
  color: var(--gold-divine);
  text-shadow: 2px 2px 4px var(--shadow-soft);
  margin-bottom: 10px;
  animation: fade-in 2s ease-out;
}

@keyframes fade-in {
  from { 
    opacity: 0;
    transform: translateY(-20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.handwritten-text {
  font-family: 'Dancing Script', cursive;
  font-size: 1.8em;
  color: var(--ink-dark);
  position: relative;
  display: inline-block;
  animation: write 3s ease-out forwards;
}

@keyframes write {
  from {
    clip-path: inset(0 100% 0 0);
  }
  to {
    clip-path: inset(0 0 0 0);
  }
}

.quote-container {
  background: linear-gradient(135deg, rgba(255, 215, 0, 0.1) 0%, rgba(212, 175, 55, 0.2) 100%);
  border-left: 4px solid var(--gold-divine);
  padding: 30px;
  margin: 30px 0;
  position: relative;
  border-radius: 0 10px 10px 0;
  animation: slide-in 1s ease-out;
}

@keyframes slide-in {
  from {
    opacity: 0;
    transform: translateX(-50px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

.quote-text {
  font-family: 'Amiri', serif;
  font-size: 1.6em;
  color: var(--ink-dark);
  line-height: 1.8;
  font-style: italic;
}

.quote-attribution {
  text-align: right;
  margin-top: 20px;
  font-size: 1.1em;
  color: var(--brown-earth);
}

.illuminated-border {
  position: absolute;
  top: 20px;
  left: 20px;
  right: 20px;
  bottom: 20px;
  border: 2px solid var(--gold-divine);
  border-radius: 10px;
  pointer-events: none;
  opacity: 0.3;
}

.illuminated-border::before,
.illuminated-border::after {
  content: '';
  position: absolute;
  width: 60px;
  height: 60px;
  border: 2px solid var(--gold-divine);
}

.illuminated-border::before {
  top: -30px;
  left: -30px;
  border-right: none;
  border-bottom: none;
}

.illuminated-border::after {
  bottom: -30px;
  right: -30px;
  border-left: none;
  border-top: none;
}

.interactive-element {
  background: var(--parchment);
  border: 2px solid var(--gold-divine);
  padding: 15px 30px;
  border-radius: 25px;
  font-family: 'Cinzel', serif;
  font-weight: 600;
  color: var(--ink-dark);
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.interactive-element::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  width: 0;
  height: 0;
  background: radial-gradient(circle, var(--gold-pure) 0%, transparent 70%);
  transform: translate(-50%, -50%);
  transition: width 0.6s ease, height 0.6s ease;
}

.interactive-element:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 20px var(--shadow-soft);
}

.interactive-element:hover::before {
  width: 300px;
  height: 300px;
}

/* Dynamic background for quotes */
.dynamic-quote-bg {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
  opacity: 0;
  transition: opacity 2s ease;
  z-index: -1;
}

.dynamic-quote-bg.active {
  opacity: 1;
  background: 
    radial-gradient(circle at 20% 50%, rgba(255, 215, 0, 0.3) 0%, transparent 50%),
    radial-gradient(circle at 80% 50%, rgba(212, 175, 55, 0.3) 0%, transparent 50%),
    radial-gradient(circle at 50% 50%, rgba(65, 105, 225, 0.2) 0%, transparent 70%);
  animation: pulse-bg 4s ease-in-out infinite;
}

@keyframes pulse-bg {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.05); }
}

/* Responsive design */
@media (max-width: 768px) {
  .parchment-container {
    margin: 20px;
    padding: 40px 30px;
  }
  
  .title-persian {
    font-size: 2em;
  }
  
  .handwritten-text {
    font-size: 1.4em;
  }
}
"""
    
    def get_default_html(self) -> str:
        """Default Baha'i-inspired HTML structure"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sacred Digital Manuscript - Baha'i Spiritual Interface</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="dynamic-quote-bg"></div>
    
    <div class="parchment-container">
        <div class="illuminated-border"></div>
        
        <header class="sacred-header">
            <div class="nine-pointed-star">
                <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                    <path d="M50,10 L61,35 L88,28 L72,50 L88,72 L61,65 L50,90 L39,65 L12,72 L28,50 L12,28 L39,35 Z" 
                          fill="none" 
                          stroke="var(--gold-divine)" 
                          stroke-width="2"
                          stroke-linejoin="round"/>
                </svg>
            </div>
            <h1 class="title-persian">بهائی</h1>
            <p class="subtitle">Sacred Digital Experience</p>
        </header>
        
        <main class="content-area">
            <div class="handwritten-section">
                <p class="handwritten-text">
                    Welcome to this sacred digital space where ancient wisdom meets modern technology...
                </p>
            </div>
            
            <div class="quote-container">
                <blockquote class="quote-text">
                    "Let your vision be world-embracing, rather than confined to your own self."
                </blockquote>
                <cite class="quote-attribution">— Bahá'u'lláh</cite>
            </div>
            
            <div class="interactive-section">
                <button class="interactive-element" onclick="revealQuote()">
                    Reveal Sacred Wisdom
                </button>
            </div>
        </main>
    </div>
    
    <script src="script.js"></script>
</body>
</html>
"""
    
    def get_default_js(self) -> str:
        """Default JavaScript for interactions"""
        return """
// Sacred Interface Interactions
const quotes = [
    {
        text: "The earth is but one country, and mankind its citizens.",
        author: "Bahá'u'lláh",
        background: "linear-gradient(135deg, rgba(255,215,0,0.3), rgba(65,105,225,0.3))"
    },
    {
        text: "Be generous in prosperity, and thankful in adversity.",
        author: "Bahá'u'lláh",
        background: "linear-gradient(135deg, rgba(212,175,55,0.3), rgba(139,69,19,0.3))"
    },
    {
        text: "The best beloved of all things in My sight is Justice.",
        author: "Bahá'u'lláh",
        background: "linear-gradient(135deg, rgba(65,105,225,0.3), rgba(255,215,0,0.3))"
    }
];

let currentQuoteIndex = 0;

function revealQuote() {
    const quoteContainer = document.querySelector('.quote-container');
    const quoteText = document.querySelector('.quote-text');
    const quoteAuthor = document.querySelector('.quote-attribution');
    const dynamicBg = document.querySelector('.dynamic-quote-bg');
    
    // Fade out current quote
    quoteContainer.style.opacity = '0';
    
    setTimeout(() => {
        // Update quote
        const quote = quotes[currentQuoteIndex];
        quoteText.textContent = `"${quote.text}"`;
        quoteAuthor.textContent = `— ${quote.author}`;
        
        // Update background
        dynamicBg.style.background = quote.background;
        dynamicBg.classList.add('active');
        
        // Fade in new quote with animation
        quoteContainer.style.opacity = '1';
        quoteContainer.style.animation = 'none';
        setTimeout(() => {
            quoteContainer.style.animation = 'slide-in 1s ease-out';
        }, 10);
        
        // Cycle through quotes
        currentQuoteIndex = (currentQuoteIndex + 1) % quotes.length;
    }, 500);
    
    // Add golden particles effect
    createGoldenParticles();
}

function createGoldenParticles() {
    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.className = 'golden-particle';
        particle.style.cssText = `
            position: absolute;
            width: 4px;
            height: 4px;
            background: var(--gold-pure);
            border-radius: 50%;
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
            animation: float-particle 3s ease-out forwards;
            pointer-events: none;
        `;
        document.body.appendChild(particle);
        
        setTimeout(() => particle.remove(), 3000);
    }
}

// Add CSS for particles
const style = document.createElement('style');
style.textContent = `
    @keyframes float-particle {
        0% {
            opacity: 0;
            transform: translateY(0) scale(0);
        }
        50% {
            opacity: 1;
            transform: translateY(-50px) scale(1);
        }
        100% {
            opacity: 0;
            transform: translateY(-100px) scale(0);
        }
    }
`;
document.head.appendChild(style);

// Initialize handwriting animation on scroll
const handwrittenElements = document.querySelectorAll('.handwritten-text');
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.animation = 'write 3s ease-out forwards';
        }
    });
});

handwrittenElements.forEach(el => observer.observe(el));

// Add hover effects for interactive elements
document.querySelectorAll('.interactive-element').forEach(element => {
    element.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-2px)';
    });
    
    element.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0)';
    });
});
"""
    
    def create_preview(self, css_html: Dict[str, str]) -> str:
        """Create a preview URL or data URI for the design"""
        # In a real implementation, this would create a preview
        # For now, return a placeholder
        return "preview_available_on_deployment"
    
    def extract_design_tokens(self, design_response: str) -> Dict[str, Any]:
        """Extract design tokens from the AI response"""
        return {
            "colors": self.design_principles["colors"],
            "typography": self.design_principles["fonts"],
            "spacing": {
                "xs": "4px",
                "sm": "8px",
                "md": "16px",
                "lg": "32px",
                "xl": "64px"
            },
            "animations": {
                "duration": {
                    "fast": "200ms",
                    "normal": "500ms",
                    "slow": "1000ms",
                    "very_slow": "3000ms"
                },
                "easing": {
                    "default": "ease-out",
                    "smooth": "cubic-bezier(0.4, 0, 0.2, 1)",
                    "bounce": "cubic-bezier(0.68, -0.55, 0.265, 1.55)"
                }
            }
        }
    
    def design_quote_display(self, quote: str, author: str) -> Dict[str, str]:
        """Design a beautiful display for a specific quote"""
        prompt = f"""Design a stunning visual display for this Baha'i quote:
"{quote}" - {author}

Requirements:
1. Persian/Arabic calligraphy style for the quote
2. Elegant handwriting animation
3. Dynamic background that reflects the quote's meaning
4. Golden illumination effects
5. Responsive design that works on all devices
6. Accessibility features (ARIA labels, contrast)

The design should feel like discovering a sacred manuscript with modern interactivity."""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_openrouter(messages)
        implementation = self.generate_css_html(f"Quote display: {quote}", response)
        
        return {
            "concept": response,
            "implementation": implementation,
            "preview": self.create_quote_preview(quote, author, implementation)
        }
    
    def create_quote_preview(self, quote: str, author: str, implementation: Dict[str, str]) -> str:
        """Generate a preview image of the quote design"""
        # This would generate an actual image in production
        # For now, return a data structure
        return {
            "quote": quote,
            "author": author,
            "styles": implementation["css"],
            "markup": implementation["html"]
        }
    
    def chat(self, message: str) -> str:
        """Interactive chat for design consultations"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": message}
        ]
        
        response = self._call_openrouter(messages)
        
        # If the response mentions creating something, generate it
        if any(keyword in message.lower() for keyword in ["design", "create", "build", "make"]):
            design = self.design_interface(message)
            return f"{response}\n\n---\nI've created a design for you:\n{json.dumps(design, indent=2)}"
        
        return response


# Example usage
if __name__ == "__main__":
    designer = BahaiUXDesignerAgent()
    
    # Example: Design a quote display
    quote_design = designer.design_quote_display(
        "The earth is but one country, and mankind its citizens.",
        "Bahá'u'lláh"
    )
    
    print("Quote Design Generated:")
    print(json.dumps(quote_design, indent=2))