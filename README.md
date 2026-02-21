# 🌟 Bahá'í Spiritual Quest - Autonomous Multi-Agent System

**The Hidden Words Digital Experience** - *كلمات مخفیه*

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yacinewatchandcode/bahai-spiritual-quest)

## 🎯 **Live Demo**
🚀 **[Live Application](https://bahai-spiritual-quest.vercel.app)** - Experience the divine wisdom

## 📖 **Overview**

This is a **Goose-inspired autonomous multi-agent system** for exploring the spiritual wisdom of Bahá'u'lláh's *Hidden Words*. Built with:

- 🤖 **Autonomous Configuration** - Goose-style interactive setup
- ✨ **Golden Quote Cards** - Beautiful Persian/Arabic text display
- 🧪 **100% QA Coverage** - 532+ comprehensive test scenarios  
- 📱 **Mobile-Ready** - React Native/Expo mobile app
- 🌟 **Real-time AI** - OpenRouter Horizon Beta integration

## 🚀 **Quick Start**

### **1. Autonomous Launch (Recommended)**
```bash
git clone https://github.com/yacinewatchandcode/bahai-spiritual-quest.git
cd bahai-spiritual-quest

# Auto-configure and launch everything
python backend/autonomous_launcher.py
```

### **2. Manual Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Configure OpenRouter API
python backend/goose_style_config.py

# Start backend
python backend/main.py

# Open http://localhost:PORT
```

### **3. Deploy to Vercel**
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

## 🎨 **Features**

### **🌟 Golden Quote System**
- **Persian Text Display** - Automatic Arabic rendering (ابن الانسان)
- **Beautiful Cards** - 3D perspective with golden gradients
- **Smart Detection** - Auto-formats Hidden Words quotes
- **Typography** - Amiri, Cinzel, Dancing Script fonts

### **🤖 Autonomous System**
- **Auto-configuration** - Interactive OpenRouter setup
- **Dependency management** - Automatic package installation
- **Health monitoring** - Continuous system checks
- **Self-recovery** - Automatic failure handling

### **🧪 Comprehensive QA**
- **532+ Test Scenarios** - Every interaction covered
- **Zero Bug Quest** - Continuous testing until perfection
- **Cross-browser Testing** - Chromium, WebKit, Firefox
- **Mobile Testing** - Responsive design validation

### **📱 Mobile Support**
- **React Native App** - Native iOS/Android experience
- **Expo Integration** - Easy development and deployment
- **Real-time Sync** - WebSocket communication
- **Offline Support** - Graceful fallback handling

## 🏗️ **Architecture**

```
Bahá'í Spiritual Quest/
├── backend/                    # FastAPI server
│   ├── main.py                # Main application
│   ├── autonomous_launcher.py # 🚀 Goose-style launcher
│   ├── rag_agent.py          # AI spiritual guidance
│   └── templates/             # UI templates
├── frontend/                  # Static web files  
├── mobile/                    # React Native app
├── .github/workflows/         # CI/CD automation
└── vercel.json               # Vercel deployment config
```

## 🌐 **API Endpoints**

- `GET /` - Main spiritual interface
- `POST /api/chat` - AI conversation endpoint
- `WS /ws` - Real-time WebSocket connection
- `POST /transcribe` - Voice input processing
- `GET /test` - QA testing interface

## 🔧 **Configuration**

### **Environment Variables**
```bash
OPENROUTER_API_KEY=your_api_key_here
SELECTED_MODEL=openrouter/horizon-beta
```

### **Vercel Deployment**
1. Fork this repository
2. Connect to Vercel
3. Add `OPENROUTER_API_KEY` environment variable
4. Deploy automatically on push

## 📊 **Testing**

### **Run Comprehensive QA**
```bash
# Run all 532+ test scenarios
python backend/comprehensive_qa_test.py

# Headless mode
python backend/comprehensive_qa_test.py --headless

# Zero Bug Quest mode
python backend/autonomous_launcher.py --qa
```

### **Mobile Testing**
```bash
cd mobile
npm install
npm start  # Opens Expo DevTools
```

## 🎭 **Spiritual Features**

### **The Hidden Words Integration**
- **Vector Database** - ChromaDB with spiritual texts
- **Semantic Search** - Find relevant wisdom passages
- **Context-Aware** - Responses tailored to spiritual needs
- **Reverent Design** - Interface honors sacred content

### **Persian/Arabic Support**
- **Beautiful Calligraphy** - Proper RTL text rendering
- **Font Loading** - Google Fonts integration
- **Cultural Sensitivity** - Respectful presentation

## 🌟 **Contributing**

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/spiritual-enhancement`)
3. Run comprehensive tests (`python backend/comprehensive_qa_test.py`)
4. Commit your changes (`git commit -m 'Add spiritual enhancement'`)
5. Push to branch (`git push origin feature/spiritual-enhancement`)
6. Open a Pull Request

## 📜 **License**

This project is dedicated to the spiritual upliftment of humanity. Use it with reverence and share the divine wisdom freely.

## 🙏 **Acknowledgments**

- **Bahá'u'lláh** - Author of The Hidden Words
- **OpenRouter** - AI infrastructure
- **Vercel** - Deployment platform
- **Goose** - Inspiration for autonomous architecture

---

<div align="center">

https://github.com/user-attachments/assets/demo.mp4

**▶️ Watch the Demo**

</div>

---


*"O Son of Being! Love Me, that I may love thee. If thou lovest Me not, My love can in no wise reach thee."*

**- Bahá'u'lláh, The Hidden Words**

---

🌟 **Made with spiritual devotion and technical excellence** ✨

[![Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=for-the-badge&logo=vercel)](https://vercel.com)
[![OpenRouter](https://img.shields.io/badge/AI-OpenRouter%20Horizon%20Beta-blue?style=for-the-badge)](https://openrouter.ai)
[![Bahá'í](https://img.shields.io/badge/Inspired%20by-Bah%C3%A1'%C3%AD%20Faith-gold?style=for-the-badge)](https://bahai.org)


## 🇪🇺 EU AI Act Compliance

This project follows EU AI Act (Regulation 2024/1689) guidelines:

| Requirement | Status | Reference |
|-------------|--------|-----------|
| **Risk Classification** | ✅ Assessed | Art. 6 — Categorized as minimal/limited risk |
| **Transparency** | ✅ Documented | Art. 52 — AI use clearly disclosed |
| **Data Governance** | ✅ Implemented | Art. 10 — Data handling documented |
| **Human Oversight** | ✅ Enabled | Art. 14 — Human-in-the-loop available |
| **Bias Mitigation** | ✅ Addressed | Art. 10(2)(f) — Fairness considered |
| **Logging & Audit** | ✅ Active | Art. 12 — System activity logged |

### AI Transparency Statement

This project uses AI models for data processing and analysis. All AI-generated outputs are clearly marked and subject to human review. No automated decision-making affects individual rights without human oversight.

### Data & Privacy

- Personal data is processed in accordance with GDPR (Regulation 2016/679)
- Data minimization principles are applied
- Users can request data access, correction, or deletion
- No data is shared with third parties without explicit consent

> For questions about AI compliance, contact: compliance@prime-ai.fr
