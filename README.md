# Spiritual Quest: Conversational Spiritual Guide

## Overview
Spiritual Quest is a conversational agent powered by a local LLM (Qwen or similar) with Retrieval-Augmented Generation (RAG) capabilities. It uses *The Hidden Words* by Bahá'u'lláh as its knowledge base, providing empathetic, spiritual responses to user queries via CLI and a React Native mobile app.

---

## Quick Start

1. **Clone or unzip the SQ_Baha folder.**
2. **Run the setup script:**
   ```sh
   cd SQ_Baha
   bash setup.sh
   ```
3. **Set up the backend Python environment:**
   ```sh
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   # Install core dependencies first
   pip install fastapi==0.109.2 uvicorn==0.27.1 requests==2.31.0 pydantic==2.6.1 python-multipart==0.0.9 qrcode==7.4.2 pillow==10.2.0 jinja2==3.1.3 python-jose==3.3.0 passlib==1.7.4 bcrypt==4.1.2 python-dotenv==1.0.1 aiofiles==23.2.1 pyyaml==6.0.1 rich==13.7.0
   # Then install ML and RAG dependencies
   pip install ollama==0.1.6 pypdf==4.0.2 chromadb==0.5.0 sentence-transformers==2.5.1
   # Fix NumPy version for ChromaDB compatibility
   pip uninstall -y numpy && pip install numpy==1.24.3
   # Install Whisper for audio transcription
   pip install openai-whisper
   # Finally, install crewai and langchain
   pip install crewai==0.30.11 langchain==0.1.20
   ```
4. **Start the backend:**
   ```sh
   source venv/bin/activate
   python3 main.py
   ```
5. **Start the mobile app:**
   ```sh
   cd ../mobile
   npm install
   npm start
   ```
   - Use Expo Go on your phone to scan the QR code.
   - Make sure your phone and computer are on the same WiFi network.
   - In `App.js`, set the WebSocket URL to your computer's local IP (e.g., `ws://192.168.1.171:8000/ws`)

---

## Features
- **Semantic search** over *The Hidden Words* (from a reformatted text file)
- **Conversational memory** for context-aware, empathetic responses
- **Exact quotation** (no verse numbers, original format preserved)
- **Text and voice input** (mobile app)
- **Local LLM** (Qwen or similar) for all generation
- **No hardcoded scripts or rules**

---

## Requirements

### 1. Local LLM
- **Qwen** (recommended) or any compatible local LLM that supports API inference
- The LLM must be accessible via an API endpoint on your machine (e.g., using [Ollama](https://ollama.com/) or [vllm](https://github.com/vllm-project/vllm))
- Make sure the LLM is running before starting the backend

### 2. Backend
- Python 3.10+
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [sentence-transformers](https://www.sbert.net/)
- [ChromaDB](https://www.trychroma.com/)
- [Whisper](https://github.com/openai/whisper) (for audio transcription)
- Other dependencies in `backend/requirements.txt`

### 3. Mobile App
- [Node.js](https://nodejs.org/) (v16+ recommended)
- [Expo CLI](https://docs.expo.dev/get-started/installation/)
- A physical device or emulator (iOS/Android)

---

## Directory Structure

```
SQ_Baha/
├── backend/
│   ├── main.py
│   ├── rag_agent.py
│   ├── requirements.txt
│   ├── hidden_words_reformatted.txt
│   ├── templates/
│   │   └── index.html
│   └── ... (other backend files)
├── mobile/
│   ├── App.js
│   ├── app.json
│   ├── index.js
│   ├── package.json
│   ├── package-lock.json
│   ├── tsconfig.json
│   ├── README.md
│   ├── assets/
│   │   ├── fonts/
│   │   └── images/
│   ├── components/
│   │   └── ...
│   ├── constants/
│   ├── hooks/
│   ├── scripts/
│   ├── .expo/
│   ├── .vscode/
│   └── .gitignore
├── README.md
└── setup.sh
```

---

## Usage
- **Text:** Type a message in the CLI or mobile app and receive a spiritual response.
- **Voice:** Hold the mic button in the mobile app, speak, and release. The app will transcribe and send your message.
- **All responses are direct quotations from The Hidden Words, in the original format.**

---

## Agent Rules (Summary)
| Rule/Keyword | Visual | Description |
|--------------|--------|------------------------------------------------------|
| SOURCE       | 📖     | Only uses the reformatted text file                  |
| QUOTE        | 🗣️     | Exact, original passage format                       |
| SEARCH       | 🔍🤖   | Semantic search for best match                       |
| MEMORY       | 🧠💬   | Remembers conversation context                       |
| EMPATHY      | 💖🙏   | Responds with empathy & spirituality                 |
| FREEDOM      | 🚫🤖📜 | No hardcoded scripts, only LLM+RAG                   |
| CLI/APP      | 💻📱   | Works in CLI and mobile app                          |

---

## Dependency Troubleshooting
- **NumPy/ChromaDB error:** If you see an error about `np.float_` being removed, downgrade NumPy:
  ```sh
  pip uninstall -y numpy && pip install numpy==1.24.3
  ```
- **Missing static directory:** If you see an error about `static` not existing, either create a `static/` folder in `backend/` or ignore this if you don't need to serve static files.
- **Whisper not found:** If you see `ModuleNotFoundError: No module named 'whisper'`, run:
  ```sh
  pip install openai-whisper
  ```

---

## Troubleshooting
- **Backend timeout:** Ensure the backend is running and accessible from your device (check firewall, correct IP, same WiFi).
- **LLM not responding:** Make sure your local LLM is running and the API endpoint is correct.
- **Audio issues:** Grant microphone permissions on your device. Try reinstalling the app if needed.
- **WebSocket errors:** Double-check the WebSocket URL in `App.js` matches your backend IP and port.

---

## Cloud Deployment
- For AWS or cloud deployment, see the [aws-samples/ray-serve-whisper-streaming-on-eks](https://github.com/aws-samples/ray-serve-whisper-streaming-on-eks) template and adapt your Dockerfile and deployment scripts accordingly.

---

## Questions?
If you have any issues, please open an issue or contact the maintainer. 
=======
# Integrated RPA Automation System

A comprehensive Robotic Process Automation (RPA) system that integrates KGG and OptimusPrime frameworks with advanced features including natural language control, predictive maintenance, and cross-system workflows.

## 🚀 Features

### Integrated RPA System
- Cross-system workflows combining KGG & OptimusPrime
- Robust error handling with detailed logging
- HTML reporting with execution metrics
- Parameter passing between systems

### Natural Language Control
- Conversational interface for RPA workflows
- Local LLM integration with OLLAMA
- Interactive mode for continuous commands
- Rule-based fallback for offline usage

### Predictive Maintenance
- Anomaly detection in automation logs
- Component health monitoring with metrics
- Actionable recommendations for optimization
- Visual HTML reports with priority levels

### UI Dashboard Integration
- Web-based control panel
- Real-time monitoring of agents
- Visual workflow execution tracking
- One-click automation triggering

## 📋 Requirements

- Python 3.10+
- Conda environment management
- Node.js (for UI dashboard)
- OLLAMA (for local LLM capabilities)

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/integrated-rpa-system.git
cd integrated-rpa-system

# Create and activate conda environment
conda env create -f environment.yml
conda activate agent_f1_env

# Install additional dependencies
pip install -U langmem crewai

# Install OLLAMA (macOS)
brew install ollama
ollama pull mistral

# Install UI dependencies
cd ui
npm install
```

## 🏃‍♂️ Quick Start

### Run Integrated RPA System
```bash
./run_integrated_rpa.sh
```

### Use Natural Language Interface
```bash
./run_nl_rpa.sh --interactive
```

### Run Predictive Maintenance
```bash
./run_predictive_maintenance.sh
```

### Start UI Dashboard
```bash
cd ui
npm run dev
```

## 🌐 Deployment

The RPA System is deployed using GitHub Pages for the UI and Render for the API backend.

### UI Deployment

The UI is automatically deployed to GitHub Pages when changes are pushed to the main branch. You can access the deployed UI at:

**https://yacinewhatchandcode.github.io/MultiAgenticSytemYBE/**

### API Deployment

The API can be deployed to Render.com using the following steps:

1. Create a new Web Service on Render.com
2. Connect your GitHub repository
3. Select the 'api' directory as the root directory
4. Set the build command to `pip install -r requirements.txt`
5. Set the start command to `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add the environment variable `PYTHON_VERSION=3.10.0`
7. Deploy the service
8. Update the `PRODUCTION_API_URL` in `ui/app/config.ts` with your Render service URL
9. Commit and push the changes to trigger a new UI deployment

### Local Development

To run the system locally for development:

1. Start the API:
   ```bash
   ./run_api.sh
   ```

2. Start the UI:
   ```bash
   cd ui
   npm run dev
   ```

3. Access the UI at http://localhost:3000

## 📁 Project Structure

```
├── integrated_rpa_automation.py  # Core integration system
├── nl_rpa_interface.py           # Natural language interface
├── predictive_maintenance.py      # Predictive maintenance system
├── run_integrated_rpa.sh         # Runner script for integrated system
├── run_nl_rpa.sh                 # Runner script for NL interface
├── run_predictive_maintenance.sh # Runner script for maintenance
├── environment.yml               # Conda environment definition
├── ui/                           # Web dashboard
├── samples/                      # Sample workflows
├── output/                       # Output directory
└── logs/                         # Log files
```

## 🤝 Integration

This system integrates with:
- KGG RPA System
- OptimusPrime Framework
- OLLAMA for local LLM capabilities
- Playwright for browser automation
- Selenium for UI testing
- OCR for image recognition

## 📄 License

MIT

## 🙏 Acknowledgements

- CrewAI for agent orchestration
- Langchain for LLM integration
- Playwright and Selenium for browser automation
- OLLAMA for local LLM capabilities
>>>>>>> 099446ff4377f4ca64f1001cb14032fa3f94e207
