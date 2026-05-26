# AI QA Helper
# AI QA Helper

A lightweight AI-powered Question-Answering assistant built using **Tavily Search API** and **Groq LLM API**.  
This project retrieves relevant information from the web and generates accurate, context-aware responses in real time.

---

## Features

- Web search integration using Tavily API  
- Fast LLM responses via Groq API  
- Simple chatbot architecture  
- Lightweight and modular design  
- Easy to extend and customize  

---

## Project Structure
```bash
AI-QA-Helper/
|
├── chatbot/ # Core chatbot logic
├── venv1/ # Virtual environment (should be ignored)
├── pycache/ # Compiled Python files
├── .env # API keys (not included in repo)
├── agent.py # Agent logic (LLM + tools orchestration)
├── graph.py # Workflow / execution graph
├── main.py # Entry point
├── state.py # State management
├── tools.py # External tools (Tavily search, etc.)
├── requirements.txt

```

---

## 🛠️ Tech Stack

- Python  
- Tavily API (web search)  
- Groq API (LLM inference)  

---

## ⚙️ Environment Setup (Step-by-Step)

### 1. Create a virtual environment
```bash
python -m venv venv
```

## 2. Activate the environment

Windows: 
```bash
venv\Scripts\activate
```

Linux / Mac:
```bash
source venv/bin/activate
```

## 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 4. Create .env file
``` bash
touch .env        # Linux / Mac
type nul > .env   # Windows
```

## 5. Add API keys to .env

```bash
TAVILY_API_KEY=your_tavily_api_key
GROQ_API_KEY=your_groq_api_key
```

## 6. Run the project
``` bash
python main.py
```
## 7. Deactivate environment

```bash
deactivate
```


