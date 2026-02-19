================================================================================
  MULTI-AGENT CHURN MITIGATION SYSTEM — Master's Thesis Project by DOGRU
================================================================================

PREREQUISITES:
  - Python 3.10 or higher must be installed on the system
  - Internet connection is required (the system calls LLM APIs)
  - At least one valid API key must be configured in the .env file (see below)

================================================================================
QUICK START (Recommended)
================================================================================

  Double-click START.bat
  
  This will automatically:
    1. Create a virtual environment (venv/) if not already present
    2. Install all dependencies from requirements.txt
    3. Run the pre-flight LLM provider check (tests each API key)
    4. Run the main pipeline (main.py)
  
  The pre-flight check tests every API key with a minimal request before
  starting. Only providers that respond successfully are used. If NO
  provider passes, the system prints instructions for obtaining free keys.

  The entire process takes 3-10 minutes depending on API response times
  and the number of N_RUNS trials (default: 10).

  All output files (charts, reports, data) will be saved to the outputs/ folder.

================================================================================
MANUAL SETUP (If START.bat does not work)
================================================================================

  Open a terminal (Command Prompt or PowerShell) in this folder, then run:

  Step 1: Create virtual environment
    python -m venv venv

  Step 2: Activate virtual environment
    Windows CMD:        venv\Scripts\activate.bat
    Windows PowerShell: .\venv\Scripts\Activate.ps1
  
  Step 3: Install dependencies
    pip install -r requirements.txt

  Step 4: Run the pipeline
    python main.py

  If PowerShell blocks the activation script, run this first:
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

================================================================================
API KEYS (.env file)
================================================================================

  The .env file must contain at least one valid API key. The system supports
  multiple providers and rotates between them automatically if rate limits
  are hit. Supported providers (priority order):

    GROQ_API_KEY=gsk_...              (recommended — fastest)
    GEMINI_API_KEY_1=AIza...
    GEMINI_API_KEY_2=AIza...
    GEMINI_API_KEY_3=AIza...
    XAI_API_KEY=xai-...
    OPEN_ROUTER_API_KEY=sk-or-v1-...
    OPEN_ROUTER_API_KEY_2=sk-or-v1-...
    CEREBRAS_API_KEY=csk-...

  Free tier keys are sufficient. The system uses ~5,000-10,000 tokens per run.

  HOW TO GET FREE API KEYS:
    Groq:       https://console.groq.com/keys
    Gemini:     https://aistudio.google.com/app/apikey
    OpenRouter: https://openrouter.ai/keys
    Cerebras:   https://cloud.cerebras.ai/
    xAI (Grok): https://console.x.ai/

