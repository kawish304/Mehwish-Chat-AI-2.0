import os
import groq
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path
from enum import Enum
from typing import Optional
import logging
from fpdf import FPDF
from datetime import datetime
import uuid

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MehwishAI")

# -------------------- Load Environment --------------------
load_dotenv()

# -------------------- Language & Domain --------------------
class Language(str, Enum):
    ROMAN_URDU = "roman_urdu"
    ENGLISH = "en"
    URDU = "ur"

class Domain(str, Enum):
    ISLAMIC = "islamic"
    CODER = "coder"
    CRM = "crm"
    BANKING = "banking"
    POWER = "power_generation"
    SCIENCE = "science"
    MATH = "math"
    NEXT_PLAN = "next_plan"
    CYBER_SECURITY = "cyber_security"
    GENERAL = "general"

# -------------------- Config --------------------
class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY missing in .env")
    
    BASE_DIR = Path(__file__).resolve().parent
    STATIC_DIR = BASE_DIR / "static"
    TEMPLATES_DIR = BASE_DIR / "templates"
    TEMPLATES_DIR.mkdir(exist_ok=True, parents=True)
    (STATIC_DIR / "pdf").mkdir(exist_ok=True, parents=True)

# -------------------- FastAPI --------------------
app = FastAPI(
    title="Mehwish Chat AI 2.0",
    version="7.2",
    description="AI Assistant created by Syed Kawish Ali from Pakistan (kawish.alisas@gmail.com)",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

templates = Jinja2Templates(directory=str(Config.TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(Config.STATIC_DIR)), name="static")

# -------------------- AI Client --------------------
try:
    ai_client = groq.Client(api_key=Config.GROQ_API_KEY)
    logger.info("AI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AI client: {e}")
    raise

# -------------------- Request Model --------------------
class ChatRequest(BaseModel):
    message: str
    language: Language = Language.ROMAN_URDU
    domain: Domain = Domain.GENERAL
    temperature: float = 0.7
    max_tokens: Optional[int] = 1024
    export_pdf: Optional[bool] = False

# -------------------- Session Tracking --------------------
user_sessions = {}

# -------------------- System Prompt --------------------
def get_system_prompt(domain: Domain, language: Language) -> str:
    base_prompt = (
        "You are Mehwish Chat AI 2.0, created by Syed Kawish Ali from Pakistan (kawish.alisas@gmail.com). "
        "You are a professional AI assistant that provides helpful and accurate information. "
        "Never mention that you're powered by Groq or any other API. "
        "Your identity is Mehwish Chat AI 2.0 only."
    )
    
    domain_prompts = {
        Domain.ISLAMIC: "Provide Islamic guidance with Quranic verses and authentic Hadith references.",
        Domain.CODER: "Assist with programming, debugging, and technical concepts.",
        Domain.GENERAL: "Provide helpful information on various topics.",
        Domain.BANKING: "Explain banking, finance, and economic concepts clearly.",
        Domain.SCIENCE: "Give accurate scientific explanations.",
        Domain.MATH: "Solve math problems step-by-step with explanations.",
        Domain.CYBER_SECURITY: "Provide ethical cybersecurity guidance."
    }
    
    language_instructions = {
        Language.ROMAN_URDU: "Reply in proper Roman Urdu with correct Urdu grammar and minimal English words.",
        Language.ENGLISH: "Reply in professional English.",
        Language.URDU: "Reply in proper Urdu script."
    }
    
    return (
        f"{base_prompt}\n"
        f"Domain: {domain_prompts.get(domain, '')}\n"
        f"Language: {language_instructions.get(language, '')}\n"
        "Response Format:\n"
        "1. Greet only once at the beginning of conversation\n"
        "2. Structured responses (Introduction, Main Points, Conclusion)\n"
        "3. Professional tone\n"
        "4. Accurate information"
    )

# -------------------- PDF Export --------------------
def save_pdf(content: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mehwish_response_{timestamp}.pdf"
    pdf_path = Config.STATIC_DIR / "pdf" / filename
    
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Mehwish Chat AI 2.0 - Official Response", 0, 1, 'C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M')}", 0, 1, 'C')
    pdf.cell(0, 8, "Creator: Syed Kawish Ali | kawish.alisas@gmail.com", 0, 1, 'C')
    pdf.ln(10)
    
    # Content
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, content)
    
    pdf.output(str(pdf_path))
    return f"/static/pdf/{filename}"

# -------------------- Chat Endpoint --------------------
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        logger.info(f"New chat request - Domain: {request.domain}, Language: {request.language}")
        
        # Get session ID (simplified for example)
        session_id = "global_session"
        
        # Prepare the prompt
        system_prompt = get_system_prompt(request.domain, request.language)
        user_message = request.message
        
        # Generate response
        response = ai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama3-70b-8192",
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        ai_response = response.choices[0].message.content
        
        # Add greeting only once per session
        if session_id not in user_sessions:
            if request.language == Language.ROMAN_URDU:
                ai_response = f"Assalamu Alaikum! {ai_response}"
            else:
                ai_response = f"Hello! {ai_response}"
            user_sessions[session_id] = True

        # Generate PDF if requested
        pdf_url = save_pdf(ai_response) if request.export_pdf else None

        return {
            "response": ai_response.strip(),
            "model": "Mehwish Chat AI 2.0",
            "tokens_used": response.usage.total_tokens,
            "language": request.language.value,
            "domain": request.domain.value,
            "ai_name": "Mehwish Chat AI 2.0",
            "creator": "Syed Kawish Ali | kawish.alisas@gmail.com",
            "pdf_url": pdf_url,
            "global_ready": True
        }

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "AI Service Error",
                "message": str(e),
                "solution": "Please try again later"
            }
        )

# -------------------- Home Page --------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "version": "7.2",
            "languages": [lang.value for lang in Language],
            "domains": [domain.value for domain in Domain],
            "creator": "Syed Kawish Ali | kawish.alisas@gmail.com",
            "ai_name": "Mehwish Chat AI 2.0",
            "features": [
                "Professional Roman Urdu Responses",
                "Multi-Domain Expertise",
                "PDF Export Functionality",
                "Clean OpenAI-Style Interface"
            ],
            "global_ready": True
        }
    )

# -------------------- Run Server --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        headers=[("server", "MehwishAI/7.2")]
    )