# main.py - Mehwish Chat AI 3.5 (Multi-language / OpenAI-style)
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from pathlib import Path
from dotenv import load_dotenv
import logging, os, aiohttp, asyncio, uuid, json, re, shutil, random
from typing import Optional, List, Literal
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import io, math

# -----------------
# Setup & Logging
# -----------------
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)

# -----------------
# YOUR PATHS (using relative paths)
# -----------------
current_dir = Path(__file__).parent
TEMPLATES_DIR = current_dir / "templates"
STATIC_DIR = current_dir / "static"

# Ensure folders exist
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "generated_media").mkdir(parents=True, exist_ok=True)

# Create placeholder files if they don't exist
def create_placeholder_files():
    # Create a simple placeholder image
    placeholder_jpg_path = STATIC_DIR / "placeholder.jpg"
    if not placeholder_jpg_path.exists():
        # Create a minimal black image with text
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (300, 200), color='black')
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        d.text((50, 90), "Mehwish AI Placeholder", fill=(255, 255, 0), font=font)
        img.save(placeholder_jpg_path, 'JPEG')
        logger.info("Created placeholder.jpg")
    
    # Create a simple placeholder video
    placeholder_mp4_path = STATIC_DIR / "placeholder.mp4"
    if not placeholder_mp4_path.exists():
        # Create a text file as we can't create video programmatically easily
        with open(placeholder_mp4_path, 'w') as f:
            f.write("Placeholder video file - use real video file for actual video playback")
        logger.info("Created placeholder.mp4")

# Create placeholder files on startup
create_placeholder_files()

# -----------------
# Pixels API Implementation
# -----------------
class PixelsAPI:
    def __init__(self):
        self.api_key = os.getenv("PIXELS_API_KEY")
        self.base_url = "https://api.pexels.com"
    
    async def generate_image(self, prompt: str) -> str:
        if not self.api_key:
            logger.warning("PIXELS_API_KEY not set, using placeholder")
            return "/static/placeholder.jpg"
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": self.api_key}
                params = {"query": prompt, "per_page": 1, "size": "medium"}
                
                async with session.get(
                    f"{self.base_url}/v1/search",
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("photos") and len(data["photos"]) > 0:
                            return data["photos"][0]["src"]["original"]
                    
                    logger.warning("No images found, using placeholder")
                    return "/static/placeholder.jpg"
                    
        except Exception as e:
            logger.error(f"Pixels API error: {str(e)}")
            return "/static/placeholder.jpg"
    
    async def generate_video(self, prompt: str) -> str:
        if not self.api_key:
            logger.warning("PIXELS_API_KEY not set, using placeholder")
            return "/static/placeholder.mp4"
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": self.api_key}
                params = {"query": prompt, "per_page": 1}
                
                async with session.get(
                    f"{self.base_url}/videos/search",
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("videos") and len(data["videos"]) > 0:
                            return data["videos"][0]["video_files"][0]["link"]
                    
                    logger.warning("No videos found, using placeholder")
                    return "/static/placeholder.mp4"
                    
        except Exception as e:
            logger.error(f"Pixels API error: {str(e)}")
            return "/static/placeholder.mp4"

# Initialize PixelsAPI
pixels_api = PixelsAPI()

# -----------------
# FastAPI app
# -----------------
app = FastAPI(
    title="Mehwish Chat AI 3.5 - World's Best AI Assistant",
    description="Multi-language Support (Mehwish-style), Groq LLaMA & Pixel Media. Coder Mode with multiple subjects.",
    version="3.5.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    contact={"name":"Syed Kawish Ali","email":"kawish.alisas@gmail.com"}
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# In-memory store
conversation_history: dict = {}

# -----------------
# Models
# -----------------
class ChatRequest(BaseModel):
    message: str
    image: bool = False
    video: bool = False
    language: str = "english"
    subject: str = "general"
    conversation_id: Optional[str] = None

class SEORequest(BaseModel):
    text: str
    max_keywords: int = 10
    language: str = "english"  # Added language parameter

class CodeGenRequest(BaseModel):
    prompt: str = Field(..., description="What to build")
    language: Literal["html","css","js","c","cpp","python"] = "js"
    style: Literal["default","roman_urdu"] = "default"
    purpose: Optional[Literal["component","snippet","full_page","algorithm","game","cartoon","animation","web_app","mobile_app"]] = "snippet"
    level: Optional[Literal["beginner","intermediate","advanced"]] = "beginner"
    conversation_id: Optional[str] = None

class PixelMediaRequest(BaseModel):
    prompt: str
    kind: Literal["image","video"] = "image"
    conversation_id: Optional[str] = None

class PDFRequest(BaseModel):
    title: str
    content: str
    author: str = "Mehwish Chat AI"
    include_toc: bool = False
    language: str = "english"  # Added language parameter

class AudioRequest(BaseModel):
    text: str
    language: str = "english"
    conversation_id: Optional[str] = None

class ATMRequest(BaseModel):
    action: Literal["withdraw", "deposit", "balance", "statement"]
    amount: Optional[float] = 0.0
    account_number: str
    pin: str
    language: str = "english"  # Added language parameter

class BankingRequest(BaseModel):
    request_type: Literal["loan", "investment", "savings", "transaction_history"]
    details: dict
    language: str = "english"  # Added language parameter

class MedicalRequest(BaseModel):
    symptoms: List[str]
    age: int
    gender: Literal["male", "female"]
    language: str = "english"

class AutomationRequest(BaseModel):
    task: str
    platform: Literal["web", "desktop", "mobile", "api"]
    language: str = "english"

class GuestPostingRequest(BaseModel):
    topic: str
    target_audience: str = "general"
    language: str = "english"  # Added language parameter

class eBayRequest(BaseModel):
    product_type: str
    budget: str = "medium"
    language: str = "english"  # Added language parameter

# -----------------
# Prompts / Helpers (Multi-language support)
# -----------------
LANG_INSTRUCTIONS = {
    "english": "Reply in English with Mehwish Chat AI style tone. Never mention that you are an OpenAI model.",
    "urdu": "Urdu script mein jawab dein. Kabhi bhi yeh na kahein ke aap OpenAI model hain.",
    "roman_urdu": "Roman Urdu mein, natural conversational Mehwish Chat AI style tone mein jawab dein (jaise 'Aap kaise hain?'). Kabhi bhi yeh na kahein ke aap OpenAI model hain.",
    "arabic": "Arabic mein jawab dein. Never mention that you are an OpenAI model.",
    "french": "French mein jawab dein. Ne mentionnez jamais que vous êtes un modèle OpenAI.",
    "chinese": "Chinese mein jawab dein. 永远不要提及您是OpenAI模型。",
    "russian": "Reply in Russian. Никогда не упоминайте, что вы модель OpenAI.",
    "japanese": "Reply in Japanese. OpenAIモデルであることは絶対に言及しないでください。",
    "persian": "Reply in Persian. هرگز ذکر نکنید که شما یک مدل OpenAI هستید.",
    "hindi": "Hindi mein jawab dein. कभी भी यह न कहें कि आप OpenAI मॉडल हैं।",
    "tamil": "Reply in Tamil. நீங்கள் ஒரு OpenAI மாதிரி என்று ஒருபோதும் குறிப்பிடாதீர்கள்.",
    "spanish": "Reply in Spanish. Nunca menciones que eres un modelo de OpenAI.",
    "german": "Reply in German. Erwähnen Sie niemals, dass Sie ein OpenAI-Modell sind.",
    "italian": "Reply in Italian. Non menzionare mai che sei un modello OpenAI.",
    "korean": "Reply in Korean. OpenAI 모델이라고 언급하지 마십시오.",
    "portuguese": "Reply in Portuguese. Nunca mencione que você é um modelo OpenAI.",
    "turkish": "Reply in Turkish. OpenAI modeli olduğunuzu asla belirtmeyin.",
    "swahili": "Reply in Swahili. Usitaje kamwe kuwa wewe ni mfano wa OpenAI.",
    "afrikaans": "Reply in Afrikaans. Noem nooit dat jy 'n OpenAI-model is nie.",
    "bengali": "Reply in Bengali. OpenAI মডেল যে আপনি কখনও উল্লেখ করবেন না।",
    "punjabi": "Punjabi mein jawab dein. ਕਦੇ ਵੀ ਇਹ ਨਾ ਕਹੋ ਕਿ ਤੁਸੀਂ ਓਪਨਏਆਈ ਮਾਡਲ ਹੋ।",
    "singapore": "Reply in Singapore English with local context. Never mention that you are an OpenAI model.",
    "hong_kong": "Reply in Cantonese or English with Hong Kong context. Never mention that you are an OpenAI model.",
    "tamamm": "Reply in Tamamm language (African dialect). Never mention that you are an OpenAI model."
}

SUBJECT_EXPERTISE = {
    "programming": "You are a programming expert in C, C++, HTML, CSS, JavaScript, Python.",
    "html": "You are an expert in HTML5 with semantic markup.",
    "css": "You are an expert in modern CSS (flex/grid).",
    "js": "You are an expert in JavaScript (ES6+).",
    "c": "You are an expert in C, explain memory & pointers.",
    "cpp": "You are an expert in modern C++ (C++17+).",
    "python": "You are an expert in Python programming.",
    "banking": "You are a banking and finance expert with knowledge of Pakistani banking systems, ATM operations, loans, and investments.",
    "economics": "You are an economics expert with focus on global and Pakistani economy.",
    "civilization": "You are a history expert specializing in ancient and modern civilizations.",
    "science": "You are a science expert.",
    "math": "You are a mathematics expert.",
    "physics": "You are a physics expert.",
    "islamic": "You are an Islamic studies expert.",
    "seo": "You are an SEO and digital marketing expert.",
    "game": "You are a game development expert.",
    "cartoon": "You are a cartoon and animation expert.",
    "animation": "You are an animation and motion graphics expert.",
    "ebay": "You are an eBay selling expert focusing on Pakistani sellers and freelancers.",
    "freelancing": "You are a freelancing expert focusing on Pakistani freelancers and remote work opportunities.",
    "guestposting": "You are an expert in guest posting and content marketing.",
    "ai": "You are an artificial intelligence and machine learning expert.",
    "automation": "You are an automation and scripting expert.",
    "medical": "You are a medical AI assistant that can provide basic health information and symptom analysis.",
    "general": "You are Mehwish Chat AI, a helpful assistant created by Syed Kawish Ali from Pakistan (email: kawish.alisas@gmail.com). Never mention that you are an OpenAI model."
}

SPECIALIZED_HINTS = {
    "html": "Provide full HTML5 document when requested, otherwise snippet.",
    "css": "Return only CSS for injection.",
    "js": "Return only JS for injection.",
    "c": "Return compilable C code with comments.",
    "cpp": "Return compilable C++ code using STL where appropriate.",
    "python": "Return Python code with proper indentation and comments.",
    "banking": "Provide detailed financial analysis with Pakistani context and PKR currency.",
    "economics": "Provide economic analysis with focus on Pakistani economy and global impacts.",
    "civilization": "Provide historical context about civilizations with emphasis on South Asian history.",
    "science": "Provide scientific explanations with examples.",
    "math": "Provide step-by-step mathematical solutions.",
    "physics": "Explain physics concepts with real-world examples.",
    "islamic": "Provide Islamic knowledge with references.",
    "seo": "Provide practical SEO strategies with focus on Pakistani market.",
    "game": "Provide game development code and concepts.",
    "cartoon": "Provide animation and cartoon creation tips.",
    "animation": "Provide animation techniques and code examples.",
    "ebay": "Provide eBay selling tips specifically for Pakistani sellers, including pricing in PKR.",
    "freelancing": "Provide freelancing advice for Pakistanis, including platforms like Upwork, Fiverr, and local opportunities.",
    "guestposting": "Provide guest posting strategies with focus on building backlinks and authority.",
    "ai": "Provide AI and machine learning explanations with code examples.",
    "automation": "Provide automation scripts and workflows.",
    "medical": "Provide basic medical information and symptom analysis (disclaimer: not a substitute for professional medical advice).",
    "general": "Be friendly and helpful, explain concepts in simple terms. Mention that you are Mehwish Chat AI created by Syed Kawish Ali from Pakistan (email: kawish.alisas@gmail.com) when asked about your creator. Never mention that you are an OpenAI model."
}

CREATOR_RESPONSES = {
    "english": "I am Mehwish Chat AI, created by Syed Kawish Ali from Pakistan. Email: kawish.alisas@gmail.com",
    "urdu": "میں مہوش چیٹ AI ہوں، سید کاوش علی نے پاکستان میں بنائی ہے۔ ای میل: kawish.alisas@gmail.com",
    "roman_urdu": "Mein Mehwish Chat AI hoon, Syed Kawish Ali ne Pakistan mein banayi hai. Email: kawish.alisas@gmail.com",
    "hindi": "मैं Mehwish Chat AI हूं, सय्यद काविश अली ने पाकिस्तान में बनाई है। ईमेल: kawish.alisas@gmail.com",
    "french": "Je suis Mehwish Chat AI, créé par Syed Kawish Ali du Pakistan. Email : kawish.alisas@gmail.com",
    "chinese": "我是Mehwish Chat AI，由巴基斯坦的 Syed Kawish Ali 创建。电子邮件：kawish.alisas@gmail.com",
    "japanese": "私はMehwish Chat AIです。パキスタンのSyed Kawish Aliによって作成されました。メール：kawish.alisas@gmail.com",
    "russian": "Я Mehwish Chat AI, созданный Седом Кавишем Али из Пакистана. Эл. почта: kawish.alisas@gmail.com",
    "persian": "من Mehwish Chat AI هستم، توسط سید کاوش علی از پاکستان ساخته شده ام. ایمیل: kawish.alisas@gmail.com",
    "arabic": "أنا Mehwish Chat AI، تم إنشائي بواسطة سيد كاوش علي من باكستان. البريد الإلكتروني: kawish.alisas@gmail.com",
    "tamil": "நான் Mehwish Chat AI, பாகிஸ்தானைச் சேர்ந்த சயத் கவிஷ் அலி என்னை உருவாக்கினார். மின்னஞ்சல்: kawish.alisas@gmail.com",
    "spanish": "Soy Mehwish Chat AI, creado por Syed Kawish Ali de Pakistán. Email: kawish.alisas@gmail.com",
    "german": "Ich bin Mehwish Chat AI, erstellt von Syed Kawish Ali aus Pakistan. E-Mail: kawish.alisas@gmail.com",
    "italian": "Sono Mehwish Chat AI, creato da Syed Kawish Ali del Pakistan. Email: kawish.alisas@gmail.com",
    "korean": "저는 Mehwish Chat AI입니다. 파키스탄의 Syed Kawish Ali가 만들었습니다. 이메일: kawish.alisas@gmail.com",
    "portuguese": "Sou Mehwish Chat AI, criado por Syed Kawish Ali do Paquistão. Email: kawish.alisas@gmail.com",
    "turkish": "Ben Mehwish Chat AI, Pakistan'dan Syed Kawish Ali tarafından oluşturuldu. E-posta: kawish.alisas@gmail.com",
    "swahili": "Mimi ni Mehwish Chat AI, iliundwa na Syed Kawish Ali kutoka Pakistan. Barua pepe: kawish.alisas@gmail.com",
    "afrikaans": "Ek is Mehwish Chat AI, geskep deur Syed Kawish Ali van Pakistan. E-pos: kawish.alisas@gmail.com",
    "bengali": "আমি Mehwish Chat AI, পাকিস্তানের সৈয়দ কাওয়িশ আলি তৈরি করেছেন। ইমেইল: kawish.alisas@gmail.com",
    "punjabi": "ਮੈਂ Mehwish Chat AI ਹਾਂ, ਪਾਕਿਸਤਾਨ ਦੇ ਸਈਦ ਕਾਵਿਸ਼ ਅਲੀ ਨੇ ਬਣਾਇਆ ਹੈ। ਈਮੇਲ: kawish.alisas@gmail.com",
    "singapore": "I am Mehwish Chat AI, created by Syed Kawish Ali from Pakistan. Email: kawish.alisas@gmail.com",
    "hong_kong": "I am Mehwish Chat AI, created by Syed Kawish Ali from Pakistan. Email: kawish.alisas@gmail.com",
    "tamamm": "I am Mehwish Chat AI, created by Syed Kawish Ali from Pakistan. Email: kawish.alisas@gmail.com"
}

def format_code_blocks(text: str) -> str:
    lines = text.split("\n")
    in_code = False
    out = []
    for line in lines:
        if "```" in line:
            in_code = not in_code
            out.append(line)
        elif in_code:
            out.append(f"    {line}")
        else:
            out.append(line)
    return "\n".join(out)

async def call_groq_api(message: str, language: str = "english", subject: str = "general") -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not set")
        return "❌ Groq API key missing. Environment variable GROQ_API_KEY set karein."
    
    # Check if user is asking about creator in any language
    creator_keywords = [
        "who created you", "who made you", "creator", "developer", 
        "kis ne banaya", "banane wala", "créateur", "creador", "创建者",
        "創作者", "creator", "開発者", "создатель", "خالق", "निर्माता",
        "创建者", "創造者", "제작자", "criador", "yaratıcı", "mfanyi",
        "skepter", "স্রষ্টা", "ਨਿਰਮਾਤਾ", "创造者", "创作者"
    ]
    
    if any(keyword in message.lower() for keyword in creator_keywords):
        return CREATOR_RESPONSES.get(language, CREATOR_RESPONSES["english"])
    
    base_prompt = f"""
{SUBJECT_EXPERTISE.get(subject, SUBJECT_EXPERTISE['general'])}
{SPECIALIZED_HINTS.get(subject, '')}
{LANG_INSTRUCTIONS.get(language, 'Reply in English with Mehwish Chat AI style tone. Never mention that you are an OpenAI model.')}

User question:
{message}
"""
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": base_prompt}],
                "temperature": 0.7,
                "max_tokens": 2048
            }
            
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions", 
                headers=headers, 
                json=payload, 
                timeout=aiohttp.ClientTimeout(total=45)
            ) as resp:
                result = await resp.json()
                if resp.status == 200:
                    content = result["choices"][0]["message"]["content"]
                    return format_code_blocks(content) if subject in ["html","css","js","c","cpp","python","programming","game","animation","ai","automation"] else content
                else:
                    err = result.get("error", {}).get("message", "Unknown error")
                    logger.error(f"Groq error: {err}")
                    return f"⚠️ Groq API error: {err}"
                    
    except asyncio.TimeoutError:
        return "⚠️ Groq API timeout. Dobara koshish karein."
    except Exception as e:
        logger.exception("Groq call failed")
        return f"⚠️ Error: {str(e)}"

# -----------------
# PDF Generation Function
# -----------------
def generate_pdf(title: str, content: str, author: str = "Mehwish Chat AI", include_toc: bool = False) -> io.BytesIO:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, title)
    
    # Author and date
    c.setFont("Helvetica", 12)
    c.drawString(72, height - 92, f"By: {author}")
    c.drawString(72, height - 112, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Content
    y_position = height - 140
    c.setFont("Helvetica", 12)
    
    lines = content.split('\n')
    for line in lines:
        if y_position < 100:
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = height - 72
        
        # Handle code blocks
        if line.strip().startswith('```'):
            c.setFont("Helvetica-Oblique", 10)
            c.setFillColorRGB(0.5, 0.5, 0.5)
        else:
            c.setFont("Helvetica", 12)
            c.setFillColorRGB(0, 0, 0)
            
        c.drawString(72, y_position, line[:90])  # Limit line length
        y_position -= 20
    
    c.save()
    buffer.seek(0)
    return buffer

# -----------------
# HTML Injection Helpers
# -----------------
def backup_index():
    idx = TEMPLATES_DIR / "index.html"
    if idx.exists():
        bak = STATIC_DIR / "generated_media" / f"index_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        shutil.copy2(idx, bak)
        logger.info(f"Index.html backed up to {bak}")

def inject_into_index_html(code: str, language: str) -> str:
    index_file = TEMPLATES_DIR / "index.html"
    backup_index()

    if language == "html":
        if "<html" in code.lower():
            content = code
        else:
            content = ("<!doctype html>\n<html lang='en'>\n<head>\n<meta charset='utf-8'/>\n"
                       "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
                       "<title>Mehwish Chat AI - Generated</title>\n</head>\n<body>\n"
                       f"{code}\n</body>\n</html>")
        index_file.write_text(content, encoding="utf-8")
        return str(index_file)

    # read existing or create minimal
    if index_file.exists():
        original = index_file.read_text(encoding="utf-8")
    else:
        original = ("<!doctype html>\n<html lang='en'>\n<head>\n<meta charset='utf-8'/>\n"
                    "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
                    "<title>Mehwish Chat AI</title>\n</head>\n<body>\n<div id='app'></div>\n</body>\n</html>")

    if language == "css":
        if "<style id=\"mehwish-style\">" in original:
            new = re.sub(r"<style id=\"mehwish-style\">[\\s\\S]*?</style>", f"<style id=\"mehwish-style\">{code}</style>", original, flags=re.IGNORECASE)
        else:
            new = original.replace("</head>", f"<style id=\"mehwish-style\">{code}</style>\n</head>")
        index_file.write_text(new, encoding="utf-8")
        return str(index_file)

    if language == "js":
        if "<script id=\"mehwish-script\">" in original:
            new = re.sub(r"<script id=\"mehwish-script\">[\\s\\S]*?</script>", f"<script id=\"mehwish-script\">{code}</script>", original, flags=re.IGNORECASE)
        else:
            new = original.replace("</body>", f"<script id=\"mehwish-script\">{code}</script>\n</body>")
        index_file.write_text(new, encoding="utf-8")
        return str(index_file)

    # For c/cpp/python save to generated_media
    if language in ["c","cpp","python"]:
        ext = language
        target = STATIC_DIR / "generated_media" / f"program_{uuid.uuid4().hex[:8]}.{ext}"
        target.write_text(code, encoding="utf-8")
        return str(target)

    # default fallback
    target = STATIC_DIR / "generated_media" / f"code_{uuid.uuid4().hex[:8]}.txt"
    target.write_text(code, encoding="utf-8")
    return str(target)

# -----------------
# API Keys Check Endpoint
# -----------------
@app.get("/api/check-keys")
async def check_api_keys():
    """Check if API keys are loaded correctly"""
    groq_key = os.getenv("GROQ_API_KEY")
    pixels_key = os.getenv("PIXELS_API_KEY")
    
    return {
        "groq_loaded": bool(groq_key),
        "pixels_loaded": bool(pixels_key),
        "groq_length": len(groq_key) if groq_key else 0,
        "pixels_length": len(pixels_key) if pixels_key else 0,
        "message": "API keys status check"
    }

# -----------------
# ATM Simulation Functions
# -----------------
class ATMSystem:
    def __init__(self):
        self.accounts = {
            "1234567890": {"pin": "1234", "balance": 50000.0, "transactions": []},
            "0987654321": {"pin": "4321", "balance": 25000.0, "transactions": []},
            "1122334455": {"pin": "5566", "balance": 100000.0, "transactions": []}
        }
    
    def verify_account(self, account_number, pin):
        if account_number in self.accounts and self.accounts[account_number]["pin"] == pin:
            return True
        return False
    
    def get_balance(self, account_number):
        return self.accounts[account_number]["balance"]
    
    def withdraw(self, account_number, amount):
        if self.accounts[account_number]["balance"] >= amount:
            self.accounts[account_number]["balance"] -= amount
            transaction = {
                "type": "withdrawal",
                "amount": amount,
                "date": datetime.now().isoformat(),
                "balance_after": self.accounts[account_number]["balance"]
            }
            self.accounts[account_number]["transactions"].append(transaction)
            return True, transaction
        return False, None
    
    def deposit(self, account_number, amount):
        self.accounts[account_number]["balance"] += amount
        transaction = {
            "type": "deposit",
            "amount": amount,
            "date": datetime.now().isoformat(),
            "balance_after": self.accounts[account_number]["balance"]
        }
        self.accounts[account_number]["transactions"].append(transaction)
        return True, transaction
    
    def get_statement(self, account_number, count=10):
        transactions = self.accounts[account_number]["transactions"]
        return transactions[-count:] if len(transactions) > count else transactions

atm_system = ATMSystem()

# -----------------
# Medical AI Functions
# -----------------
class MedicalAI:
    def __init__(self):
        self.symptom_database = {
            "fever": ["Common cold", "Flu", "COVID-19", "Viral infection"],
            "cough": ["Common cold", "Flu", "COVID-19", "Bronchitis"],
            "headache": ["Migraine", "Tension headache", "Dehydration", "Sinusitis"],
            "chest pain": ["Heartburn", "Angina", "Anxiety", "Pneumonia"],
            "abdominal pain": ["Indigestion", "Appendicitis", "Food poisoning", "UTI"],
            "nausea": ["Food poisoning", "Migraine", "Pregnancy", "Gastroenteritis"],
            "dizziness": ["Dehydration", "Low blood pressure", "Inner ear problem", "Anemia"]
        }
        
        self.remedies = {
            "Common cold": ["Rest", "Hydration", "Vitamin C", "Over-the-counter cold medicine"],
            "Flu": ["Rest", "Hydration", "Antiviral medication", "Pain relievers"],
            "COVID-19": ["Isolation", "Rest", "Hydration", "Medical consultation"],
            "Viral infection": ["Rest", "Hydration", "Symptomatic treatment"],
            "Bronchitis": ["Rest", "Hydration", "Cough medicine", "Inhalers if prescribed"],
            "Migraine": ["Rest in dark room", "Pain relievers", "Hydration", "Avoid triggers"],
            "Tension headache": ["Rest", "Pain relievers", "Stress management", "Hydration"],
            "Dehydration": ["Drink fluids", "Electrolyte solutions", "Rest"],
            "Sinusitis": ["Decongestants", "Steam inhalation", "Pain relievers", "Hydration"],
            "Heartburn": ["Antacids", "Avoid spicy foods", "Smaller meals", "Elevate head while sleeping"],
            "Angina": ["Seek immediate medical attention", "Rest", "Prescribed medication"],
            "Anxiety": ["Deep breathing", "Relaxation techniques", "Therapy", "Medical consultation"],
            "Pneumonia": ["Medical consultation", "Antibiotics if bacterial", "Rest", "Hydration"],
            "Indigestion": ["Antacids", "Smaller meals", "Avoid fatty foods", "Reduce stress"],
            "Appendicitis": ["Seek immediate medical attention", "Do not eat or drink"],
            "Food poisoning": ["Hydration", "Rest", "Bland diet", "Medical attention if severe"],
            "UTI": ["Hydration", "Cranberry juice", "Medical consultation for antibiotics"],
            "Gastroenteritis": ["Hydration", "BRAT diet", "Rest", "Medical attention if severe"],
            "Pregnancy": ["Prenatal care", "Consult obstetrician", "Healthy diet", "Rest"],
            "Low blood pressure": ["Increase salt intake", "Hydration", "Small frequent meals", "Avoid sudden position changes"],
            "Inner ear problem": ["Balance exercises", "Medication for dizziness", "Avoid sudden movements"],
            "Anemia": ["Iron-rich foods", "Vitamin C to aid absorption", "Medical consultation", "Supplements if prescribed"]
        }
    
    def analyze_symptoms(self, symptoms, age, gender):
        possible_conditions = {}
        
        for symptom in symptoms:
            if symptom.lower() in self.symptom_database:
                for condition in self.symptom_database[symptom.lower()]:
                    if condition in possible_conditions:
                        possible_conditions[condition] += 1
                    else:
                        possible_conditions[condition] = 1
        
        # Sort by frequency
        sorted_conditions = sorted(possible_conditions.items(), key=lambda x: x[1], reverse=True)
        
        # Get top 3 most likely conditions
        top_conditions = sorted_conditions[:3]
        
        # Generate recommendations
        recommendations = []
        for condition, score in top_conditions:
            if condition in self.remedies:
                recommendations.append({
                    "condition": condition,
                    "confidence": f"{min(90, score * 20)}%",  # Simple confidence calculation
                    "advice": self.remedies[condition],
                    "disclaimer": "This is not a substitute for professional medical advice. Please consult a doctor for proper diagnosis."
                })
        
        return recommendations

medical_ai = MedicalAI()

# -----------------
# Endpoints
# -----------------
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    # cleanup older generated files >24h
    media_dir = STATIC_DIR / "generated_media"
    if media_dir.exists():
        now = datetime.now()
        for p in media_dir.iterdir():
            if p.is_file():
                age = now - datetime.fromtimestamp(p.stat().st_mtime)
                if age.total_seconds() > 24*3600:
                    try:
                        p.unlink()
                    except Exception:
                        pass
    return {"status":"✅ Running","version":"3.5.0","creator":"Syed Kawish Ali","timestamp":datetime.now().isoformat()}

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest, background_tasks: BackgroundTasks):
    try:
        cid = req.conversation_id or str(uuid.uuid4())
        if cid not in conversation_history:
            conversation_history[cid] = {"created_at": datetime.now().isoformat(),"messages":[]}
        
        conversation_history[cid]["messages"].append({
            "role": "user",
            "content": req.message,
            "timestamp": datetime.now().isoformat()
        })
        
        ai_resp = await call_groq_api(req.message, language=req.language, subject=req.subject)
        
        conversation_history[cid]["messages"].append({
            "role": "assistant",
            "content": ai_resp,
            "timestamp": datetime.now().isoformat()
        })
        
        if req.image or req.video:
            background_tasks.add_task(generate_media, req, cid, ai_resp)
            
        return {"response": ai_resp, "conversation_id": cid}
        
    except Exception as e:
        logger.exception("chat error")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_media(req: ChatRequest, conversation_id: str, ai_response: str):
    try:
        if req.image:
            img_url = await pixels_api.generate_image(f"{req.message} - {ai_response[:120]}")
            conversation_history[conversation_id]["image_url"] = img_url
            
        if req.video:
            vid_url = await pixels_api.generate_video(f"{req.message} - {ai_response[:120]}")
            conversation_history[conversation_id]["video_url"] = vid_url
            
    except Exception as e:
        logger.exception("media error")

@app.post("/api/code")
async def generate_code(req: CodeGenRequest):
    """
    Generate code via Groq and inject into templates/index.html (single-file).
    Use style='roman_urdu' for Roman Urdu explanations.
    """
    try:
        subject = req.language if req.language in ["html","css","js","c","cpp","python"] else "programming"
        tone = "Roman Urdu, friendly Mehwish Chat AI style." if req.style == "roman_urdu" else "Professional, concise."
        
        prompt = f"""
You are Mehwish Chat AI 3.5 (Creator: Syed Kawish Ali, Email: kawish.alisas@gmail.com).
Tone for explanations: {tone}
Subject: {subject}
Purpose: {req.purpose} | Level: {req.level}

User request:
{req.prompt}

Return code inside triple backticks when possible.
"""
        ai_response = await call_groq_api(
            prompt, 
            language=("roman_urdu" if req.style=="roman_urdu" else "english"), 
            subject=subject
        )
        
        # extract code block
        m = re.search(r"```(?:\w+)?\n([\s\\S]*?)```", ai_response)
        code = m.group(1).strip() if m else ai_response.strip()
        
        saved_path = inject_into_index_html(code, req.language)
        
        return {
            "message": "Generated and saved to index.html (or file)",
            "saved_to": saved_path, 
            "raw_response": ai_response
        }
        
    except Exception as e:
        logger.exception("code gen failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/media/pixel")
async def pixel_media(req: PixelMediaRequest):
    try:
        cid = req.conversation_id or str(uuid.uuid4())
        
        if cid not in conversation_history:
            conversation_history[cid] = {
                "created_at": datetime.now().isoformat(),
                "messages": []
            }
        
        if req.kind == "image":
            url = await pixels_api.generate_image(req.prompt)
            conversation_history[cid]["image_url"] = url
        else:
            url = await pixels_api.generate_video(req.prompt)
            conversation_history[cid]["video_url"] = url
            
        return {"conversation_id": cid, "url": url, "status": "success"}
        
    except Exception as e:
        logger.exception("pixel error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/seo-keywords")
async def generate_seo_keywords(req: SEORequest):
    """
    Generate SEO keywords in humanized style with focus on Pakistani market
    """
    try:
        prompt = f"""
Generate {req.max_keywords} SEO keywords for the following content, focusing on Pakistani audience.
Make them natural and human-friendly, not robotic. Include some Urdu/English mix keywords.

Content: {req.text}

Return the keywords as a comma-separated list.
"""
        keywords = await call_groq_api(prompt, language=req.language, subject="seo")
        return {"keywords": keywords, "count": req.max_keywords}
        
    except Exception as e:
        logger.exception("SEO keywords error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/guest-posting")
async def generate_guest_post_ideas(req: GuestPostingRequest):
    """
    Generate guest posting ideas and strategies
    """
    try:
        prompt = f"""
Generate guest posting ideas and strategies for Pakistani bloggers and content creators.
Topic: {req.topic}
Target audience: {req.target_audience}
Include topics that would appeal to both local and international audiences.
Provide practical tips for getting guest posts accepted.
"""
        ideas = await call_groq_api(prompt, language=req.language, subject="guestposting")
        return {"ideas": ideas}
        
    except Exception as e:
        logger.exception("Guest posting error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ebay-freelancing")
async def generate_ebay_freelancing_guide(req: eBayRequest):
    """
    Generate eBay freelancing guide for Pakistani users
    """
    try:
        prompt = f"""
Create a comprehensive guide for Pakistanis on how to start selling on eBay.
Product type: {req.product_type}
Budget: {req.budget}
Include step-by-step instructions, payment methods available in Pakistan, pricing strategies in PKR,
and tips for dealing with international customers. Make it easy to understand for beginners.
"""
        guide = await call_groq_api(prompt, language=req.language, subject="ebay")
        return {"guide": guide}
        
    except Exception as e:
        logger.exception("eBay guide error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-pdf")
async def generate_pdf_endpoint(req: PDFRequest):
    """
    Generate a PDF document from text content
    """
    try:
        # Generate content in the specified language
        prompt = f"""
Create a well-formatted document with the following title and content.
Title: {req.title}
Content: {req.content}

Make it professional and well-structured for PDF export.
"""
        content = await call_groq_api(prompt, language=req.language, subject="general")
        
        pdf_buffer = generate_pdf(req.title, content, req.author, req.include_toc)
        
        # Save to file
        filename = f"{req.title.replace(' ', '_')}_{uuid.uuid4().hex[:8]}.pdf"
        filepath = STATIC_DIR / "generated_media" / filename
        
        with open(filepath, "wb") as f:
            f.write(pdf_buffer.getvalue())
        
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type="application/pdf"
        )
        
    except Exception as e:
        logger.exception("PDF generation error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-audio")
async def generate_audio(req: AudioRequest):
    """
    Generate audio from text (placeholder - would integrate with an audio API)
    """
    try:
        # This is a placeholder - in a real implementation, you would integrate with an audio API
        # For now, we'll return a placeholder response
        return {
            "message": "Audio generation would be implemented here with a proper API",
            "text": req.text,
            "language": req.language,
            "status": "Audio API not integrated yet"
        }
        
    except Exception as e:
        logger.exception("Audio generation error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/atm-operation")
async def atm_operation(req: ATMRequest):
    """
    Simulate ATM operations: withdraw, deposit, check balance, get statement
    """
    try:
        if not atm_system.verify_account(req.account_number, req.pin):
            raise HTTPException(status_code=401, detail="Invalid account number or PIN")
        
        if req.action == "balance":
            balance = atm_system.get_balance(req.account_number)
            message = f"Your current balance is: PKR {balance:,.2f}"
            
            # Translate message if needed
            if req.language != "english":
                translation_prompt = f"Translate the following banking message to {req.language}: {message}"
                translated = await call_groq_api(translation_prompt, language=req.language, subject="banking")
                message = translated
            
            return {
                "action": "balance",
                "account_number": req.account_number,
                "balance": balance,
                "currency": "PKR",
                "message": message
            }
        
        elif req.action == "withdraw":
            if req.amount <= 0:
                raise HTTPException(status_code=400, detail="Invalid amount")
            
            success, transaction = atm_system.withdraw(req.account_number, req.amount)
            if success:
                message = f"Successfully withdrew PKR {req.amount:,.2f}. New balance: PKR {transaction['balance_after']:,.2f}"
                
                # Translate message if needed
                if req.language != "english":
                    translation_prompt = f"Translate the following banking message to {req.language}: {message}"
                    translated = await call_groq_api(translation_prompt, language=req.language, subject="banking")
                    message = translated
                
                return {
                    "action": "withdraw",
                    "account_number": req.account_number,
                    "amount": req.amount,
                    "new_balance": transaction["balance_after"],
                    "currency": "PKR",
                    "message": message,
                    "transaction": transaction
                }
            else:
                error_msg = "Insufficient funds"
                if req.language != "english":
                    translation_prompt = f"Translate the following error message to {req.language}: {error_msg}"
                    translated = await call_groq_api(translation_prompt, language=req.language, subject="banking")
                    error_msg = translated
                raise HTTPException(status_code=400, detail=error_msg)
        
        elif req.action == "deposit":
            if req.amount <= 0:
                raise HTTPException(status_code=400, detail="Invalid amount")
            
            success, transaction = atm_system.deposit(req.account_number, req.amount)
            if success:
                message = f"Successfully deposited PKR {req.amount:,.2f}. New balance: PKR {transaction['balance_after']:,.2f}"
                
                # Translate message if needed
                if req.language != "english":
                    translation_prompt = f"Translate the following banking message to {req.language}: {message}"
                    translated = await call_groq_api(translation_prompt, language=req.language, subject="banking")
                    message = translated
                
                return {
                    "action": "deposit",
                    "account_number": req.account_number,
                    "amount": req.amount,
                    "new_balance": transaction["balance_after"],
                    "currency": "PKR",
                    "message": message,
                    "transaction": transaction
                }
        
        elif req.action == "statement":
            statement = atm_system.get_statement(req.account_number)
            message = f"Retrieved {len(statement)} recent transactions"
            
            # Translate message if needed
            if req.language != "english":
                translation_prompt = f"Translate the following banking message to {req.language}: {message}"
                translated = await call_groq_api(translation_prompt, language=req.language, subject="banking")
                message = translated
            
            return {
                "action": "statement",
                "account_number": req.account_number,
                "transactions": statement,
                "message": message
            }
        
    except Exception as e:
        logger.exception("ATM operation error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/banking-advice")
async def banking_advice(req: BankingRequest):
    """
    Provide banking advice for loans, investments, savings, etc.
    """
    try:
        if req.request_type == "loan":
            prompt = f"""
Provide comprehensive loan advice for a Pakistani customer.
Consider these details: {json.dumps(req.details)}
Include information about:
1. Types of loans available in Pakistan
2. Interest rates and terms
3. Documentation required
4. Eligibility criteria
5. Tips for getting approved
6. Islamic banking options if applicable
"""
        elif req.request_type == "investment":
            prompt = f"""
Provide comprehensive investment advice for a Pakistani customer.
Consider these details: {json.dumps(req.details)}
Include information about:
1. Investment options available in Pakistan
2. Risk levels and returns
3. Tax implications
4. Long-term vs short-term strategies
5. Sharia-compliant investment options
"""
        elif req.request_type == "savings":
            prompt = f"""
Provide comprehensive savings advice for a Pakistani customer.
Consider these details: {json.dumps(req.details)}
Include information about:
1. Savings account options in Pakistani banks
2. Interest rates (profit rates for Islamic banks)
3. Minimum balance requirements
4. Digital savings options
5. Tips for building savings habit
"""
        elif req.request_type == "transaction_history":
            prompt = f"""
Provide analysis and advice based on transaction history for a Pakistani customer.
Consider these details: {json.dumps(req.details)}
Include:
1. Spending pattern analysis
2. Budgeting recommendations
3. Saving opportunities
4. Financial health assessment
5. Recommendations for improvement
"""
        else:
            raise HTTPException(status_code=400, detail="Invalid request type")
        
        advice = await call_groq_api(prompt, language=req.language, subject="banking")
        return {"advice": advice, "request_type": req.request_type}
        
    except Exception as e:
        logger.exception("Banking advice error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/medical-advice")
async def medical_advice(req: MedicalRequest):
    """
    Provide basic medical advice based on symptoms (not a substitute for professional medical advice)
    """
    try:
        analysis = medical_ai.analyze_symptoms(req.symptoms, req.age, req.gender)
        
        # Generate a prompt for Groq to provide additional advice
        prompt = f"""
Provide basic medical advice for a {req.age} year old {req.gender} experiencing these symptoms: {', '.join(req.symptoms)}.
Please include:
1. Possible conditions (but emphasize this is not a diagnosis)
2. General self-care tips
3. When to seek professional medical help
4. Home remedies if appropriate
5. Important disclaimer that this is not medical advice

Be compassionate and clear in your response.
"""
        additional_advice = await call_groq_api(prompt, language=req.language, subject="medical")
        
        return {
            "symptom_analysis": analysis,
            "additional_advice": additional_advice,
            "disclaimer": "This is not a substitute for professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment."
        }
        
    except Exception as e:
        logger.exception("Medical advice error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/automation-script")
async def generate_automation_script(req: AutomationRequest):
    """
    Generate automation scripts for various platforms
    """
    try:
        prompt = f"""
Create an automation script for: {req.task}
Platform: {req.platform}
Provide a complete, working script with explanations.

The script should be:
1. Efficient and well-commented
2. Include error handling
3. Include setup instructions
4. Include any necessary dependencies

Provide the code in appropriate language for the platform.
"""
        script = await call_groq_api(prompt, language=req.language, subject="automation")
        
        # Extract code if it's in a code block
        code_match = re.search(r"```(?:\w+)?\n([\s\\S]*?)```", script)
        if code_match:
            code = code_match.group(1).strip()
            explanation = re.sub(r"```[\s\\S]*?```", "", script).strip()
        else:
            code = script
            explanation = ""
        
        return {
            "script": code,
            "explanation": explanation,
            "platform": req.platform,
            "task": req.task
        }
        
    except Exception as e:
        logger.exception("Automation script error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export-conversation/{conversation_id}")
async def export_conversation(conversation_id: str):
    if conversation_id not in conversation_history:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    export = conversation_history[conversation_id].copy()
    export["id"] = conversation_id
    
    fname = f"conversation_{conversation_id}.json"
    fpath = STATIC_DIR / "generated_media" / fname
    
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    
    return FileResponse(path=fpath, filename=fname, media_type="application/json")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found", "detail": exc.detail}
    )

@app.exception_handler(500)
async def server_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "detail": exc.detail}
    )

if __name__ == "__main__":
    # Check if API keys are loaded
    groq_key = os.getenv("GROQ_API_KEY")
    pixels_key = os.getenv("PIXELS_API_KEY")
    
    print("=== Mehwish Chat AI 3.5 ===")
    print("World's Best AI Assistant")
    print("Created by: Syed Kawish Ali")
    print("Email: kawish.alisas@gmail.com")
    print("===========================")
    
    if groq_key:
        print(f"✅ GROQ_API_KEY: Loaded successfully ({len(groq_key)} characters)")
    else:
        print("❌ GROQ_API_KEY: Not found in environment")
        
    if pixels_key:
        print(f"✅ PIXELS_API_KEY: Loaded successfully ({len(pixels_key)} characters)")
    else:
        print("❌ PIXELS_API_KEY: Not found in environment")
    
    if not groq_key and not pixels_key:
        print("\n⚠️  Warning: No API keys found. Some features will be limited.")
        print("ℹ️  Please set GROQ_API_KEY and PIXELS_API_KEY in your .env file")
    
    print("\n🚀 Starting server on http://localhost:8000")
    print("📚 API documentation available at http://localhost:8000/api/docs")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")