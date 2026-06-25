import os
import secrets
from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from werkzeug.security import generate_password_hash, check_password_hash
from jinja2 import pass_context

from db import init_db, get_db_connection
from ai_service import transcribe, DEFAULT_API_KEY
from fastapi.staticfiles import StaticFiles

# Initialize FastAPI App
app = FastAPI(title="Tamil Speech-to-Text System")

# Configure Session Middleware
app.add_middleware(
    SessionMiddleware, 
    secret_key=os.getenv("SECRET_KEY", "dev_secret_key_for_tamil_speech_to_text")
)

# Configure Static Files and Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Custom Context Processor for Flash Messages (equivalent to get_flashed_messages)
@pass_context
def get_flashed_messages(context: dict, with_categories: bool = False):
    request = context.get("request")
    if not request:
        return []
    flashes = request.session.pop("_flashes", [])
    if with_categories:
        return flashes
    return [msg for cat, msg in flashes]

# Expose get_flashed_messages to templates env globals
templates.env.globals["get_flashed_messages"] = get_flashed_messages

def flash(request: Request, message: str, category: str = "info"):
    if "_flashes" not in request.session:
        request.session["_flashes"] = []
    request.session["_flashes"].append((category, message))

# Startup Event to initialize database
@app.on_event("startup")
def on_startup():
    success, msg = init_db()
    if not success:
        print(f"CRITICAL WARNING: Database initialization failed: {msg}")

# ----------------------------------------------------
# Web App Routes
# ----------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serves the transcription dashboard."""
    if "user_id" not in request.session:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html", 
        context={"default_api_key": DEFAULT_API_KEY}
    )

@app.get("/register", response_class=HTMLResponse, name="register")
async def register_get(request: Request):
    """Serves registration page."""
    if "user_id" in request.session:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(request=request, name="register.html")

@app.post("/register", name="register")
async def register_post(
    request: Request,
    fullname: str = Form(""),
    email: str = Form(""),
    password: str = Form(""),
    confirm_password: str = Form("")
):
    """Handles user registration."""
    if "user_id" in request.session:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    fullname = fullname.strip()
    email = email.strip()

    if not fullname or not email or not password or not confirm_password:
        flash(request, "All fields are required.", "error")
        return templates.TemplateResponse(request=request, name="register.html")

    if password != confirm_password:
        flash(request, "Passwords do not match.", "error")
        return templates.TemplateResponse(request=request, name="register.html")

    if len(password) < 6:
        flash(request, "Password must be at least 6 characters long.", "error")
        return templates.TemplateResponse(request=request, name="register.html")

    hashed_password = generate_password_hash(password)

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Check if user already exists
        cur.execute("SELECT id FROM users WHERE email = %s;", (email,))
        if cur.fetchone():
            flash(request, "An account with that email already exists.", "error")
            cur.close()
            return templates.TemplateResponse(request=request, name="register.html")

        # Insert user into database
        cur.execute(
            "INSERT INTO users (fullname, email, password) VALUES (%s, %s, %s);",
            (fullname, email, hashed_password)
        )
        conn.commit()
        cur.close()

        flash(request, "Registration successful! Please log in.", "success")
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    except Exception as e:
        flash(request, f"An error occurred during registration: {str(e)}", "error")
        return templates.TemplateResponse(request=request, name="register.html")
    finally:
        if conn:
            conn.close()

@app.get("/login", response_class=HTMLResponse, name="login")
async def login_get(request: Request):
    """Serves login page."""
    if "user_id" in request.session:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(request=request, name="login.html")

@app.post("/login", name="login")
async def login_post(
    request: Request,
    email: str = Form(""),
    password: str = Form(""),
    remember: str = Form("false")
):
    """Handles user login."""
    if "user_id" in request.session:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    email = email.strip()

    if not email or not password:
        flash(request, "Please enter both email and password.", "error")
        return templates.TemplateResponse(request=request, name="login.html")

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Query database for user
        cur.execute("SELECT id, fullname, email, password FROM users WHERE email = %s;", (email,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user['password'], password):
            request.session.clear()
            request.session['user_id'] = user['id']
            request.session['fullname'] = user['fullname']
            request.session['email'] = user['email']
            
            flash(request, f"Welcome back, {user['fullname']}!", "success")
            return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        else:
            flash(request, "Invalid email or password.", "error")
            return templates.TemplateResponse(request=request, name="login.html")

    except Exception as e:
        flash(request, f"An error occurred during login: {str(e)}", "error")
        return templates.TemplateResponse(request=request, name="login.html")
    finally:
        if conn:
            conn.close()

@app.get("/logout")
async def logout(request: Request):
    """Logs out the user and destroys the session."""
    request.session.clear()
    flash(request, "You have been logged out successfully.", "success")
    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

@app.post("/transcribe")
async def process_transcription(
    request: Request,
    audio: UploadFile = File(...),
    api_key: str = Form(DEFAULT_API_KEY),
    model_name: str = Form("gemini-2.5-flash")
):
    """Receives audio file, runs transcription, and returns result."""
    if "user_id" not in request.session:
        return JSONResponse({"error": "Unauthorized. Please log in."}, status_code=status.HTTP_401_UNAUTHORIZED)

    if not audio.filename:
        return JSONResponse({"error": "No audio file selected."}, status_code=status.HTTP_400_BAD_REQUEST)

    # Save to temporary path inside workspace directory
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    ext = os.path.splitext(audio.filename)[1]
    if not ext:
        ext = '.webm'
        
    temp_filename = f"temp_upload_{secrets.token_hex(8)}{ext}"
    temp_path = os.path.join(temp_dir, temp_filename)

    try:
        # Save upload contents to temp file
        with open(temp_path, "wb") as buffer:
            contents = await audio.read()
            buffer.write(contents)
        
        # Transcribe audio using the api service
        text, text_file = transcribe(temp_path, api_key, model_name)
        
        if "Authentication Error" in text or "API Error" in text:
            return JSONResponse({"error": text}, status_code=status.HTTP_400_BAD_REQUEST)
            
        return JSONResponse({
            "transcription": text,
            "success": True
        })

    except Exception as e:
        return JSONResponse({"error": f"Transcription engine error: {str(e)}"}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
    finally:
        # Clean up temporary audio file if exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_err:
                print(f"Error removing temp file: {cleanup_err}")

if __name__ == "__main__":
    import uvicorn
    print("--- STARTING FASTAPI AUTHENTICATION APP ON PORT 5000 ---")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
