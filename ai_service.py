import os
import base64
import json
import urllib.request
import urllib.error
import time
# Gradio import removed


# Function to load environment variables from .env file
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip().strip('"').strip("'")
                        os.environ[key] = val

# Load .env
load_env()

DEFAULT_API_KEY = os.getenv("DEFAULT_API_KEY", "")

def transcribe(audio_path, api_key, model_name):
    with open("debug_entry.txt", "w", encoding="utf-8") as f:
        f.write(f"Entered transcribe with audio_path: {audio_path}")
        
    if not audio_path:
        return "Please provide an audio recording.", None
    if not api_key:
        return "Please provide a valid API Key.", None

    try:
        # Read the audio file and encode as base64
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        
        base64_audio = base64.b64encode(audio_data).decode("utf-8")
        
        # Determine mime type accurately
        mime_type = "audio/wav"
        lower_path = audio_path.lower()
        if lower_path.endswith(".mp3"):
            mime_type = "audio/mp3"
        elif lower_path.endswith(".ogg"):
            mime_type = "audio/ogg"
        elif lower_path.endswith(".flac"):
            mime_type = "audio/flac"
        elif lower_path.endswith((".m4a", ".aac")):
            mime_type = "audio/aac"
        elif lower_path.endswith(".webm"):
            mime_type = "audio/webm"
        elif lower_path.endswith(".mp4"):
            mime_type = "audio/mp4"

        # Prepare the  API request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": api_key.strip()
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": "Transcribe the following Tamil speech. Only output the exact Tamil text that is spoken."
                        },
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": base64_audio
                            }
                        }
                    ]
                }
            ]
        }
        
        # Send request
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    
                    # Parse response safely
                    if "candidates" in result and len(result["candidates"]) > 0:
                        candidate = result["candidates"][0]
                        finish_reason = candidate.get("finishReason")
                        
                        if finish_reason and finish_reason != "STOP":
                            return f"API Error: Transcription blocked/failed (Finish Reason: {finish_reason}). Raw response: {json.dumps(result)}", None
                        
                        content = candidate.get("content")
                        parts = content.get("parts") if content else None
                        
                        if not parts or len(parts) == 0:
                            # If the model finished with STOP but returned no parts, it means no text was generated (e.g. silence)
                            text = ""
                        else:
                            text = parts[0].get("text", "").strip()
                        
                        # Save to text file for download
                        file_path = "transcription_result.txt"
                        try:
                            with open(file_path, "w", encoding="utf-8") as text_file:
                                text_file.write(text)
                        except Exception as e:
                            print(f"Error saving file: {e}")
                            
                        return text, file_path
                    else:
                        return "API Error: No candidates returned. Response: " + json.dumps(result), None
            except urllib.error.HTTPError as e:
                try:
                    error_msg = e.read().decode("utf-8", errors="replace")
                except:
                    error_msg = str(e)
                
                # Parse JSON error message if possible to show a clean message
                clean_msg = error_msg
                try:
                    err_data = json.loads(error_msg)
                    if "error" in err_data and "message" in err_data["error"]:
                        clean_msg = err_data["error"]["message"]
                except:
                    pass

                if e.code == 503 and attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                if e.code == 403:
                    return "Authentication Error: The provided API Key is invalid or has been disabled. Please enter a valid API Key.", None
                if e.code == 429:
                    return "API Rate Limit Exceeded: You have exceeded the daily request limit for this model. Please wait a bit or try again later.", None
                
                return f"API Error ({e.code}): {clean_msg}", None
            
    except Exception as e:
        import traceback
        err_str = traceback.format_exc()
        print(f"DEBUG TRANSCRIPTION ERROR: {err_str}")
        try:
            with open("error_log.txt", "w", encoding="utf-8") as f:
                f.write(err_str)
        except Exception as write_err:
            print(f"Failed to write error log: {write_err}")
        
        # Raise standard exception to be caught in app.py
        raise Exception(f"Backend Error: {str(e)}")
