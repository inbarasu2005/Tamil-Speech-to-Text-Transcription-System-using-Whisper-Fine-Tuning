import os
import base64
import json
import urllib.request
import urllib.error
import time
import gradio as gr

# Setup API Key from user provided curl
DEFAULT_API_KEY = "AIzaSyB7pEluRhpKZLxf6TxZd7VSOfljKrYsA-s"

def transcribe(audio_path, api_key, model_name):
    if not audio_path:
        return "Please provide an audio recording."
    if not api_key:
        return "Please provide a valid API Key."

    try:
        # Read the audio file and encode as base64
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        
        base64_audio = base64.b64encode(audio_data).decode("utf-8")
        
        # Determine mime type roughly
        mime_type = "audio/wav"
        if audio_path.lower().endswith(".mp3"):
            mime_type = "audio/mp3"
        elif audio_path.lower().endswith(".ogg"):
            mime_type = "audio/ogg"
        elif audio_path.lower().endswith(".flac"):
            mime_type = "audio/flac"

        # Prepare the Gemini API request
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
                    
                    # Parse response
                    if "candidates" in result and len(result["candidates"]) > 0:
                        text = result["candidates"][0]["content"]["parts"][0]["text"]
                        return text.strip()
                    else:
                        return "No transcription returned. Response: " + str(result)
            except urllib.error.HTTPError as e:
                error_msg = e.read().decode("utf-8")
                if e.code == 503 and attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                return f"API Error {e.code}:\n{error_msg}"
            
    except Exception as e:
        import traceback
        with open("last_ui_error.txt", "w") as f:
            f.write(traceback.format_exc())
        return f"⚠️ Error: {str(e)}"

# Define the Gradio interface
with gr.Blocks(title="Tamil Speech-to-Text with Gemini", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Tamil Speech-to-Text Transcription System")
    gr.Markdown("Record your voice or upload an audio file to convert Tamil speech into text quickly and accurately.")
    
    with gr.Row():
        with gr.Column(scale=1):
            api_key_input = gr.Textbox(
                label="Gemini API Key", 
                type="password", 
                value=DEFAULT_API_KEY,
                visible=False
            )
            model_input = gr.Textbox(
                label="Model Name", 
                value="gemini-flash-latest",
                visible=False
            )
            
            with gr.Tabs():
                with gr.Tab("🎤 Record"):
                    audio_mic = gr.Audio(sources=["microphone"], type="filepath", label="Mic Input")
                    btn_mic = gr.Button("Transcribe Recording", variant="primary")
                with gr.Tab("📁 Upload"):
                    audio_upload = gr.Audio(sources=["upload"], type="filepath", label="File Input")
                    btn_upload = gr.Button("Transcribe File", variant="primary")
                    
        with gr.Column(scale=1):
            text_output = gr.Textbox(label="Transcription Result", lines=12, placeholder="Transcription will appear here...")
            
    # Map both buttons to the same transcribe function and output
    btn_mic.click(fn=transcribe, inputs=[audio_mic, api_key_input, model_input], outputs=text_output)
    btn_upload.click(fn=transcribe, inputs=[audio_upload, api_key_input, model_input], outputs=text_output)
def get_free_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

if __name__ == "__main__":
    free_port = get_free_port()
    print(f"--- DEBUG: Starting Gradio on port {free_port} ---")
    demo.launch(
        server_name="127.0.0.1", 
        server_port=free_port,
        share=False,
        inbrowser=True,
        show_api=False
    )
