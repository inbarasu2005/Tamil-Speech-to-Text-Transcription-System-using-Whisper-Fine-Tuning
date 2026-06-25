import urllib.request
import urllib.parse
import urllib.error
import http.cookiejar
import wave
import os
import secrets

BASE_URL = "http://127.0.0.1:5000"

def create_silent_wav(filename):
    """Generates a simple 1-second silent WAV file."""
    with wave.open(filename, 'wb') as wav:
        # Mono, 2 bytes per sample, 16000Hz samplerate
        wav.setparams((1, 2, 16000, 16000, 'NONE', 'not compressed'))
        # Write 1 second of silence
        wav.writeframes(b'\x00' * 32000)

def test_transcribe():
    print("[TEST] Verifying /transcribe endpoint...")
    
    # 1. Setup session cookies
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    urllib.request.install_opener(opener)
    
    # 2. Register and log in to get session cookie
    email = f"transcribetester_{secrets.token_hex(4)}@example.com"
    print(f"   Using account: {email}")
    
    # Register
    reg_data = urllib.parse.urlencode({
        'fullname': 'Transcribe Tester',
        'email': email,
        'password': 'password123',
        'confirm_password': 'password123'
    }).encode('utf-8')
    urllib.request.urlopen(urllib.request.Request(BASE_URL + "/register", data=reg_data, method="POST"))
    
    # Login
    login_data = urllib.parse.urlencode({
        'email': email,
        'password': 'password123'
    }).encode('utf-8')
    urllib.request.urlopen(urllib.request.Request(BASE_URL + "/login", data=login_data, method="POST"))
    
    # 3. Create a temporary audio file
    temp_audio = "temp_test_audio.wav"
    create_silent_wav(temp_audio)
    
    try:
        # 4. Construct multipart form data for file upload
        boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
        
        # Read WAV bytes
        with open(temp_audio, 'rb') as f:
            file_bytes = f.read()
            
        # Build multipart payload
        parts = []
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="{temp_audio}"\r\nContent-Type: audio/wav\r\n\r\n'.encode('utf-8'))
        parts.append(file_bytes)
        parts.append(f'\r\n--{boundary}--\r\n'.encode('utf-8'))
        
        payload = b''.join(parts)
        
        # Send POST request
        req = urllib.request.Request(
            BASE_URL + "/transcribe",
            data=payload,
            method="POST"
        )
        req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
        
        print("   Sending transcription request...")
        response = urllib.request.urlopen(req)
        result = json = urllib.request.urlopen(req).read().decode('utf-8')
        print(f"   Response from server: {result}")
        print("   [PASS] Endpoint responded successfully.")
        
    except Exception as e:
        print(f"   [FAIL] Transcription endpoint request failed: {e}")
        if hasattr(e, 'read'):
            try:
                print(f"   Server response: {e.read().decode('utf-8')}")
            except:
                pass
    finally:
        # Cleanup
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

if __name__ == "__main__":
    test_transcribe()
