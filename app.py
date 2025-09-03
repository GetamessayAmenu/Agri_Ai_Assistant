import os
import random
import string
import requests
import openai
from gtts import gTTS
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from openai.error import AuthenticationError, OpenAIError

# create app first so we can use app.logger safely
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'webm'}

# ensure necessary folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/audio', exist_ok=True)

load_dotenv()  # load .env into environment if present

# read API keys from env (do NOT hardcode!)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")  # optional admin token to protect runtime key changes

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    app.logger.warning("OPENAI_API_KEY not set. OpenAI requests will fail until it's provided.")

if not HUGGING_FACE_API_KEY:
    app.logger.warning("HUGGING_FACE_API_KEY not set. Audio transcription may fail.")

# Provide an optional admin endpoint to set the OpenAI key at runtime (protected by ADMIN_TOKEN if set).
# Note: storing keys via this endpoint writes to .env for convenience — do NOT enable in production without proper protection.
@app.route("/admin/set_openai_key", methods=["POST"])
def admin_set_openai_key():
    # check admin token if configured
    if ADMIN_TOKEN:
        token = request.headers.get("X-Admin-Token") or request.form.get("token") or (request.get_json(silent=True) or {}).get("token")
        if token != ADMIN_TOKEN:
            return jsonify({"error": "unauthorized"}), 403

    payload = request.form or (request.get_json(silent=True) or {})
    key = (payload.get("key") or payload.get("OPENAI_API_KEY") or "").strip()
    if not key:
        return jsonify({"error": "no key provided"}), 400

    # set for current process
    openai.api_key = key
    os.environ["OPENAI_API_KEY"] = key

    # append to .env for convenience (if writable) — avoid duplicate entries
    try:
        env_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            # replace existing line if present
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            found = False
            for i, line in enumerate(lines):
                if line.strip().startswith("OPENAI_API_KEY="):
                    lines[i] = f"OPENAI_API_KEY={key}\n"
                    found = True
                    break
            if not found:
                lines.append(f"\nOPENAI_API_KEY={key}\n")
            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        else:
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(f"OPENAI_API_KEY={key}\n")
        app.logger.info("OPENAI_API_KEY set via admin endpoint and saved to .env")
    except Exception as e:
        app.logger.exception("Failed to write .env: %s", e)
        # still return success because runtime key was set
        return jsonify({"status": "ok", "warning": "runtime key set but .env write failed"}), 200

    return jsonify({"status": "ok"}), 200

# optional endpoint to clear runtime key (protected by ADMIN_TOKEN if set)
@app.route("/admin/clear_openai_key", methods=["POST"])
def admin_clear_openai_key():
    if ADMIN_TOKEN:
        token = request.headers.get("X-Admin-Token") or request.form.get("token") or (request.get_json(silent=True) or {}).get("token")
        if token != ADMIN_TOKEN:
            return jsonify({"error": "unauthorized"}), 403

    openai.api_key = None
    os.environ.pop("OPENAI_API_KEY", None)
    app.logger.info("OPENAI_API_KEY cleared from runtime environment")
    return jsonify({"status": "cleared"}), 200

def get_anwer_openai(question):
    """
    Call OpenAI ChatCompletion and return text or a friendly error string.
    """
    if not getattr(openai, "api_key", None):
        app.logger.error("OpenAI API key missing when calling get_anwer_openai")
        return "OpenAI API key missing on server. Set OPENAI_API_KEY and restart the app."

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "I want you to act like a helpful agriculture chatbot and help farmers with their query"},
                {"role": "user", "content": question}
            ],
            timeout=30
        )
        return completion["choices"][0]["message"]["content"].strip()
    except AuthenticationError as e:
        app.logger.error("OpenAI AuthenticationError: %s", e)
        return "OpenAI authentication failed. Check OPENAI_API_KEY on the server."
    except OpenAIError as e:
        app.logger.error("OpenAI API error: %s", e)
        return "OpenAI API error. See server logs."
    except Exception as e:
        app.logger.exception("Unexpected error calling OpenAI: %s", e)
        return "Unexpected server error while contacting OpenAI."

def text_to_audio(text, filename):
    """
    Convert text to MP3 and save under static/audio/<filename>.mp3
    """
    try:
        tts = gTTS(text)
        out_path = f'static/audio/{filename}.mp3'
        tts.save(out_path)
        return out_path
    except Exception as e:
        app.logger.exception("text_to_audio failed: %s", e)
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # audio upload handling
        if 'audio' in request.files:
            f = request.files['audio']
            if f and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                f.save(filepath)
                transcription = process_audio(filepath)
                return jsonify({'text': transcription})

        text = request.form.get('text') or (request.get_json(silent=True) or {}).get('text')
        if text:
            response = process_text(text)  # returns dict {'text','voice'}
            return jsonify(response)

        return jsonify({'text': 'Invalid request'}), 400
    except Exception as e:
        app.logger.exception("Unhandled /chat error: %s", e)
        # return JSON error instead of Werkzeug HTML traceback
        return jsonify({'text': 'Internal server error'}), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_audio(filepath):
    # simple Hugging Face inference POST (expects HUGGING_FACE_API_KEY)
    if not HUGGING_FACE_API_KEY:
        app.logger.error("No HUGGING_FACE_API_KEY; cannot transcribe audio.")
        return "Transcription service not configured."

    API_URL = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    with open(filepath, "rb") as f:
        data = f.read()
    try:
        response = requests.post(API_URL, headers=headers, data=data, timeout=60)
        response.raise_for_status()
        data = response.json()
        # guard against unexpected response shape
        text = data.get('text') if isinstance(data, dict) else None
        if not text:
            # some HF models return different structure
            app.logger.error("Unexpected transcription response: %s", data)
            return "Could not transcribe audio."
        return text
    except Exception as e:
        app.logger.exception("Audio transcription failed: %s", e)
        return "Audio transcription failed."

def process_text(text):
    """
    Process incoming text, call OpenAI, synthesize audio, and return structure expected by client.
    """
    try:
        return_text = get_anwer_openai(text)
        # ensure return_text is a string
        if not isinstance(return_text, str):
            return_text = str(return_text)

        # generate voice file name and produce audio if desired (text_to_audio should handle errors)
        res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        voice_file = ""
        try:
            out_path = text_to_audio(return_text, res)
            # url_for should reference the static path including extension
            voice_file = url_for('static', filename='audio/' + res + '.mp3')
        except Exception as e:
            app.logger.exception("text_to_audio failed: %s", e)
            voice_file = ""

        return {'text': return_text, 'voice': voice_file}
    except Exception as e:
        app.logger.exception("process_text failed: %s", e)
        return {'text': 'Server error processing request. Check server logs.', 'voice': ''}

if __name__ == '__main__':
    app.run(debug=True)

