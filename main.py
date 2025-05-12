import whisper
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deep_translator import GoogleTranslator
import uuid
import os

app = FastAPI()

# Abilita CORS per poter chiamare l'API da Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o specifica "http://localhost:3000" ecc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carica il modello Whisper una volta sola
model = whisper.load_model("base")

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    source_lang: str = Form(...),
    target_lang: str = Form(...)
):
    try:
        # Salva temporaneamente il file audio
        filename = f"uploads/{uuid.uuid4()}_{file.filename}"
        with open(filename, "wb") as f:
            f.write(await file.read())

        # Trascrivi con Whisper
        result = model.transcribe(
            filename,
            language=None if source_lang == "auto" else source_lang,
            task="transcribe"
        )
        testo_orig = result["text"]

        # Traduci se necessario
        if target_lang != source_lang:
            testo_tradotto = GoogleTranslator(source='auto', target=target_lang).translate(testo_orig)
        else:
            testo_tradotto = testo_orig

        # Elimina file dopo l'elaborazione
        os.remove(filename)

        return JSONResponse({
            "transcription": testo_orig,
            "translation": testo_tradotto
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
