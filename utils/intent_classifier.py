import joblib
from sentence_transformers import SentenceTransformer
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
import os
import re

# Get the absolute path to the model directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "data", "intent_clf_transformer.joblib")
EMBED_MODEL_PATH = os.path.join(BASE_DIR, "data", "embed_model")

# Load model components only once
clf, label_encoder = joblib.load(MODEL_PATH)
embed_model = SentenceTransformer(EMBED_MODEL_PATH)

DetectorFactory.seed=0

def detect_intent_multilingual(text: str) -> str:
    """
    Predicts the intent label of the input text using multilingual embeddings.
    """
    embedding = embed_model.encode([text])
    prediction = clf.predict(embedding)
    return label_encoder.inverse_transform(prediction)[0]

def detect_language_robust(text):
    clean_text = re.sub(r'[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u0600-\u06FF\u0B80-\u0BFF]', '', text).strip()
    lowered = clean_text.lower()
    if len(lowered) < 5:
        # For very short text, manually infer using known phrases (as a lightweight fallback)
        short_map = {
    "hi": "en", "hello": "en", "hey": "en", "greetings": "en",
    "hola": "es", "buenas": "es", "buenos días": "es", "qué tal": "es",
    "bonjour": "fr", "salut": "fr", "coucou": "fr",
    "hallo": "de", "guten tag": "de", "servus": "de", "grüß gott": "de",
    "ciao": "it", "salve": "it", "buongiorno": "it",
    "こんにちは": "ja", "おはよう": "ja", "もしもし": "ja",
    "안녕": "ko", "안녕하세요": "ko", "안녕하십니까": "ko",
    "你好": "zh-cn", "您好": "zh-cn", "哈喽": "zh-cn",
    "नमस्ते": "hi", "नमस्कार": "hi", "प्रणाम": "hi",
    "வணக்கம்": "ta", "ஹலோ": "ta",
    "హలో": "te", "నమస్కారం": "te",
    "ഹലോ": "ml", "നമസ്കാരം": "ml",
    "සුභ දවසක්": "si", "ආයුබෝවන්": "si",
    "สวัสดี": "th", "หวัดดี": "th",
    "xin chào": "vi", "chào": "vi",
    "merhaba": "tr", "selam": "tr",
    "سلام": "fa", "درود": "fa",
    "مرحبا": "ar", "أهلا": "ar", "السلام عليكم": "ar",
    "שלום": "he", "אהלן": "he",
    "здрасте": "ru", "привет": "ru", "здравствуйте": "ru",
    "hei": "no", "hallo": "no",
    "hej": "sv", "hallå": "sv",
    "hei": "fi", "terve": "fi",
    "sawubona": "zu", "molo": "xh",
    "salaam": "ur", "ہیلو": "ur"
}

        return short_map.get(lowered, "en")
    try:
        return detect(text)
    except:
        return "en"

def welcome_message(user_text):
    user_lang=detect_language_robust(user_text)
    translator=GoogleTranslator(source='en', target=user_lang)
    return translator.translate("Hello! How can I help you today?")

def unrelated_message(user_text):
    user_lang=detect_language_robust(user_text)
    translator=GoogleTranslator(source='en', target=user_lang)
    return translator.translate("Sorry, I didn't understand that request.")