import os
import logging
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ===== CONFIG =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # from GitHub Secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # from GitHub Secrets

# Text generation endpoint
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Image generation endpoint (experimental, uses Imagen/Gemini)
GEMINI_IMAGE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0:generateImage?key={GEMINI_API_KEY}"

# ===== LOGGING =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===== AI TEXT REPLY =====
def get_gemini_text(prompt: str) -> str:
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(GEMINI_URL, json=payload, timeout=30)
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        logger.error(f"Gemini text error: {e}")
        return "⚠️ Sorry, AI could not reply."


# ===== AI IMAGE GENERATION =====
def get_gemini_image(prompt: str) -> str:
    payload = {"prompt": prompt}
    try:
        resp = requests.post(GEMINI_IMAGE_URL, json=payload, timeout=60)
        data = resp.json()

        # Gemini may return image as base64 → but Telegram needs a URL or file
        if "images" in data and len(data["images"]) > 0:
            image_b64 = data["images"][0]["data"]
            # Save as temp file
            filename = "output.jpg"
            with open(filename, "wb") as f:
                f.write(requests.utils.unquote_to_bytes(image_b64))
            return filename
        else:
            return None
    except Exception as e:
        logger.error(f"Gemini image error: {e}")
        return None


# ===== HANDLERS =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to LumiInvest AI!\n\nJust send me a message.\n"
        "📸 To generate an image, start with `image:`\nExample: `image: a cat in sunglasses`"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text

    # Check for image request
    if text.lower().startswith("image:"):
        prompt = text[6:].strip()
        await update.message.reply_text("🎨 Generating image, please wait...")

        image_file = get_gemini_image(prompt)
        if image_file:
            await update.message.reply_photo(photo=open(image_file, "rb"), caption=f"🖼️ Generated: {prompt}")
        else:
            await update.message.reply_text("⚠️ Sorry, I couldn't generate the image.")
    else:
        # Normal text reply
        reply = get_gemini_text(text)
        await update.message.reply_text(f"🤖 {reply}")


# ===== MAIN =====
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling()


if __name__ == "__main__":
    main()
