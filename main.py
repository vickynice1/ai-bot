import os
import logging
import requests
import base64
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ===== CONFIG =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # from GitHub Secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # from GitHub Secrets

# Text generation endpoint
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Imagen endpoint
GEMINI_IMAGE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

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
def get_gemini_image(prompt: str) -> str | None:
    payload = {"instances": [{"prompt": prompt}]}  # ✅ Correct schema
    try:
        resp = requests.post(GEMINI_IMAGE_URL, json=payload, timeout=60)
        data = resp.json()

        # If API returns base64 image
        if "predictions" in data and len(data["predictions"]) > 0:
            image_b64 = data["predictions"][0]["bytesBase64Encoded"]
            filename = "output.jpg"
            with open(filename, "wb") as f:
                f.write(base64.b64decode(image_b64))  # ✅ Correct decoding
            return filename
        else:
            logger.error(f"Unexpected image response: {data}")
            return None
    except Exception as e:
        logger.error(f"Gemini image error: {e}")
        return None


# ===== HANDLERS =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to LumiInvest AI!\n\n"
        "💬 Just send me a message to chat.\n"
        "📸 To generate an image, start with `image:`\n"
        "Example: `image: a cat in sunglasses`"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text

    # Show typing...
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    if text.lower().startswith("image:"):
        prompt = text[6:].strip()
        await update.message.reply_text("🎨 Generating image, please wait...")

        image_file = get_gemini_image(prompt)
        if image_file:
            await update.message.reply_photo(photo=open(image_file, "rb"), caption=f"🖼️ Generated: {prompt}")
        else:
            await update.message.reply_text("⚠️ Sorry, I couldn't generate the image.")
    else:
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
