import os
import logging
import requests
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ===== CONFIG =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # GitHub Secret
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # GitHub Secret
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face token (GitHub Secret)

# Gemini Text API
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Hugging Face Image API (Stable Diffusion)
HF_MODEL = "runwayml/stable-diffusion-v1-5"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# ===== LOGGING =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===== GEMINI TEXT REPLY =====
def get_gemini_text(prompt: str) -> str:
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(GEMINI_URL, json=payload, timeout=30)
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        logger.error(f"Gemini text error: {e}")
        return "⚠️ Sorry, AI could not reply."


# ===== HUGGING FACE IMAGE GENERATION =====
def get_hf_image(prompt: str) -> str | None:
    try:
        payload = {"inputs": prompt}
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)

        if resp.status_code == 200:
            filename = "generated.png"
            with open(filename, "wb") as f:
                f.write(resp.content)
            return filename
        else:
            logger.error(f"Hugging Face error {resp.status_code}: {resp.text}")
            return None
    except Exception as e:
        logger.error(f"Hugging Face image error: {e}")
        return None


# ===== HANDLERS =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to LumiInvest AI!\n\n"
        "💬 Just send me a message to chat.\n"
        "📸 To generate an image, start with `image:`\n"
        "Example: `image: a futuristic city`"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text

    if text.lower().startswith("image:"):
        prompt = text[6:].strip()
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
        await update.message.reply_text("🎨 Generating image, please wait...")

        image_file = get_hf_image(prompt)
        if image_file:
            await update.message.reply_photo(photo=open(image_file, "rb"), caption=f"🖼️ Generated: {prompt}")
        else:
            await update.message.reply_text("⚠️ Sorry, I couldn't generate the image.")
    else:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
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
