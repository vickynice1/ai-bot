import os
import logging
import requests
import time
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

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
    payload = {"inputs": prompt}
    for attempt in range(5):
        try:
            resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                filename = "generated.png"
                with open(filename, "wb") as f:
                    f.write(resp.content)
                return filename
            elif resp.status_code == 503 and "loading" in resp.text.lower():
                wait_time = 5 * (attempt + 1)
                logger.info(f"Model loading... retrying in {wait_time}s")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Hugging Face error {resp.status_code}: {resp.text}")
                return None
        except Exception as e:
            logger.error(f"Hugging Face image error: {e}")
            return None
    return None

# ===== SPLIT LONG MESSAGES =====
def split_message(text: str, chunk_size: int = 4000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ===== HANDLERS =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to LumiInvest AI!\n\n"
        "💬 Just send me a message to chat.\n"
        "📸 To generate an image, start with `image:`\n"
        "Example: `image: a futuristic city`"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    # IMAGE GENERATION
    if text.lower().startswith("image:"):
        prompt = text[6:].strip()
        await update.message.reply_text("⏳ Please wait while I generate your image...")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
        image_file = get_hf_image(prompt)
        if image_file:
            await update.message.reply_photo(photo=open(image_file, "rb"), caption=f"🖼️ Generated: {prompt}")
        else:
            await update.message.reply_text("⚠️ Sorry, I couldn't generate the image.")
        return

    # FIXED RESPONSE
    if any(keyword in text.lower() for keyword in ["what were you created for", "who created you", "purpose", "created for"]):
        reply = get_gemini_text(
            f"Translate into the same language as: '{text}'. "
            "Message: '🌍 I was created to help Telegram users. I'm here to answer your questions and assist you.'"
        )
        await update.message.reply_text(f"🤖 {reply}")
        return

    # NORMAL CHAT
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    reply = get_gemini_text(text)

    chunks = split_message(reply)
    await update.message.reply_text(f"🤖 {chunks[0]}")

    # Store remaining chunks for later
    if len(chunks) > 1:
        context.user_data["reply_chunks"] = chunks[1:]
        keyboard = InlineKeyboardMarkup.from_button(
            InlineKeyboardButton("➡️ Continue", callback_data="continue_reply")
        )
        await update.message.reply_text("Message too long. Tap below to continue ⬇️", reply_markup=keyboard)

async def continue_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if "reply_chunks" in context.user_data and context.user_data["reply_chunks"]:
        next_chunk = context.user_data["reply_chunks"].pop(0)
        await query.message.reply_text(next_chunk)

        if context.user_data["reply_chunks"]:  # Still more left
            keyboard = InlineKeyboardMarkup.from_button(
                InlineKeyboardButton("➡️ Continue", callback_data="continue_reply")
            )
            await query.message.reply_text("⬇️ Continue reading:", reply_markup=keyboard)

# ===== MAIN =====
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(continue_reply, pattern="continue_reply"))
    app.run_polling()

if __name__ == "__main__":
    main()
