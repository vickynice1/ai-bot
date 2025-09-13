import os
import logging
import requests
import time
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

# ===== CONFIG =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Gemini Text API
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Replicate API
REPLICATE_URL = "https://api.replicate.com/v1/predictions"

# Use version ID, not model name
# SDXL version ID (example from replicate.com/stability-ai/stable-diffusion-xl)
REPLICATE_MODEL_VERSION = "8f8dd66e8e67a63d02f69413fbb4e9e3d8a8efbb2f4c4df9c3c1c2d3e5f58c86"

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


# ===== REPLICATE IMAGE GENERATION =====
def get_replicate_image(prompt: str) -> str | None:
    headers = {
        "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "version": REPLICATE_MODEL_VERSION,
        "input": {"prompt": prompt}
    }

    try:
        # Create prediction
        resp = requests.post(REPLICATE_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code != 201:
            logger.error(f"Replicate error {resp.status_code}: {resp.text}")
            return None

        prediction = resp.json()
        prediction_url = prediction["urls"]["get"]

        # Poll until completed
        for _ in range(20):
            result = requests.get(prediction_url, headers=headers).json()
            if result["status"] == "succeeded":
                return result["output"][0]  # image URL
            elif result["status"] == "failed":
                logger.error(f"Replicate prediction failed: {result}")
                return None
            time.sleep(5)

    except Exception as e:
        logger.error(f"Replicate error: {e}")

    return None


# ===== SPLIT LONG MESSAGES =====
def split_message(text: str, chunk_size: int = 4000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


# ===== HANDLERS =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("🇬🇧 English", callback_data="lang_en"),
            InlineKeyboardButton("🇳🇬 Igbo", callback_data="lang_ig"),
        ],
        [
            InlineKeyboardButton("🇷🇺 Russian", callback_data="lang_ru"),
            InlineKeyboardButton("🇵🇹 Portuguese", callback_data="lang_pt"),
        ],
    ]
    await update.message.reply_text(
        "👋 Welcome to LumiInvest AI!\n\nPlease choose your language:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def set_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    lang_map = {
        "lang_en": "English",
        "lang_ig": "Igbo",
        "lang_ru": "Russian",
        "lang_pt": "Portuguese",
    }

    choice = lang_map.get(query.data, "English")
    context.user_data["language"] = choice

    await query.edit_message_text(f"✅ Language set to {choice}! Now you can chat with me.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    user = update.effective_user
    language = context.user_data.get("language", "English")

    # Store user info
    context.user_data["first_name"] = user.first_name
    context.user_data["username"] = user.username

    # IMAGE GENERATION
    if text.lower().startswith("image:"):
        prompt = text[6:].strip()
        await update.message.reply_text("⏳ Generating your image with Replicate...")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)

        image_url = get_replicate_image(prompt)
        if image_url:
            await update.message.reply_photo(photo=image_url, caption=f"🖼️ Generated: {prompt}")
        else:
            await update.message.reply_text("⚠️ Sorry, I couldn't generate the image.")
        return

    # NAME QUESTION
    if "what is my name" in text.lower():
        if user.username:
            name_reply = f"Your name is {user.first_name} (@{user.username})"
        else:
            name_reply = f"Your name is {user.first_name}"

        reply = get_gemini_text(f"Translate into {language}: '{name_reply}'")
        await update.message.reply_text(f"🤖 {reply}")
        return

    # NORMAL CHAT
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    reply = get_gemini_text(f"Reply in {language} to this message: {text}")

    chunks = split_message(reply)
    await update.message.reply_text(f"🤖 {chunks[0]}")

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

        if context.user_data["reply_chunks"]:
            keyboard = InlineKeyboardMarkup.from_button(
                InlineKeyboardButton("➡️ Continue", callback_data="continue_reply")
            )
            await query.message.reply_text("⬇️ Continue reading:", reply_markup=keyboard)


# ===== MAIN =====
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(set_language, pattern="^lang_"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(continue_reply, pattern="continue_reply"))
    app.run_polling()


if __name__ == "__main__":
    main()
