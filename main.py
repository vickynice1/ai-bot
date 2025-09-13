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
HF_TOKEN = os.getenv("HF_TOKEN")

# Gemini Text API
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Hugging Face Image API
HF_MODEL = "runwayml/stable-diffusion-v1-5"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

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


# ===== HUGGING FACE IMAGE =====
def get_hf_image(prompt: str) -> str | None:
    payload = {"inputs": prompt}
    headers_list = [
        {"Authorization": f"Bearer {HF_TOKEN}"},
        {"Authorization": HF_TOKEN},
    ]
    for headers in headers_list:
        for attempt in range(5):
            try:
                resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
                if resp.status_code == 200:
                    filename = "generated.png"
                    with open(filename, "wb") as f:
                        f.write(resp.content)
                    return filename
                elif resp.status_code == 503 and "loading" in resp.text.lower():
                    time.sleep(5 * (attempt + 1))
                    continue
                else:
                    logger.error(f"HF error {resp.status_code}: {resp.text}")
                    break
            except Exception as e:
                logger.error(f"HF image error: {e}")
                break
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
        await update.message.reply_text("⏳ Generating your image...")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
        image_file = get_hf_image(prompt)
        if image_file:
            await update.message.reply_photo(photo=open(image_file, "rb"), caption=f"🖼️ Generated: {prompt}")
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
