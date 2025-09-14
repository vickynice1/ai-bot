import os
import logging
import time
import random
import requests
from io import BytesIO
from langdetect import detect
from PIL import Image, ImageDraw, ImageFont
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
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")  # images only
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-3.5-turbo")

# Gemini 2.5 API
GEMINI_TEXT_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
GEMINI_IMAGE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-image:generateImage?key={GEMINI_API_KEY}"

# Replicate API (images only)
REPLICATE_IMAGE_MODEL = "black-forest-labs/flux-dev"
REPLICATE_URL = "https://api.replicate.com/v1/predictions"

# OpenAI API
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_IMAGE_URL = "https://api.openai.com/v1/images/generations"

# ===== LOGGING =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== TEXT AI FUNCTIONS =====
def get_gemini_text(prompt: str) -> str | None:
    try:
        resp = requests.post(GEMINI_TEXT_URL, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30)
        if resp.status_code != 200:
            logger.error(f"Gemini text API failed: {resp.status_code} {resp.text}")
            return None
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        logger.error(f"Gemini text error: {e}")
        return None

def get_openai_text(messages: list[dict]) -> str | None:
    try:
        resp = requests.post(
            OPENAI_CHAT_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"model": OPENAI_TEXT_MODEL, "messages": messages, "temperature": 0.7},
            timeout=30
        )
        if resp.status_code != 200:
            logger.error(f"OpenAI text API failed: {resp.status_code} {resp.text}")
            return None
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"OpenAI text exception: {e}")
        return None

# ===== IMAGE AI FUNCTIONS =====
def get_gemini_image(prompt: str) -> str | None:
    try:
        resp = requests.post(GEMINI_IMAGE_URL, json={"prompt": prompt, "size": "1024x1024"}, timeout=30)
        if resp.status_code != 200:
            logger.error(f"Gemini image API failed: {resp.status_code} {resp.text}")
            return None
        return resp.json().get("url")
    except Exception as e:
        logger.error(f"Gemini image exception: {e}")
        return None

def get_openai_image(prompt: str) -> str | None:
    try:
        resp = requests.post(
            OPENAI_IMAGE_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"model": "gpt-image-1", "prompt": prompt, "size": "1024x1024"},
            timeout=30
        )
        if resp.status_code != 200:
            logger.error(f"OpenAI image API failed: {resp.status_code} {resp.text}")
            return None
        data = resp.json()
        if "data" in data and len(data["data"]) > 0:
            return data["data"][0].get("url")
        logger.error(f"OpenAI image returned no data: {data}")
        return None
    except Exception as e:
        logger.error(f"OpenAI image exception: {e}")
        return None

def get_replicate_image(prompt: str) -> str | None:
    headers = {"Authorization": f"Bearer {REPLICATE_API_KEY}"}
    payload = {
        "version": REPLICATE_IMAGE_MODEL,
        "input": {"prompt": prompt, "num_outputs": 1, "aspect_ratio": "1:1", "output_format": "png"}
    }
    try:
        resp = requests.post(REPLICATE_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code != 201:
            logger.error(f"Replicate image error {resp.status_code}: {resp.text}")
            return None
        prediction_url = resp.json()["urls"]["get"]
        for _ in range(20):
            result = requests.get(prediction_url, headers=headers).json()
            if result["status"] == "succeeded":
                return result["output"][0]
            elif result["status"] == "failed":
                return None
            time.sleep(5)
    except Exception as e:
        logger.error(f"Replicate image exception: {e}")
        return None

def get_random_image(prompt: str) -> str | None:
    """Try OpenAI -> Gemini -> Replicate for image generation."""
    ai_funcs = [get_openai_image, get_gemini_image, get_replicate_image]
    for func in ai_funcs:
        try:
            url = func(prompt)
            if url:
                return url
        except Exception as e:
            logger.error(f"Image AI {func.__name__} failed: {e}")
    return None

# ===== PIL IMAGE GENERATION FUNCTION =====
def generate_pillow_image(user_text: str) -> BytesIO:
    """Generates a simple Pillow image with text."""
    width, height = 500, 300
    bg_color = (255, 255, 255)
    rect_color = (0, 128, 255)
    image = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(image)

    # Draw rectangle
    draw.rectangle((50, 50, width-50, height-50), fill=rect_color)

    # Draw text
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    text_width, text_height = draw.textsize(user_text, font=font)
    draw.text(
        ((width - text_width) / 2, (height - text_height) / 2),
        user_text, fill=(255, 255, 255), font=font
    )

    # Save to BytesIO
    bio = BytesIO()
    image.save(bio, format='PNG')
    bio.seek(0)
    return bio

# ===== LADDERED MULTI-AI TEXT REPLY =====
def generate_laddered_reply(messages: list[dict]) -> str:
    """Try OpenAI first, then Gemini."""
    ai_funcs = [
        get_openai_text,
        lambda msgs: get_gemini_text("\n".join([m["content"] for m in msgs]))
    ]
    for func in ai_funcs:
        try:
            reply = func(messages)
            if reply:
                return reply
        except Exception as e:
            logger.error(f"Laddered AI {func.__name__} failed: {e}")
    return "⚠️ Sorry, I couldn't generate a reply."

# ===== HANDLERS =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    full_name = f"{user.first_name or ''} {user.last_name or ''}".strip()
    greeting = f"👋 Hello {full_name}!\n\nWelcome to LumiInvest AI!\nJust type your message and I will reply in your language.\nUse /imagine <prompt> to generate AI images or /generate <text> for a custom Pillow image."
    await update.message.reply_text(greeting)
    context.user_data["conversation_history"] = []

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    context.user_data.setdefault("conversation_history", [])
    try:
        detect(text)
    except:
        pass

    context.user_data["conversation_history"].append({"role": "user", "content": text})
    recent_context = context.user_data["conversation_history"][-15:]
    ai_reply = generate_laddered_reply(recent_context)
    if not ai_reply:
        ai_reply = "⚠️ Sorry, I couldn't generate a reply."

    context.user_data["conversation_history"].append({"role": "assistant", "content": ai_reply})

    # Split long messages into chunks
    chunk_size = 4000
    chunks = [ai_reply[i:i + chunk_size] for i in range(0, len(ai_reply), chunk_size)]
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

async def imagine_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Usage: /imagine <prompt>")
        return
    prompt = " ".join(context.args)
    await update.message.reply_text("⏳ Generating your AI image...")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
    image_url = get_random_image(prompt)
    if image_url:
        await update.message.reply_photo(photo=image_url, caption=f"🖼️ Generated: {prompt}")
    else:
        await update.message.reply_text("⚠️ Sorry, I couldn't generate the AI image.")

async def generate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate a custom Pillow image with user text"""
    if not context.args:
        await update.message.reply_text("⚠️ Usage: /generate <text>")
        return
    user_text = " ".join(context.args)
    await update.message.reply_text("⏳ Generating your custom image...")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
    image_bytes = generate_pillow_image(user_text)
    await update.message.reply_photo(photo=image_bytes, caption=f"🖼️ Custom Image: {user_text}")

# ===== MAIN =====
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("imagine", imagine_command))
    app.add_handler(CommandHandler("generate", generate_command))  # Pillow command
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(continue_reply, pattern="continue_reply"))
    app.run_polling()

if __name__ == "__main__":
    main()
