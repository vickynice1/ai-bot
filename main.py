import os
import logging
import time
import re
import requests
from io import BytesIO
from langdetect import detect
from PIL import Image, ImageDraw, ImageFont
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

# ===== GOOGLE GEMINI SDK =====
from google import genai
from google.genai import types

# ===== CONFIG =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")  # images only
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-3.5-turbo")

# ===== LOGGING =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== TELEGRAM FORMAT FIX =====
def format_for_telegram(text: str) -> str:
    # Convert **bold** → *bold*
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)

    # Escape Telegram MarkdownV2 special chars
    special_chars = r"_[]()~`>#+-=|{}.!"
    for ch in special_chars:
        text = text.replace(ch, f"\\{ch}")
    return text

# ===== TEXT AI FUNCTIONS =====
GEMINI_TEXT_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

def get_gemini_text(prompt: str) -> str | None:
    try:
        resp = requests.post(
            GEMINI_TEXT_URL,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30
        )
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
REPLICATE_IMAGE_MODEL = "black-forest-labs/flux-dev"
REPLICATE_URL = "https://api.replicate.com/v1/predictions"
OPENAI_IMAGE_URL = "https://api.openai.com/v1/images/generations"

# --- Gemini SDK image generation ---
def get_gemini_image_sdk(prompt: str) -> BytesIO | None:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[prompt]
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_bytes = BytesIO(part.inline_data.data)
                image_bytes.seek(0)
                return image_bytes
        return None
    except Exception as e:
        logger.error(f"Gemini SDK image error: {e}")
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

def get_random_image(prompt: str) -> BytesIO | str | None:
    funcs = [get_gemini_image_sdk, get_openai_image, get_replicate_image]
    for func in funcs:
        try:
            url_or_bytes = func(prompt)
            if url_or_bytes:
                return url_or_bytes
        except Exception as e:
            logger.error(f"Image AI {func.__name__} failed: {e}")
    return None

# ===== PIL IMAGE GENERATION FUNCTION =====
def generate_pillow_image(user_text: str) -> BytesIO:
    width, height = 500, 300
    bg_color = (255, 255, 255)
    rect_color = (0, 128, 255)
    image = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(image)

    draw.rectangle((50, 50, width-50, height-50), fill=rect_color)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    # Pillow >=10 fix: use textbbox()
    bbox = draw.textbbox((0, 0), user_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    draw.text(
        ((width - text_width) / 2, (height - text_height) / 2),
        user_text, fill=(255, 255, 255), font=font
    )

    bio = BytesIO()
    image.save(bio, format='PNG')
    bio.seek(0)
    return bio

# ===== LADDERED MULTI-AI TEXT REPLY =====
def generate_laddered_reply(messages: list[dict]) -> str:
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
    greeting = (
        f"👋 Hello {full_name}!\n\n"
        "Welcome to LumiInvest AI!\n"
        "Just type your message and I will reply in your language.\n"
        "Use /imagine <prompt> to generate AI images or /generate <text> for a custom Pillow image."
    )
    await update.message.reply_text(greeting, parse_mode=ParseMode.MARKDOWN_V2)
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

    ai_reply = format_for_telegram(ai_reply)

    context.user_data["conversation_history"].append({"role": "assistant", "content": ai_reply})

    chunk_size = 4000
    chunks = [ai_reply[i:i + chunk_size] for i in range(0, len(ai_reply), chunk_size)]
    await update.message.reply_text(
        f"🤖 {chunks[0]}",
        parse_mode=ParseMode.MARKDOWN_V2
    )
    if len(chunks) > 1:
        context.user_data["reply_chunks"] = chunks[1:]
        keyboard = InlineKeyboardMarkup.from_button(
            InlineKeyboardButton("➡️ Continue", callback_data="continue_reply")
        )
        await update.message.reply_text(
            "Message too long. Tap below to continue ⬇️",
            reply_markup=keyboard
        )

async def continue_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if "reply_chunks" in context.user_data and context.user_data["reply_chunks"]:
        next_chunk = context.user_data["reply_chunks"].pop(0)
        await query.message.reply_text(
            next_chunk,
            parse_mode=ParseMode.MARKDOWN_V2
        )
        if context.user_data["reply_chunks"]:
            keyboard = InlineKeyboardMarkup.from_button(
                InlineKeyboardButton("➡️ Continue", callback_data="continue_reply")
            )
            await query.message.reply_text(
                "⬇️ Continue reading:",
                reply_markup=keyboard
            )

async def imagine_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Usage: /imagine <prompt>")
        return
    prompt = " ".join(context.args)
    await update.message.reply_text("⏳ Generating your AI image...")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
    image = get_random_image(prompt)
    if isinstance(image, BytesIO):
        await update.message.reply_photo(photo=image, caption=f"🖼️ Generated: {prompt}")
    elif isinstance(image, str):
        await update.message.reply_photo(photo=image, caption=f"🖼️ Generated: {prompt}")
    else:
        await update.message.reply_text("⚠️ Sorry, I couldn't generate the AI image.")

async def generate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    app.add_handler(CommandHandler("generate", generate_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(continue_reply, pattern="continue_reply"))
    app.run_polling()

if __name__ == "__main__":
    main()
