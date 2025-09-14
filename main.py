import os
import json
import logging
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

# ===== AI MODELS =====
import requests
import torch
from diffusers import StableDiffusionPipeline

# ===== CONFIG =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-3.5-turbo")
ADMIN_ID = int(os.getenv("ADMIN_ID", "123456789"))  # set your Telegram ID here
DATA_FILE = "user_data.json"

# ===== LOGGING =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== USER DATA (POINTS) =====
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

def get_points(user_id):
    data = load_data()
    return data.get(str(user_id), {}).get("points", 2)  # free users start with 2 points

def update_points(user_id, delta):
    data = load_data()
    user_data = data.get(str(user_id), {"points": 2})
    user_data["points"] = max(0, user_data.get("points", 2) + delta)
    data[str(user_id)] = user_data
    save_data(data)

# ===== TEXT AI =====
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

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

# ===== LOCAL STABLE DIFFUSION =====
MODEL_ID = "runwayml/stable-diffusion-v1-5"
try:
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, dtype=torch.float16)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        logger.info("Stable Diffusion model loaded to GPU.")
    else:
        logger.info("Stable Diffusion model loaded to CPU (slower).")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID)
    logger.info("Loaded Stable Diffusion model in CPU float32.")

def generate_sd_image(prompt: str, steps: int, scale: float) -> BytesIO | None:
    try:
        image = pipe(prompt, num_inference_steps=steps, guidance_scale=scale).images[0]
        bio = BytesIO()
        image.save(bio, format="PNG")
        bio.seek(0)
        return bio
    except Exception as e:
        logger.error(f"Stable Diffusion error: {e}")
        return None

# ===== PIL IMAGE GENERATION =====
def generate_pillow_image(user_text: str) -> BytesIO:
    width, height = 500, 300
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 50, width - 50, height - 50), fill=(0, 128, 255))

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), user_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text(((width - text_width) / 2, (height - text_height) / 2),
              user_text, fill=(255, 255, 255), font=font)

    bio = BytesIO()
    image.save(bio, format="PNG")
    bio.seek(0)
    return bio

# ===== LADDERED TEXT REPLY =====
def generate_laddered_reply(messages: list[dict]) -> str:
    ai_funcs = [get_openai_text]
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
        "Choose an option below 👇"
    )
    keyboard = [
        [InlineKeyboardButton("💰 Balance", callback_data="balance")],
        [InlineKeyboardButton("🖼️ Generate Image", callback_data="imagine_menu")],
        [InlineKeyboardButton("💳 Buy Points", callback_data="buy_points")]
    ]
    await update.message.reply_text(greeting, reply_markup=InlineKeyboardMarkup(keyboard))
    context.user_data["conversation_history"] = []

async def menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()

    if query.data == "balance":
        points = get_points(user_id)
        await query.edit_message_text(
            f"💰 You have {points} points.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Refresh Balance", callback_data="balance")],
                [InlineKeyboardButton("🖼️ Generate Image", callback_data="imagine_menu")],
                [InlineKeyboardButton("💳 Buy Points", callback_data="buy_points")]
            ])
        )

    elif query.data == "buy_points":
        await query.message.reply_text("💳 Request sent to admin.")
        await context.bot.send_message(
            chat_id=ADMIN_ID,
            text=f"📥 User {query.from_user.full_name} (ID: {user_id}) wants to buy points.\nUse /confirm {user_id} <points> to approve."
        )

    elif query.data == "imagine_menu":
        keyboard = [
            [InlineKeyboardButton("Low Quality (Free)", callback_data="imagine_low")],
            [InlineKeyboardButton("Medium Quality (1 pt)", callback_data="imagine_medium")],
            [InlineKeyboardButton("High Quality (2 pts)", callback_data="imagine_high")],
            [InlineKeyboardButton("💰 Balance", callback_data="balance")]
        ]
        await query.message.reply_text("🎨 Choose image quality:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data.startswith("imagine_"):
        quality = query.data.split("_")[1]
        await query.message.reply_text(f"Send me your prompt now for {quality} quality image.")
        context.user_data["awaiting_prompt"] = quality

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    user_id = update.effective_user.id

    # If awaiting prompt for imagine
    if "awaiting_prompt" in context.user_data:
        quality = context.user_data.pop("awaiting_prompt")
        points = get_points(user_id)

        if quality == "low":
            steps, scale, cost = 20, 6.0, 0
        elif quality == "medium":
            steps, scale, cost = 30, 7.5, 1
        else:  # high
            steps, scale, cost = 50, 8.5, 2

        if points < cost:
            await update.message.reply_text("⚠️ Not enough points! Use Buy Points option.")
            return

        await update.message.reply_text(f"⏳ Generating {quality} quality image (cost {cost} pts)...")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
        image = generate_sd_image(text, steps, scale)
        if image:
            if cost > 0:
                update_points(user_id, -cost)
            new_balance = get_points(user_id)
            await update.message.reply_photo(
                photo=image,
                caption=f"🖼️ Prompt: {text} ({quality})\n💰 Balance: {new_balance} pts"
            )
        else:
            await update.message.reply_text("⚠️ Image generation failed.")
        return

    # Otherwise handle as chat
    context.user_data.setdefault("conversation_history", [])
    try:
        detect(text)
    except:
        pass
    context.user_data["conversation_history"].append({"role": "user", "content": text})
    recent_context = context.user_data["conversation_history"][-15:]
    ai_reply = generate_laddered_reply(recent_context) or "⚠️ Sorry, I couldn't generate a reply."
    context.user_data["conversation_history"].append({"role": "assistant", "content": ai_reply})
    await update.message.reply_text(f"🤖 {ai_reply}")

async def confirm_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("⚠️ Only admin can confirm purchases.")
        return
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /confirm <user_id> <points>")
        return
    user_id, points = context.args[0], int(context.args[1])
    update_points(user_id, points)
    await update.message.reply_text(f"✅ Added {points} points to {user_id}.")
    await context.bot.send_message(chat_id=user_id, text=f"🎉 Admin credited you with {points} points!")

# ===== MAIN =====
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("confirm", confirm_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(menu_handler))
    app.run_polling()

if __name__ == "__main__":
    main()
