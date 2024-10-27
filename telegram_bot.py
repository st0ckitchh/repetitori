from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
import logging
import io
import os
from typing import Optional, Dict

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot Configuration from Heroku environment variables
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
OPENAI_KEY = os.environ.get('OPENAI_KEY')
ANTHROPIC_KEY = os.environ.get('ANTHROPIC_KEY')

# Validate environment variables
if not all([TELEGRAM_TOKEN, OPENAI_KEY, ANTHROPIC_KEY]):
    logger.error("Missing required environment variables. Please check Heroku config vars.")
    raise ValueError("Missing required environment variables")

# User State Storage
user_states: Dict[int, Dict] = {}

def format_math_text(text):
    """Format mathematical text to be more readable"""
    replacements = {
        '\\\\': '\\',  # Handle double backslashes first
        '\\overrightarrow': '→',
        '\\[': '',
        '\\]': '',
        '\\vec': '→',
        '_1': '₁',
        '_2': '₂',
        '_3': '₃',
        '_4': '₄',
        '\\{': '{',
        '\\}': '}',
        '\\cdot': '·',
        '\\times': '×',
        '\\rightarrow': '→',
        '\\leftarrow': '←',
        '\\Rightarrow': '⇒',
        '\\Leftarrow': '⇐',
        '\\approx': '≈',
        '\\neq': '≠',
        '\\leq': '≤',
        '\\geq': '≥',
        '\\sqrt': '√',
        '\\infty': '∞',
        '\\pi': 'π',
        ' \\': ' ',  # Remove standalone backslashes
        '\\(': '(',
        '\\)': ')',
    }
    
    # First pass: handle basic replacements
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Second pass: clean up any remaining single backslashes
    text = text.replace('\\', '')
    
    # Handle special cases for vector notation
    text = text.replace('M1', 'M₁').replace('M2', 'M₂')
    
    # Add proper line breaks for equations
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if '=' in line and not line.startswith(('•', '-', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
            formatted_lines.append(f'```{line}```')
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

async def analyze_image_with_openai(image_data: bytes, question: str):
    try:
        import base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_KEY}"
        }
        
        payload = {
            "model": "o1-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f'''You are a professional student helper. Here i am sending a photo of the topic/problem. Please analyze the topic/problem and anwer the question.It can be any topic (math, logics, biology, any topic) Be as informative as possible and follow this guidlines:
1. Start with a clear statement of what we're solving
2. Break down the solution into numbered steps
3. If it's math, Put each mathematical equation on a new line. If it's not - place paragraphs on new lines.
4. If it's math problem, use simple mathematical notation without escape characters
5. If it's math, for vectors, use arrow notation (→) directly without latex commands
6. If it's math, Write vector names simply (like M₁ instead of M_1)
7. Explain each step clearly
8. Show all explanations, calculations if it's math
9. End with a clear conclusion
10. Provide a detailed theoretical materials so that students can learn the topic. Be as useful as possible. Your descriptions must be simple and understadnable.

Question: {question}'''
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 10000
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.text}")
                
            data = response.json()
            return data['choices'][0]['message']['content']

    except Exception as e:
        logger.error(f"Error in analyze_image_with_openai: {str(e)}")
        raise

async def translate_with_claude(text: str):
    try:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 4000,
            "messages": [
                {
                    "role": "user",
                    "content": f'''Please translate this mathematical solution to Georgian, maintaining the following:
1. Keep all mathematical formulas, symbols, and numbers exactly as they are
2. Keep all vector notations (→) unchanged
3. Preserve line breaks, especially for equations
4. Keep step numbers and formatting intact
5. Translate only the explanatory text
6. Preserve all mathematical notation exactly as shown

Text to translate:
{text}'''
                }
            ]
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"Claude API error: {response.text}")
                
            data = response.json()
            return data['content'][0]['text']

    except Exception as e:
        logger.error(f"Error in translate_with_claude: {str(e)}")
        raise

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    user_id = update.effective_user.id
    welcome_message = (
        "👋 Welcome to the Math Analysis Bot!\n\n"
        "I can help you solve mathematical problems. Here's how to use me:\n"
        "1. Send me an image of your math problem\n"
        "2. Ask your question about the problem\n"
        "3. I'll provide a detailed solution in both English and Georgian\n\n"
        "Commands:\n"
        "/start - Show this welcome message\n"
        "/help - Get help information\n"
        "/cancel - Cancel current operation"
    )
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    help_text = (
        "🤖 Bot Usage Instructions:\n\n"
        "1. Send an image of your math problem\n"
        "2. After the image is received, ask your question\n"
        "3. Wait for the analysis (this might take a few seconds)\n"
        "4. Get step-by-step solutions in both English and Georgian\n\n"
        "Tips for best results:\n"
        "• Make sure the image is clear and readable\n"
        "• Frame your question clearly\n"
        "• Be specific about what you want to know\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/cancel - Cancel current operation"
    )
    await update.message.reply_text(help_text)

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel the current operation."""
    user_id = update.effective_user.id
    if user_id in user_states:
        del user_states[user_id]
        await update.message.reply_text("Operation cancelled. You can start again by sending a new image.")
    else:
        await update.message.reply_text("No active operation to cancel.")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle received images."""
    try:
        user_id = update.effective_user.id
        
        # Get the largest available photo
        photo = max(update.message.photo, key=lambda x: x.file_size)
        
        # Download the photo
        file = await context.bot.get_file(photo.file_id)
        f = io.BytesIO()
        await file.download_to_memory(f)
        
        # Store the image data in user state
        user_states[user_id] = {
            "image_data": f.getvalue()
        }
        
        await update.message.reply_text(
            "Image received! 🖼\n"
            "Now, please ask your question about this math problem."
        )
        
    except Exception as e:
        logger.error(f"Error handling image: {str(e)}")
        await update.message.reply_text(
            "Sorry, there was an error processing your image. "
            "Please try sending it again."
        )

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages (questions about the image)."""
    user_id = update.effective_user.id
    
    if user_id not in user_states or "image_data" not in user_states[user_id]:
        await update.message.reply_text(
            "Please send an image of your math problem first before asking a question."
        )
        return
    
    try:
        # Send "processing" message
        processing_message = await update.message.reply_text(
            "Processing your request... ⏳\n"
            "This might take a few seconds."
        )
        
        # Get image data and question
        image_data = user_states[user_id]["image_data"]
        question = update.message.text
        
        # Analyze image
        english_answer = await analyze_image_with_openai(image_data, question)
        formatted_english = format_math_text(english_answer)
        
        # Send English response first
        await update.message.reply_text(
            "🇬🇧 *English Solution:*\n\n"
            f"{formatted_english}",
            parse_mode='Markdown'
        )
        
        # Update processing message
        await processing_message.edit_text(
            "English solution complete ✅\n"
            "Now translating to Georgian... ⏳"
        )
        
        # Translate to Georgian
        georgian_answer = await translate_with_claude(english_answer)
        formatted_georgian = format_math_text(georgian_answer)
        
        # Send Georgian response
        await update.message.reply_text(
            "🇬🇪 *ქართული ამოხსნა:*\n\n"
            f"{formatted_georgian}",
            parse_mode='Markdown'
        )
        
        # Clean up user state
        del user_states[user_id]
        
        # Update processing message
        await processing_message.edit_text("✅ Solution complete!")
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        await update.message.reply_text(
            "Sorry, there was an error processing your request. "
            "Please try again or send a new image."
        )
        if user_id in user_states:
            del user_states[user_id]

def main():
    """Start the bot."""
    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("cancel", cancel))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    # Start the Bot
    print("🤖 Math Analysis Bot is starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
