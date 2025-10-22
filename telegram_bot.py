
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from ml_bot.app.train import train_model
from ml_bot.app.predict import predict
from flask import Flask
from threading import Thread

# Simple web app for Render to see
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Bot is alive."

def run_flask():
    # Render expects the web service on port 8080
    app.run(host="0.0.0.0", port=8080)

TOKEN = "8454856306:AAHTouyuu4ii1xFQSMkzSbv51fTlFnNtwJo"  # Replace this with your real bot token

keyboard = [['🚀 Deploy Model', '📈 Predict', '📊 Status']]
markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=False, resize_keyboard=True)

def start(update: Update, context: CallbackContext):
    update.message.reply_text("Welcome to your ML Bot Control Panel!", reply_markup=markup)

def handle_message(update: Update, context: CallbackContext):
    text = update.message.text

    if text == "🚀 Deploy Model":
        update.message.reply_text("Training model...")
        try:
            acc = train_model()
            update.message.reply_text(f"Model trained ✅ Accuracy: {acc:.2%}")
        except Exception as e:
            update.message.reply_text(f"Training failed ❌: {e}")

    elif text == "📈 Predict":
        input_data = {"experience_level": 3, "remote_ratio": 100, "company_size": 2}
        try:
            result = predict(input_data)
            update.message.reply_text(f"Prediction: {result}")
        except Exception as e:
            update.message.reply_text(f"Prediction failed ❌: {e}")

    elif text == "📊 Status":
        update.message.reply_text("Bot is online and operational.")

    else:
        update.message.reply_text("Unknown command.")

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
