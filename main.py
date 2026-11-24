import telebot
from logic import detect_object, graphic_percent, get_recommendation

BOT_TOKEN = '8306783272:AAFiZ91rRxaPo0T00vPKcBQ_kGi4LNKj1NU'

bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.reply_to(message, '''Привет, это бот для определения сколько % фото зеленое,
      это можно использовать для определения озелененности города
      так как он также показывает сколько процентов города должно быть озеленено,
      чтобы использовать это достаточно просто скинуть ему фото''')


@bot.message_handler(content_types=['photo'])
def message_photo(message):
    if not message.photo:
        return bot.send_message(message.chat.id, "Give a photo")
    file_info = bot.get_file(message.photo[-1].file_id)
    file_name = file_info.file_path.split('/')[-1]
    downloaded_file = bot.download_file(file_info.file_path) 
    with open(file_name, 'wb') as new_file:
        new_file.write(downloaded_file)
    objects = detect_object(file_name)
    text = ''
    for name, percent in objects:
        text += graphic_percent(name, percent) + "\n"
    text += "\n" + get_recommendation(objects)
    bot.reply_to(message, text)


bot.polling()
