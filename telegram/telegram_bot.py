"""
Simple telegram bot that uses cucafera-chat to answer user's mesages.
It only remembers the system prompt and the last message the user sent.
"""

import os
import telebot
from dotenv import load_dotenv
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format

load_dotenv()

bot = telebot.TeleBot(os.environ.get('BOT_TOKEN'))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("pauhidalgoo/cucafera-chat")
model = AutoModelForCausalLM.from_pretrained("pauhidalgoo/cucafera-chat")

model, tokenizer = setup_chat_format(model, tokenizer)

print(tokenizer.eos_token_id)
print(tokenizer.pad_token_id)

genconf = GenerationConfig(
    do_sample=True,
    max_length=2048,
    top_k=50,
    num_return_sequences=1,
    temperature=0.6,
    eos_token_id = tokenizer.eos_token_id,
    pad_token_id = 3,
    top_p=0.9,
    repetition_penalty=1.2,
    guidance_scale = 1.2,
    max_new_tokens = 150,
)

user_conversations = {}

@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, "Hola, amb què et puc ajudar?")
    user_conversations[message.chat.id] = []

@bot.message_handler(commands=['reset', 'reseteja'])
def send_welcome(message):
    bot.reply_to(message, "Ja no recordo res.")
    user_conversations[message.chat.id] = []

@bot.message_handler(commands=['redo', 'repeteix'])
def send_welcome(message):
    # This function needs revisiting, it isn't fully correct
    conversation = user_conversations.get(message.chat.id, ["<|im_start|>system \n Ets un assistent útil, que respon sempre el que et demanen i correctament. <|im_end|>\n"])
    conversation.pop()
    print(conversation)
    input_text = "".join(conversation)
    print(input_text)

    tokens = np.array(tokenizer.encode(input_text, return_tensors='pt', truncation="only_first", add_special_tokens=False))
    tokens = tokens[-2048:]
    

    tokens = torch.tensor(tokens, dtype=torch.long)

    prompt_length = tokens.shape[1]

    a = model.generate(tokens, genconf)
    response = tokenizer.decode(a[0][prompt_length:], skip_special_tokens=True)

    response = response.strip()
    
    conversation.append(response)
    
    user_conversations[message.chat.id] = conversation

    bot.reply_to(message, response)


@bot.message_handler(func=lambda msg: True)
def respond(message):
    conversation = user_conversations.get(message.chat.id, ["<|im_start|>system \n Respon a totes les preguntes de forma curta, clara i concisa. <|im_end|>\n"])
    prompt = f"<|im_start|>user\n{message.text}<|im_end|><|im_start|>assistant\n"

    conversation.append(prompt)

    input_text = "".join(conversation)

    tokens = np.array(tokenizer.encode(input_text, return_tensors='pt', truncation="only_first", add_special_tokens=False))
    tokens = tokens[-2048:]
    

    tokens = torch.tensor(tokens, dtype=torch.long)

    prompt_length = tokens.shape[1]

    a = model.generate(tokens, genconf)
    response = tokenizer.decode(a[0][prompt_length:], skip_special_tokens=True)

    response = response.strip()

    conversation.append(response)
    
    user_conversations[message.chat.id] = ["<|im_start|>system \n Respon a totes les preguntes de forma curta, clara i concisa. Si algú <|im_end|>\n"] + [prompt] + [response]

    bot.reply_to(message, response)

bot.infinity_polling()

