import os
import telebot
from dotenv import load_dotenv
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format

# Load environment variables
load_dotenv()

# Initialize the Telegram bot
bot = telebot.TeleBot(os.environ.get('BOT_TOKEN'))

# Import necessary libraries for the model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("pauhidalgoo/cucafera-chat")
model = AutoModelForCausalLM.from_pretrained("pauhidalgoo/cucafera-chat")
# Set up the chat format with default 'chatml' format
model, tokenizer = setup_chat_format(model, tokenizer)

print(tokenizer.eos_token_id)
print(tokenizer.pad_token_id)

# Generation configuration
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

# Dictionary to store conversation history for each user
user_conversations = {}

@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, "Hola, amb què et puc ajudar?")
    # Initialize an empty conversation history for the user
    user_conversations[message.chat.id] = []

@bot.message_handler(commands=['reset', 'reseteja'])
def send_welcome(message):
    bot.reply_to(message, "Ja no recordo res.")
    # Initialize an empty conversation history for the user
    user_conversations[message.chat.id] = []

@bot.message_handler(commands=['redo', 'repeteix'])
def send_welcome(message):
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

    # Extract the assistant's reply from the generated text
    response = response.strip()
    
    # Add the assistant's response to the conversation history
    conversation.append(response)
    
    # Update the stored conversation history
    user_conversations[message.chat.id] = conversation

    # Send the response back to the user
    bot.reply_to(message, response)


@bot.message_handler(func=lambda msg: True)
def respond(message):
    # Retrieve the conversation history for the user
    conversation = user_conversations.get(message.chat.id, ["<|im_start|>system \n Respon a totes les preguntes de forma curta, clara i concisa. <|im_end|>\n"])
    # Format the new prompt with the required template
    prompt = f"<|im_start|>user\n{message.text}<|im_end|><|im_start|>assistant\n"

    # Add the prompt to the conversation history
    conversation.append(prompt)

    # Join the conversation history into a single input string
    input_text = "".join(conversation)

    # Tokenize and generate the model's response
    tokens = np.array(tokenizer.encode(input_text, return_tensors='pt', truncation="only_first", add_special_tokens=False))
    tokens = tokens[-2048:]
    

    tokens = torch.tensor(tokens, dtype=torch.long)

    prompt_length = tokens.shape[1]

    a = model.generate(tokens, genconf)
    response = tokenizer.decode(a[0][prompt_length:], skip_special_tokens=True)

    # Extract the assistant's reply from the generated text
    response = response.strip()
    
    # Add the assistant's response to the conversation history
    conversation.append(response)
    
    # Update the stored conversation history
    user_conversations[message.chat.id] = ["<|im_start|>system \n Respon a totes les preguntes de forma curta, clara i concisa. Si algú <|im_end|>\n"] + [prompt] + [response]

    # Send the response back to the user
    bot.reply_to(message, response)

bot.infinity_polling()

