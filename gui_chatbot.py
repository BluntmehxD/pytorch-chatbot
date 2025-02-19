import random
import json
import torch
import tkinter as tk
from tkinter import scrolledtext

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('pytorch-chatbot/intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "pytorch-chatbot/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I do not understand..."

def send():
    msg = entry_box.get("1.0", 'end-1c').strip()
    entry_box.delete("1.0", tk.END)

    if msg != '':
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, "You: " + msg + '\n\n')
        chat_log.config(foreground="#442265", font=("Verdana", 12))
        
        res = get_response(msg)
        chat_log.insert(tk.END, bot_name + ": " + res + '\n\n')
            
        chat_log.config(state=tk.DISABLED)
        chat_log.yview(tk.END)

base = tk.Tk()
base.title("Chatbot")
base.geometry("400x500")
base.resizable(width=False, height=False)

chat_log = scrolledtext.ScrolledText(base, bd=0, bg="white", height="8", width="50", font="Arial", state=tk.DISABLED)
chat_log.place(x=6, y=6, height=386, width=370)

entry_box = tk.Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
entry_box.place(x=6, y=401, height=90, width=265)

send_button = tk.Button(base, text="Send", command=send, width="12", height=5, bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff', font=("Arial", 12))
send_button.place(x=276, y=401, height=90, width=100)

base.mainloop()
