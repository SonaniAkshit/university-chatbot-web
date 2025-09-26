import random
import json
import torch
import numpy as np

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE, map_location=device)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "UniBuddy"

print("Let's chat! (type 'quit' to exit)")
while True:
    sentence = input("You: ")
    if sentence.lower() in ("quit", "exit"):
        break

    sentence_tokens = tokenize(sentence)
    X = bag_of_words(sentence_tokens, all_words)
    X = np.expand_dims(X, axis=0)  # shape (1, input_size)
    X_tensor = torch.from_numpy(X).to(dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(X_tensor)
        _, predicted = torch.max(output, dim=1)
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()].item()

    tag = tags[predicted.item()]
    if prob > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                break
    else:
        print(f"{bot_name}: I do not understand... (confidence {prob:.2f})")
