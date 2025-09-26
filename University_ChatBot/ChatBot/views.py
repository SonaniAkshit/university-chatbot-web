import os
import json
import random
import torch
from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings

from ChatBot_Logic.model import NeuralNet
from ChatBot_Logic.nltk_utils import bag_of_words, tokenize

# Load intents
INTENTS_PATH = os.path.join(settings.BASE_DIR, "ChatBot_Logic", "intents.json")
with open(INTENTS_PATH, "r", encoding='utf-8') as f:
    intents = json.load(f)

# Load model
FILE = os.path.join(settings.BASE_DIR, "ChatBot_Logic", "data.pth")
data = torch.load(FILE, map_location=torch.device("cpu"))

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "UniBuddy"

def chatbot_view(request):
    return render(request, "chatbot.html")

def get_response(request):
    user_msg = request.GET.get("msg", "").strip()
    if not user_msg:
        return JsonResponse({"reply": "Please type something."})

    sentence_tokens = tokenize(user_msg)
    X = bag_of_words(sentence_tokens, all_words)
    X_tensor = torch.from_numpy(X).unsqueeze(0).to(dtype=torch.float32)

    with torch.no_grad():
        output = model(X_tensor)
        _, predicted = torch.max(output, dim=1)
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()].item()

    tag = tags[predicted.item()]
    # Lower the confidence threshold to make the chatbot less rigid.
    # A value of 0.50 is a good balance between accuracy and responsiveness.
    if prob > 0.50:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return JsonResponse({"reply": random.choice(intent["responses"])})
    # For now return safe fallback
    return JsonResponse({"reply": "I’m not sure I understand. Can you rephrase?"})

def register_view(request):
    return render(request, "register.html")

def login_view(request):
    return render(request, "login.html")

def profile_view(request):
    return render(request, "profile.html")
