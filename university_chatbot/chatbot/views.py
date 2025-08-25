import os
import json
import random
import torch
from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings

from bot_core.model import NeuralNet
from bot_core.nltk_utils import bag_of_words, tokenize

# Load intents
with open(os.path.join(settings.BASE_DIR, "bot_core", "intents.json"), "r") as f:
    intents = json.load(f)

# Load trained model
FILE = os.path.join(settings.BASE_DIR, "bot_core", "data.pth")
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
    user_msg = request.GET.get("msg")
    if not user_msg:
        return JsonResponse({"reply": "Please type something."})

    sentence = tokenize(user_msg)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).unsqueeze(0)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.7:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return JsonResponse({"reply": random.choice(intent["responses"])})

    return JsonResponse({"reply": "I’m not sure I understand. Can you rephrase?"})
