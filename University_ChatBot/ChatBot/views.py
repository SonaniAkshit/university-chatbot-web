import os
import json
import random
import torch
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.conf import settings
from .models import User
from django.db import DatabaseError
from django.contrib import messages
from django.shortcuts import render, redirect
from werkzeug.security import generate_password_hash
from django.contrib.auth import authenticate, login
from .models import User  # your custom User model
# Import chatbot model and utils
from django.contrib.auth import logout
from django.shortcuts import redirect

from ChatBot_Logic.model import NeuralNet
from ChatBot_Logic.nltk_utils import bag_of_words, tokenize
from django.shortcuts import render, get_object_or_404

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


# Chatbot views
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
    if prob > 0.50:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return JsonResponse({"reply": random.choice(intent["responses"])})
    return JsonResponse({"reply": "I’m not sure I understand. Can you rephrase?"})


def register_view(request):
    if request.method == "POST":
        full_name = request.POST.get("full_name", "").strip()
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip()
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")
        gender = request.POST.get("gender")
        role = request.POST.get("role")

        form_errors = {}

        # Validations
        if not full_name:
            form_errors["full_name"] = "Full name is required."
        if not username:
            form_errors["username"] = "Username is required."
        elif User.objects.filter(username=username).exists():
            form_errors["username"] = "Username already exists."
        if not email:
            form_errors["email"] = "Email is required."
        elif User.objects.filter(email=email).exists():
            form_errors["email"] = "Email already exists."
        if not password:
            form_errors["password"] = "Password is required."
        if password != confirm_password:
            form_errors["confirm_password"] = "Passwords do not match."
        if not gender:
            form_errors["gender"] = "Select a gender."
        if not role:
            form_errors["role"] = "Select a role."

        if form_errors:
            return render(request, "register.html", {"form_errors": form_errors, "form_data": request.POST})

        # Save user with plain password
        try:
            User.objects.create(
                full_name=full_name,
                username=username,
                email=email,
                password=password,  # store actual password
                gender=gender,
                role=role
            )
            messages.success(request, "Registration successful! Please login.")
            return redirect("login")
        except DatabaseError:
            messages.error(request, "Something went wrong. Please try again later.")
            return render(request, "register.html", {"form_data": request.POST})

    return render(request, "register.html")


from django.contrib import messages  # make sure this is imported

def login_view(request):
    if request.method == "POST":
        identifier = request.POST.get("login_identifier").strip()  # username or email
        password = request.POST.get("password")

        try:
            # Look up user by username or email
            try:
                user = User.objects.get(username=identifier)
            except User.DoesNotExist:
                user = User.objects.get(email=identifier)
        except User.DoesNotExist:
            messages.error(request, "Invalid username/email or password.")
            return render(request, "login.html", {"form_data": request.POST})
        except DatabaseError:
            messages.error(request, "Something went wrong. Please try again later.")
            return render(request, "login.html", {"form_data": request.POST})

        # Check password directly
        if user.password != password:
            messages.error(request, "Invalid username/email or password.")
            return render(request, "login.html", {"form_data": request.POST})

        # Login successful: create session
        request.session['user_id'] = user.id
        request.session['username'] = user.username

        # Set welcome message
        messages.success(request, f"Welcome, {user.username}! You are now logged in.")

        return redirect("chatbot_home")

    return render(request, "login.html")


def profile_view(request):
    user_id = request.session.get('user_id')  # session user ID
    if not user_id:
        return redirect('login')  # redirect if not logged in

    user = get_object_or_404(User, id=user_id)

    # If you have a Profile model linked with OneToOneField to User
    profile = getattr(user, 'profile', None)

    context = {
        'user_obj': user,
        'profile': profile
    }

    return render(request, "profile.html", context)

def logout_view(request):
    logout(request)  # Clears the session
    return redirect('login')  # Redirects to login page