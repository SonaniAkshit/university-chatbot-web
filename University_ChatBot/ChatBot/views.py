import os
import json
import random
import torch
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.utils import timezone
from django.conf import settings
from .models import User, ChatMessage, AdminUser
from django.shortcuts import get_object_or_404
from django.db import DatabaseError
from django.contrib import messages
from django.shortcuts import render, redirect
from werkzeug.security import generate_password_hash
from django.contrib.auth import authenticate, login
from .models import User  # your custom User model
# Import chatbot model and utils
from django.contrib.auth import logout
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_exempt
from django.db import DatabaseError

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
        identifier = request.POST.get("login_identifier", "").strip()  # username or email
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

        # Check password directly (⚠️ You should hash later, this is unsafe in production)
        if user.password != password:
            messages.error(request, "Invalid username/email or password.")
            return render(request, "login.html", {"form_data": request.POST})

        # ✅ Update status to active
        user.status = "active"
        user.save(update_fields=["status"])

        # Login successful: create session
        request.session["user_id"] = user.id
        request.session["username"] = user.username

        # Welcome message
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
    user_id = request.session.get("user_id")  # Get current logged-in user id
    if user_id:
        try:
            user = User.objects.get(id=user_id)
            user.status = "inactive"  # Set status to inactive
            user.save(update_fields=["status"])
        except User.DoesNotExist:
            pass  # User not found, ignore

    logout(request)  # Clears the session
    return redirect("login")  # Redirects to login page

def update_profile_view(request):
    user_id = request.session.get('user_id')
    if not user_id:
        return redirect('login')

    user = get_object_or_404(User, id=user_id)

    if request.method == "POST":
        full_name = request.POST.get("full_name", "").strip()
        email = request.POST.get("email", "").strip()
        username = request.POST.get("username", "").strip()
        gender = request.POST.get("gender", "").strip()
        role = request.POST.get("role", "").strip()

        form_errors = {}

        # Basic validations
        if not full_name:
            form_errors["full_name"] = "Full name is required."
        if not email:
            form_errors["email"] = "Email is required."
        elif User.objects.filter(email=email).exclude(id=user.id).exists():
            form_errors["email"] = "Email already exists."
        if not username:
            form_errors["username"] = "Username is required."
        elif User.objects.filter(username=username).exclude(id=user.id).exists():
            form_errors["username"] = "Username already exists."
        if not gender:
            form_errors["gender"] = "Gender is required."
        if not role:
            form_errors["role"] = "Role is required."

        if form_errors:
            return render(request, "profile.html", {
                "user_obj": user,
                "profile": getattr(user, 'profile', None),
                "form_errors": form_errors
            })

        try:
            user.full_name = full_name
            user.email = email
            user.username = username
            user.gender = gender
            user.role = role
            user.save()

            messages.success(request, "Profile updated successfully.")
            return redirect("profile")  # or your profile page url name
        except DatabaseError:
            messages.error(request, "Something went wrong. Please try again later.")

    return redirect("profile")

def update_profile_image(request):
    if request.method == "POST" and request.FILES.get("profile_image"):
        user_id = request.session.get('user_id')
        if not user_id:
            messages.error(request, "You need to login first!")
            return redirect('login')
        
        user = User.objects.get(id=user_id)
        user.profile_image = request.FILES['profile_image']
        user.save()
        messages.success(request, "Profile image updated successfully!")
        return redirect('profile')
    else:
        messages.error(request, "No image selected!")
        return redirect('profile')
    
def get_response(request):
    user_msg = request.GET.get("msg", "").strip()
    if not user_msg:
        return JsonResponse({"reply": "Please type something."})

    # Get logged-in user
    user_id = request.session.get('user_id')
    user = None
    username = None
    if user_id:
        user = get_object_or_404(User, id=user_id)
        username = user.username
        # Save user message
        ChatMessage.objects.create(
            user=user,
            username=username,
            message=user_msg,
            is_bot=False
        )

    # Chatbot logic
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
                bot_reply = random.choice(intent["responses"])
                # Save bot response
                if user:
                    ChatMessage.objects.create(
                        user=user,
                        username=username,
                        message=bot_reply,
                        is_bot=True
                    )
                return JsonResponse({"reply": bot_reply})

    bot_reply = "I’m not sure I understand. Can you rephrase?"
    if user:
        ChatMessage.objects.create(
            user=user,
            username=username,
            message=bot_reply,
            is_bot=True
        )

    return JsonResponse({"reply": bot_reply})

def chat_history_view(request):
    username = request.session.get('username')
    if not username:
        return redirect('login')  # redirect if not logged in

    # Fetch chat history for this session username
    chat_history = ChatMessage.objects.filter(username=username).order_by('timestamp')

    return render(request, "chat_history.html", {"chat_history": chat_history})



def clear_chat_view(request):
    user_id = request.session.get('user_id')
    if user_id:
        user = get_object_or_404(User, id=user_id)
        ChatMessage.objects.filter(user=user).delete()
        return JsonResponse({"status": "success"})
    return JsonResponse({"status": "failed"}, status=403)

def clear_chat_history(request):
    username = request.session.get('username')
    if username:
        ChatMessage.objects.filter(username=username).delete()
    return redirect('chat_history')


# def chatbot_admin_index(request):
#     return render(request, 'chatbotadmin/index.html')


def admin_login(request):
    if request.method == "POST":
        identifier = request.POST.get("login_identifier", "").strip()  # username or email
        password = request.POST.get("password")

        try:
            # Lookup admin by username or email
            try:
                admin = AdminUser.objects.get(username=identifier)
            except AdminUser.DoesNotExist:
                admin = AdminUser.objects.get(email=identifier)
        except AdminUser.DoesNotExist:
            messages.error(request, "Invalid username/email or password.")
            return render(request, "chatbotadmin/login.html", {"form_data": request.POST})
        except DatabaseError:
            messages.error(request, "Something went wrong. Please try again later.")
            return render(request, "chatbotadmin/login.html", {"form_data": request.POST})

        # Check password (⚠️ Plain text for now; hash recommended in production)
        if admin.password != password:
            messages.error(request, "Invalid username/email or password.")
            return render(request, "chatbotadmin/login.html", {"form_data": request.POST})

        # Login successful: set session
        request.session["admin_id"] = admin.id
        request.session["admin_username"] = admin.username

        # Success message
        messages.success(request, f"Welcome, {admin.username}! You are now logged in.")

        return redirect("admin_panel")  # Redirect to your admin dashboard

    return render(request, "chatbotadmin/login.html")

def admin_users(request):
    users = User.objects.all().order_by('-joined_date')
    context = {
        'users': users,
        'students_count': User.objects.filter(role='student').count(),
        'parents_count': User.objects.filter(role='parent').count(),
        'faculty_count': User.objects.filter(role='faculty').count(),
        'others_count': User.objects.filter(role__in=['staff','other']).count(),
    }
    return render(request, 'chatbotadmin/users.html', context)
    # return render(request, "chatbotadmin/users.html", {"users": users})

def admin_dashboard(request):

    # Check if admin is logged in
    if not request.session.get('admin_username'):
        return redirect('admin_login')  # Redirect to login if session not found

    admin_username = request.session.get('admin_username')
    # Proceed with your admin panel logic
    
    # Count by category
    students_count = User.objects.filter(role='student').count()
    parents_count = User.objects.filter(role='parent').count()
    faculty_count = User.objects.filter(role='faculty').count()
    others_count = User.objects.filter(role__in=['staff','other']).count()
    
    total_count = students_count + parents_count + faculty_count + others_count

    # Recent users - last 5 registered
    recent_users = User.objects.order_by('-joined_date')[:5]

    context = {
        'students_count': students_count,
        'parents_count': parents_count,
        'faculty_count': faculty_count,
        'others_count': others_count,
        'total_count': total_count,
        'recent_users': recent_users,
    }
    return render(request, 'chatbotadmin/index.html', context)

def admin_logout(request):
    # Clear admin session
    request.session.pop('admin_id', None)
    request.session.pop('admin_username', None)
    
    # Optional: Add a message
    messages.success(request, "You have been successfully logged out.")
    
    # Redirect to admin login page
    return redirect('admin_login')