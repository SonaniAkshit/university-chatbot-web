import os
import json
import random
import torch
from django.db import DatabaseError
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.db import DatabaseError
from django.contrib import messages
from django.contrib.auth import logout
from django.utils import timezone
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

import subprocess
from django.conf import settings

from .models import User, Conversation, AdminUser, PendingQuestion
from ChatBot_Logic.model import NeuralNet
from ChatBot_Logic.nltk_utils import bag_of_words, tokenize


# ====== CHATBOT MODEL LOADING ======

INTENTS_PATH = os.path.join(settings.BASE_DIR, "ChatBot_Logic", "intents.json")
with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)

FILE = os.path.join(settings.BASE_DIR, "ChatBot_Logic", "data.pth")
data = torch.load(FILE, map_location=torch.device("cpu"))

# ====== CHATBOT MODEL LOADING ======

INTENTS_PATH = os.path.join(settings.BASE_DIR, "ChatBot_Logic", "intents.json")
MODEL_FILE = os.path.join(settings.BASE_DIR, "ChatBot_Logic", "data.pth")

# globals that we will reload
intents = None
model = None
all_words = None
tags = None
input_size = None
hidden_size = None
output_size = None
bot_name = "UniBuddy"
CONF_THRESHOLD = 0.50


def load_chatbot_artifacts():
    """
    Load intents.json + data.pth into global variables.
    Call this once at startup and again after retraining.
    """
    global intents, model, all_words, tags, input_size, hidden_size, output_size

    # load intents
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        intents = json.load(f)

    # load trained model data
    data = torch.load(MODEL_FILE, map_location=torch.device("cpu"))

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]

    # rebuild model
    m = NeuralNet(input_size, hidden_size, output_size)
    m.load_state_dict(model_state)
    m.eval()

    model = m
    print("Chatbot artifacts reloaded. Tags:", tags)


# load once when Django starts
load_chatbot_artifacts()



# ====== CHATBOT VIEWS ======

def chatbot_view(request):
    return render(request, "chatbot.html")


def get_response(request):
    """
    Main AJAX endpoint for chatbot.
    - Reads user message
    - Saves user & bot messages into Conversation.messages (single document per user)
    - Runs intent model
    - If high confidence: normal intent reply
    - If low confidence: saves PendingQuestion and sends 'saved for admin' message
    - Returns reply as JSON
    """
    user_msg = request.GET.get("msg", "").strip()
    if not user_msg:
        return JsonResponse({"reply": "Please type something."})

    # Get logged-in user from session
    user_id = request.session.get("user_id")
    user = None
    conv = None

    if user_id:
        user = get_object_or_404(User, id=user_id)

        # One conversation document per user
        conv, created = Conversation.objects.get_or_create(
            user=user,
            defaults={"username": user.username, "messages": []},
        )

        # Append user message to conversation
        msgs = list(conv.messages)
        msgs.append({
            "message": user_msg,
            "is_bot": False,
            "timestamp": timezone.now().isoformat(),
        })
        conv.messages = msgs
        conv.save(update_fields=["messages", "updated_at"])

    # Chatbot intent logic
    sentence_tokens = tokenize(user_msg)
    X = bag_of_words(sentence_tokens, all_words)
    X_tensor = torch.from_numpy(X).unsqueeze(0).to(dtype=torch.float32)

    with torch.no_grad():
        output = model(X_tensor)
        _, predicted = torch.max(output, dim=1)
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()].item()

    tag = tags[predicted.item()]

    # High confidence → normal intent reply
    if prob > CONF_THRESHOLD:
        bot_reply = None
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                bot_reply = random.choice(intent["responses"])
                break
        if bot_reply is None:
            bot_reply = "I’m not sure I understand. Can you rephrase?"

        if user and conv:
            msgs = list(conv.messages)
            msgs.append({
                "message": bot_reply,
                "is_bot": True,
                "timestamp": timezone.now().isoformat(),
            })
            conv.messages = msgs
            conv.save(update_fields=["messages", "updated_at"])

        return JsonResponse({
            "reply": bot_reply,
            "unknown": False,
            "tag": tag,
            "confidence": prob,
        })

    # Low confidence → save as PendingQuestion for admin
    if user:
        PendingQuestion.objects.create(
            user=user,
            username=user.username,
            question_text=user_msg,
            model_tag=tag,
            confidence=prob,
        )

    bot_reply = (
        "Right now I don't know the answer to this question. "
        "I have saved it so the admin can update my knowledge and "
        "I can answer it better in the future."
    )

    if user and conv:
        msgs = list(conv.messages)
        msgs.append({
            "message": bot_reply,
            "is_bot": True,
            "timestamp": timezone.now().isoformat(),
        })
        conv.messages = msgs
        conv.save(update_fields=["messages", "updated_at"])

    return JsonResponse({
        "reply": bot_reply,
        "unknown": True,
        "tag": tag,
        "confidence": prob,
    })


# ====== USER REGISTRATION / AUTH ======

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
            return render(
                request,
                "register.html",
                {"form_errors": form_errors, "form_data": request.POST},
            )

        # Save user with plain password (NOTE: not safe for production)
        try:
            User.objects.create(
                full_name=full_name,
                username=username,
                email=email,
                password=password,
                gender=gender,
                role=role,
            )
            messages.success(request, "Registration successful! Please login.")
            return redirect("login")
        except DatabaseError:
            messages.error(request, "Something went wrong. Please try again later.")
            return render(request, "register.html", {"form_data": request.POST})

    return render(request, "register.html")


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

        # Update status to active
        user.status = "active"
        user.save(update_fields=["status"])

        # Create session
        request.session["user_id"] = user.id
        request.session["username"] = user.username

        messages.success(request, f"Welcome, {user.username}! You are now logged in.")
        return redirect("chatbot_home")

    return render(request, "login.html")


def logout_view(request):
    user_id = request.session.get("user_id")
    if user_id:
        try:
            user = User.objects.get(id=user_id)
            user.status = "inactive"
            user.save(update_fields=["status"])
        except User.DoesNotExist:
            pass

    logout(request)
    return redirect("login")


# ====== PROFILE VIEWS ======

def profile_view(request):
    user_id = request.session.get("user_id")
    if not user_id:
        return redirect("login")

    user = get_object_or_404(User, id=user_id)
    profile = getattr(user, "profile", None)

    context = {
        "user_obj": user,
        "profile": profile,
    }
    return render(request, "profile.html", context)


def update_profile_view(request):
    user_id = request.session.get("user_id")
    if not user_id:
        return redirect("login")

    user = get_object_or_404(User, id=user_id)

    if request.method == "POST":
        full_name = request.POST.get("full_name", "").strip()
        email = request.POST.get("email", "").strip()
        username = request.POST.get("username", "").strip()
        gender = request.POST.get("gender", "").strip()
        role = request.POST.get("role", "").strip()

        form_errors = {}

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
            return render(
                request,
                "profile.html",
                {
                    "user_obj": user,
                    "profile": getattr(user, "profile", None),
                    "form_errors": form_errors,
                },
            )

        try:
            user.full_name = full_name
            user.email = email
            user.username = username
            user.gender = gender
            user.role = role
            user.save()

            messages.success(request, "Profile updated successfully.")
            return redirect("profile")
        except DatabaseError:
            messages.error(request, "Something went wrong. Please try again later.")

    return redirect("profile")


def update_profile_image(request):
    if request.method == "POST" and request.FILES.get("profile_image"):
        user_id = request.session.get("user_id")
        if not user_id:
            messages.error(request, "You need to login first!")
            return redirect("login")

        user = User.objects.get(id=user_id)
        user.profile_image = request.FILES["profile_image"]
        user.save()
        messages.success(request, "Profile image updated successfully!")
        return redirect("profile")

    messages.error(request, "No image selected!")
    return redirect("profile")


# ====== USER CHAT HISTORY VIEWS (Conversation) ======

def chat_history_view(request):
    user_id = request.session.get("user_id")
    if not user_id:
        return redirect("login")

    user = get_object_or_404(User, id=user_id)
    conv = Conversation.objects.filter(user=user).first()

    messages_list = conv.messages if conv else []

    # chat_history is a list of dicts: {message, is_bot, timestamp}
    return render(request, "chat_history.html", {"chat_history": messages_list})


def clear_chat_view(request):
    """
    Optional AJAX endpoint to clear chat, returns JSON.
    """
    user_id = request.session.get("user_id")
    if not user_id:
        return JsonResponse({"status": "failed"}, status=403)

    user = get_object_or_404(User, id=user_id)
    conv = Conversation.objects.filter(user=user).first()
    if conv:
        conv.messages = []
        conv.save(update_fields=["messages", "updated_at"])

    return JsonResponse({"status": "success"})


def clear_chat_history(request):
    """
    Clears chat history and redirects back to chat_history page.
    """
    user_id = request.session.get("user_id")
    if not user_id:
        messages.error(request, "No active session found. Please log in again.")
        return redirect("login")

    user = get_object_or_404(User, id=user_id)
    conv = Conversation.objects.filter(user=user).first()
    if conv:
        conv.messages = []
        conv.save(update_fields=["messages", "updated_at"])

    messages.success(request, "Your chat history has been cleared successfully.")
    return redirect("chat_history")


# ====== ADMIN AUTH & DASHBOARD ======

def admin_login(request):
    if request.method == "POST":
        identifier = request.POST.get("login_identifier", "").strip()
        password = request.POST.get("password")

        try:
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

        if admin.password != password:
            messages.error(request, "Invalid username/email or password.")
            return render(request, "chatbotadmin/login.html", {"form_data": request.POST})

        request.session["admin_id"] = admin.id
        request.session["admin_username"] = admin.username

        messages.success(request, f"Welcome, {admin.username}! You are now logged in.")
        return redirect("admin_panel")

    return render(request, "chatbotadmin/login.html")


def admin_users(request):
    users = User.objects.all().order_by("-joined_date")
    context = {
        "users": users,
        "students_count": User.objects.filter(role="student").count(),
        "parents_count": User.objects.filter(role="parent").count(),
        "faculty_count": User.objects.filter(role="faculty").count(),
        "others_count": User.objects.filter(role__in=["staff", "other"]).count(),
    }
    return render(request, "chatbotadmin/users.html", context)


def admin_dashboard(request):
    if not request.session.get("admin_username"):
        return redirect("admin_login")

    # User counts by role
    students_count = User.objects.filter(role="student").count()
    parents_count = User.objects.filter(role="parent").count()
    faculty_count = User.objects.filter(role="faculty").count()
    others_count = User.objects.filter(role__in=["staff", "other"]).count()

    total_count = students_count + parents_count + faculty_count + others_count

    # Recent users (last 5)
    recent_users = User.objects.order_by("-joined_date")[:5]

    # Today registered users (djongo-safe: filter in Python, not with __date)
    today = timezone.now().date()
    all_users = User.objects.order_by("-joined_date")
    today_users = [u for u in all_users if u.joined_date and u.joined_date.date() == today]

    # Total chats
    total_chats = Conversation.objects.count()

    # Djongo-safe pending questions count (no boolean filter in SQL)
    all_questions = list(PendingQuestion.objects.all())
    resolved_count = sum(1 for q in all_questions if q.is_resolved)
    pending_questions_count = len(all_questions) - resolved_count

    context = {
        "students_count": students_count,
        "parents_count": parents_count,
        "faculty_count": faculty_count,
        "others_count": others_count,
        "total_count": total_count,
        "recent_users": recent_users,
        "today_users": today_users,          # now a plain list
        "total_chats": total_chats,
        "pending_questions_count": pending_questions_count,
    }
    return render(request, "chatbotadmin/index.html", context)

def admin_logout(request):
    request.session.pop("admin_id", None)
    request.session.pop("admin_username", None)
    messages.success(request, "You have been successfully logged out.")
    return redirect("admin_login")


# ====== ADMIN CHAT HISTORY (Conversation) ======

def admin_chat_history(request):
    if not request.session.get("admin_username"):
        return redirect("admin_login")

    users = User.objects.all().order_by("-joined_date")
    chat_usernames = Conversation.objects.values_list("username", flat=True)

    return render(
        request,
        "chatbotadmin/chat_history.html",
        {"users": users, "chat_usernames": list(chat_usernames)},
    )


def get_user_chat_history(request, user_id):
    """
    Returns JSON chat history for given user for admin modal.
    Uses Conversation.messages.
    """
    user = get_object_or_404(User, id=user_id)
    conv = Conversation.objects.filter(user=user).first()
    messages_list = conv.messages if conv else []

    return JsonResponse(
        {"username": user.username, "full_name": user.full_name, "chats": messages_list}
    )


# ====== ADMIN PENDING QUESTIONS ======
def admin_pending_questions(request):
    if not request.session.get("admin_username"):
        return redirect("admin_login")

    try:
        qs = PendingQuestion.objects.all().order_by("-created_at")
        pending = [q for q in qs if not q.is_resolved]
    except DatabaseError:
        messages.error(request, "Error loading pending questions.")
        pending = []

    grouped = {}
    for q in pending:
        key = q.username or "Unknown"
        if key not in grouped:
            grouped[key] = {
                "username": key,
                "email": q.user.email if q.user else "",
                "count": 0,
                "questions": []
            }
        grouped[key]["count"] += 1
        grouped[key]["questions"].append({
            "id": q.id,
            "text": q.question_text,
            "created_at": q.created_at.strftime("%Y-%m-%d %H:%M")
        })

    pending_data = list(grouped.values())

    return render(
        request,
        "chatbotadmin/pending_questions.html",
        {"pending_data": pending_data},
    )


@require_POST
def resolve_pending_question(request, pk):
    """
    Mark a pending question as resolved after admin has updated intents/train.
    """
    if not request.session.get("admin_username"):
        return redirect("admin_login")

    pending = get_object_or_404(PendingQuestion, pk=pk)
    admin_note = request.POST.get("admin_note", "").strip()

    pending.admin_note = admin_note
    pending.is_resolved = True
    pending.save()

    messages.success(request, "Pending question marked as resolved.")
    return redirect("admin_pending_questions")

def admin_intent_manager(request):
    if not request.session.get("admin_username"):
        return redirect("admin_login")

    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    intents_list = data.get("intents", [])

    selected_tag = request.GET.get("tag")
    # we just pass the tag; frontend will pick intent from intents_list
    return render(
        request,
        "chatbotadmin/intent_manager.html",
        {
            "intents": intents_list,
            "intents_json": json.dumps(intents_list),
            "selected_tag": json.dumps(selected_tag) if selected_tag else "null",
        },
    )

@csrf_exempt
@require_POST
def admin_save_intent(request):
    body = json.loads(request.body)

    tag = body.get("tag")
    patterns = body.get("patterns", [])
    responses = body.get("responses", [])
    question_id = body.get("question_id")

    intents_path = os.path.join(settings.BASE_DIR, "ChatBot_Logic", "intents.json")

    with open(intents_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    existing = None
    for item in data["intents"]:
        if item["tag"].lower() == tag.lower():
            existing = item
            break

    if existing:
        existing["patterns"].extend(patterns)
        existing["responses"].extend(responses)
    else:
        data["intents"].append({
            "tag": tag,
            "patterns": patterns,
            "responses": responses
        })

    with open(intents_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    if question_id:
        q = get_object_or_404(PendingQuestion, id=question_id)
        q.is_resolved = True
        q.admin_note = f"Mapped to intent: {tag}"
        q.save()

    return JsonResponse({"success": True})

def admin_retrain_model(request):
    if not request.session.get("admin_username"):
        return redirect("admin_login")

    try:
        # run training script
        train_script = os.path.join(settings.BASE_DIR, "ChatBot_Logic", "train.py")
        subprocess.run(
            ["python", train_script],
            check=True,
            cwd=os.path.join(settings.BASE_DIR, "ChatBot_Logic"),
        )

        # reload model + intents into memory
        load_chatbot_artifacts()

        messages.success(request, "Model retrained and reloaded successfully!")
    except Exception as e:
        messages.error(request, f"Training failed: {e}")

    return redirect("admin_intent_manager")

def admin_intents_browser(request):
    if not request.session.get("admin_username"):
        return redirect("admin_login")

    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    intents_list = data.get("intents", [])

    return render(
        request,
        "chatbotadmin/intents_browser.html",
        {
            "intents": intents_list,
            "intents_json": json.dumps(intents_list),  # for JS
        },
    )

from django.views.decorators.http import require_POST

@require_POST
def admin_delete_intent(request):
    if not request.session.get("admin_username"):
        return JsonResponse({"error": "Unauthorized"}, status=403)

    try:
        body = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    tag = body.get("tag")
    if not tag:
        return JsonResponse({"error": "Missing tag"}, status=400)

    if not os.path.exists(INTENTS_PATH):
        return JsonResponse({"error": "Intents file not found"}, status=500)

    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    intents = data.get("intents", [])
    new_intents = [i for i in intents if i.get("tag") != tag]

    if len(new_intents) == len(intents):
        return JsonResponse({"error": "Tag not found"}, status=404)

    data["intents"] = new_intents

    with open(INTENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return JsonResponse({"success": True})
