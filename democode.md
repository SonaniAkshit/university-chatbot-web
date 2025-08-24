Perfect 👍 let’s lay this out clearly.
We’ll keep all your existing chatbot code (model, training, utils, intents.json, data.pth) but organize it inside a Django project so the bot runs from a web interface.

---

## 📂 Project Structure

```bash
university_chatbot/
│
├── manage.py
├── requirements.txt
│
├── university_chatbot/              # main Django project folder
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── chatbot/                         # Django app for the chatbot
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── migrations/
│   ├── models.py     # (not used for now)
│   ├── views.py      # Django views (chat API + web UI)
│   ├── urls.py
│   ├── templates/
│   │   └── chatbot/
│   │       └── chat.html   # frontend for the chatbot
│   └── static/
│       └── chatbot/
│           └── style.css   # optional styling
│
├── bot_core/                        # keep your PyTorch chatbot code here
│   ├── __init__.py
│   ├── model.py
│   ├── nltk_utils.py
│   ├── train.py
│   ├── chat.py
│   ├── intents.json
│   └── data.pth
```

---

## 🔹 Files Setup

### `chatbot/urls.py`

```python
from django.urls import path
from . import views

urlpatterns = [
    path("", views.chat_page, name="chat_page"),
    path("api/", views.chat_api, name="chat_api"),
]
```

### `university_chatbot/urls.py`

```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("chat/", include("chatbot.urls")),
]
```

---

### `chatbot/views.py`

Here we load the trained model (`data.pth`) and wrap the `get_response` logic.

```python
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import torch
from bot_core.model import NeuralNet
from bot_core.nltk_utils import bag_of_words, tokenize

# Load intents and trained model
with open("bot_core/intents.json", "r") as f:
    intents = json.load(f)

FILE = "bot_core/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).unsqueeze(0)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return f"{intent['responses'][0]}"
    return "I do not understand..."

# Django views
def chat_page(request):
    return render(request, "chatbot/chat.html")

@csrf_exempt
def chat_api(request):
    if request.method == "POST":
        user_message = request.POST.get("message")
        bot_reply = get_response(user_message)
        return JsonResponse({"reply": bot_reply})
```

---

### `chatbot/templates/chatbot/chat.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>University FAQ Chatbot</title>
    <style>
        body { font-family: Arial; margin: 30px; }
        #chat-box { border: 1px solid #ccc; padding: 10px; width: 400px; height: 400px; overflow-y: auto; }
        input { width: 300px; padding: 5px; }
        button { padding: 5px 10px; }
    </style>
</head>
<body>
    <h2>🎓 University FAQ Chatbot</h2>
    <div id="chat-box"></div>
    <form id="chat-form">
        <input type="text" id="user-input" placeholder="Ask me anything..." />
        <button type="submit">Send</button>
    </form>

    <script>
        const chatBox = document.getElementById("chat-box");
        document.getElementById("chat-form").onsubmit = async (e) => {
            e.preventDefault();
            let msg = document.getElementById("user-input").value;
            chatBox.innerHTML += `<p><b>You:</b> ${msg}</p>`;
            let res = await fetch("/chat/api/", {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: "message=" + encodeURIComponent(msg),
            });
            let data = await res.json();
            chatBox.innerHTML += `<p><b>Bot:</b> ${data.reply}</p>`;
            chatBox.scrollTop = chatBox.scrollHeight;
            document.getElementById("user-input").value = "";
        };
    </script>
</body>
</html>
```

---

## 🔹 Run the Project

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

Open: **[http://127.0.0.1:8000/chat/](http://127.0.0.1:8000/chat/)** 🚀

---

👉 Now your PyTorch chatbot runs inside a Django web app with a simple UI.

Do you want me to also add a **ready-made `requirements.txt` for Django + PyTorch + NLTK** so you don’t miss dependencies?

Yes 👍 this structure and code will work, but let me explain clearly so you know what’s needed:

* **Your existing chatbot code** (`model.py`, `nltk_utils.py`, `train.py`, `intents.json`, `data.pth`) is unchanged — it just moves into the `bot_core/` folder.
* The **Django part** (`chatbot/` app + project setup) acts like a wrapper around it.
* `views.py` loads your trained model once, then every time the user types something, Django passes the message to `get_response()` (your chatbot logic).
* The **chat.html** page is a minimal frontend that calls `/chat/api/` via AJAX and prints the response.

If you set it up exactly as written:

1. Install dependencies (`Django`, `torch`, `nltk`).
2. Train your model with `python bot_core/train.py` so `data.pth` exists.
3. Run `python manage.py runserver`.
4. Open `http://127.0.0.1:8000/chat/`.

You’ll see a working chatbot in your browser 🎉.

👉 The only thing you might need to tweak is the **path to files** (`intents.json`, `data.pth`). I assumed they’re in `bot_core/`, but if you place them elsewhere, just adjust the file paths in `views.py`.

Do you want me to refine it into a **step-by-step setup guide** so you can follow without confusion?
