# 📘University Chatbot (Django + PyTorch NLP) - UniBuddy

UniBuddy is a **web-based chatbot** built with **Django** and **PyTorch**. It is designed to answer common university-related queries such as admissions, deadlines, scholarships, and campus information.

The chatbot uses a simple **feedforward neural network** trained on intents (questions and answers) with **NLTK for preprocessing** and **PyTorch for training**.

---

## 🚀 Features

* Trainable NLP model using PyTorch
* User-friendly **ChatGPT-like web interface**
* Supports **dynamic Q\&A** based on intents
* Built-in **clear conversation button**
* Easily extendable with new intents and responses

---

## ⚙️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/SonaniAkshit/university-chatbot-web.git
cd university-chatbot-web
```

### 2. Create virtual environment

```bash
python -m venv myenv
myenv\Scripts\activate      # On Windows
```

<!-- ### 3. Install dependencies

```bash
pip install -r requirements.txt
``` -->

---

## 🧠 Training the Model

1. Modify **intents.json** to add your own intents and responses.
2. Run training:

   ```bash
   python train.py
   ```
3. This will create/update the file:

   ```
   data.pth
   ```

   which contains trained model weights.

---

## 🌐 Running the Web App

1. Start Django server:

   ```bash
   cd university_chatbot
   python manage.py runserver
   ```
2. Open browser:

   ```
   http://127.0.0.1:8000/
   ```
3. Chat with UniBuddy 🎓

---

## 📝 Example Usage

**User:**

> When is the admission deadline?

**UniBuddy:**

> The admission deadline is June 30th.

**User:**

> Tell me about scholarships.

**UniBuddy:**

> We offer merit-based and need-based scholarships. Please visit the scholarships section on our website.

---

## 📦 Requirements

Main libraries:

```
Django
torch
nltk
numpy
```

<!-- Install all via:

```bash
pip install -r requirements.txt
``` -->

---

## 🙌 Contributing

* Add new intents to **intents.json**
* Retrain the model with `python train.py`
* Improve UI in **templates/chatbot/**

---

## 📜 License

This project is open-source under the MIT License.

---