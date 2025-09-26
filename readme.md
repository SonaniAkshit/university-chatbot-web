# 🎓 University Chatbot (UniBuddy)

UniBuddy is a web-based chatbot built with **Django** and **PyTorch**.  
It helps students and visitors interact with university-related information in a simple conversational way.  
The chatbot is trained on predefined intents and responses using a neural network.

---

## 🚀 Features
- Chat with a bot about university-related queries (admissions, courses, campus info, etc.).
- Django-based web interface with Bootstrap styling.
- MongoDB integration for data storage (via Djongo).
- NLP preprocessing with **NLTK**.
- Neural network model built and trained using **PyTorch**.

---

## 🛠️ Tech Stack
- **Backend:** Django 3.2, Djongo, MongoDB
- **Frontend:** HTML, Bootstrap 5
- **ML/NLP:** PyTorch, NLTK
- **Database:** MongoDB

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/university-chatbot.git
cd university-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv myenv
source myenv/bin/activate    # Linux/Mac
myenv\Scripts\activate       # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup NLTK Data

```python
import nltk
nltk.download('punkt')
```

### 5. Configure MongoDB

Make sure MongoDB is running locally or update `settings.py`:

```python
DATABASES = {
    'default': {
        'ENGINE': 'djongo',
        'NAME': 'ChatBot_DB',
        'CLIENT': {
            'host': 'mongodb://localhost:27017',
        }
    }
}
```

### 6. Apply Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### 7. Train the Chatbot Model

```bash
python train.py
```

This generates `data.pth` with the trained model.

### 8. Run the Development Server

```bash
python manage.py runserver
```

Visit 👉 `http://127.0.0.1:8000`

---

## 💬 Example Console Chat

```bash
python chat.py
```

```
You: hi
UniBuddy: Hello! How can I help you today?
```

---

## ✅ To-Do

* [ ] Add more intents for detailed university FAQs
* [ ] Enhance UI with chat bubbles
* [ ] Deploy to cloud (Heroku/Render)

---