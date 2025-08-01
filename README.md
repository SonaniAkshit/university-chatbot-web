# University Chatbot with NLP and Animations

A Django-based chatbot for university admission queries using TF-IDF for keyword-based NLP and a JSON data store. Features a Gemini-inspired animated chat interface.

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run migrations: `python manage.py migrate`
4. Collect static files: `python manage.py collectstatic`
5. Start server: `python manage.py runserver`

## Features
- Handles admission queries (eligibility, application process, deadlines, documents) using NLP.
- Professional, animated chat interface with slide-in messages and typing indicator.
- Backend powered by Django with TF-IDF for keyword matching.
- Responses stored in `responses.json` for easy updates.

## Requirements
- Python 3.8+
- 2GB RAM (minimum for TF-IDF processing)
- Stable internet for dependency installation

## Future Improvements
- Add database for dynamic data.
- Support more university services (fees, courses, library).
- Improve NLP with fine-tuned models or embeddings.
- Enhance animations with more interactive elements.

## Notes
- Uses TF-IDF and cosine similarity for intent matching.
- Suitable for MCA 3rd-semester project and 4th-semester internship.
- Deploy with Render or Heroku; use ngrok for local testing.

## Project Structure

```
university_chatbot/
├── manage.py
├── university_chatbot/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   ├── asgi.py
├── chatbot/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── migrations/
│   │   ├── __init__.py
│   ├── models.py
│   ├── tests.py
│   ├── views.py
│   ├── urls.py
│   ├── nlp/
│   │   ├── chatbot_logic.py
│   │   ├── responses.json
│   ├── static/
│   │   ├── css/
│   │   │   ├── chat.css
│   │   ├── js/
│   │   │   ├── chat.js
│   ├── templates/
│   │   ├── chat.html
├── requirements.txt
├── README.md
```