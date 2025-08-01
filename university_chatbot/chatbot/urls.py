from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_view, name='chat'),  # Handles /chatbot/
    path('webhook', views.webhook, name='webhook'),  # Handles /chatbot/webhook
]