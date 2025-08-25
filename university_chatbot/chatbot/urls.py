from django.urls import path
from . import views

urlpatterns = [
    path("", views.chatbot_view, name="chatbot_home"),
    path("get/", views.get_response, name="chatbot_get"),
]
