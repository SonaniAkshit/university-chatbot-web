from django.urls import path
from . import views

urlpatterns = [
    path("", views.chatbot_view, name="chatbot_home"),
    path("get/", views.get_response, name="chatbot_get"),
    path("register/", views.register_view, name="register"),
    path('login/', views.login_view, name='login'),
    path('profile/', views.profile_view, name='profile'),
    path('logout/', views.logout_view, name='logout'),
    path("update-profile/", views.update_profile_view, name="update_profile"),
    path('update-profile-image/', views.update_profile_image, name='update_profile_image'),

]
