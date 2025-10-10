from django.urls import path
from . import views

urlpatterns = [

    #user
    path('', views.chatbot_view, name='chatbot_home'),
    path('get/', views.get_response, name='get_response'),
    path('history/', views.chat_history_view, name='chat_history'),
    path('clear/', views.clear_chat_view, name='clear_chat'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile_view, name='profile'),
    path('profile/update/', views.update_profile_view, name='update_profile'),
    path('profile/update-image/', views.update_profile_image, name='update_profile_image'),
    path('clear_history/', views.clear_chat_history, name='clear_chat_history'),

    #admin
    path('adminpanel/', views.admin_dashboard, name='admin_panel'),
    path('adminlogin/', views.admin_login, name='admin_login'),
    path('adminusers/', views.admin_users, name='admin_users'),
    path('adminlogout/', views.admin_logout, name='admin_logout'),
    path("chat-history/", views.admin_chat_history, name="admin_chat_history"),
    path("admin/get_user_chat_history/<int:user_id>/", views.get_user_chat_history, name="get_user_chat_history"),

]
