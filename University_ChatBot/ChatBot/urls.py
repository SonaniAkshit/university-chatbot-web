from django.urls import path
from . import views

urlpatterns = [

    # ================================
    # USER ROUTES
    # ================================
    path("", views.chatbot_view, name="chatbot_home"),
    path("get/", views.get_response, name="get_response"),

    # chat history using Conversation model
    path("history/", views.chat_history_view, name="chat_history"),
    path("clear/", views.clear_chat_view, name="clear_chat"),
    path("clear_history/", views.clear_chat_history, name="clear_chat_history"),

    # auth
    path("register/", views.register_view, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),

    # profile
    path("profile/", views.profile_view, name="profile"),
    path("profile/update/", views.update_profile_view, name="update_profile"),
    path("profile/update-image/", views.update_profile_image, name="update_profile_image"),


    # ================================
    # ADMIN ROUTES
    # ================================
    path("adminpanel/", views.admin_dashboard, name="admin_panel"),
    path("adminlogin/", views.admin_login, name="admin_login"),
    path("adminlogout/", views.admin_logout, name="admin_logout"),

    path("adminusers/", views.admin_users, name="admin_users"),

    # admin chat history (Conversation model)
    path("admin/chat-history/", views.admin_chat_history, name="admin_chat_history"),
    path("admin/get_user_chat_history/<int:user_id>/", views.get_user_chat_history, name="get_user_chat_history"),


    # ================================
    # ADMIN – PENDING QUESTIONS
    # ================================
    path("admin/pending-questions/", views.admin_pending_questions, name="admin_pending_questions"),
    path("admin/pending-questions/<int:pk>/resolve/", views.resolve_pending_question, name="resolve_pending_question"),

    path("admin/intent-manager/", views.admin_intent_manager, name="admin_intent_manager"),
    path("admin/intent-manager/<int:question_id>/", views.admin_intent_manager, name="admin_edit_intent"),

    path("admin/save-intent/", views.admin_save_intent, name="admin_save_intent"),
    path("admin/retrain-model/", views.admin_retrain_model, name="admin_retrain_model"),
    path("admin/intents-browser/", views.admin_intents_browser, name="admin_intents_browser"),
    path("admin/delete-intent/", views.admin_delete_intent, name="admin_delete_intent"),



]
