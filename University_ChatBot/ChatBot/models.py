from django.db import models
from django.utils import timezone


# ===========================
# USER MODEL
# ===========================

class User(models.Model):
    full_name = models.CharField(max_length=100)
    username = models.CharField(max_length=50, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=255)

    gender = models.CharField(
        max_length=10,
        choices=[("male", "Male"), ("female", "Female"), ("other", "Other")],
    )

    role = models.CharField(
        max_length=20,
        choices=[
            ("student", "Student"),
            ("parent", "Parent"),
            ("faculty", "Faculty"),
            ("staff", "Staff"),
            ("other", "Other"),
        ],
    )

    profile_image = models.ImageField(
        upload_to="profile_images/", blank=True, null=True, default=None
    )

    joined_date = models.DateTimeField(default=timezone.now)

    status = models.CharField(
        max_length=10,
        choices=[("active", "Active"), ("inactive", "Inactive")],
        default="inactive",
    )

    def __str__(self):
        return self.username


# ===========================
# ADMIN USER MODEL
# ===========================

class AdminUser(models.Model):
    username = models.CharField(max_length=150, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=128)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.username


# ===========================
# SINGLE-DOCUMENT CHAT HISTORY
# ===========================
# ✔ One document per user
# ✔ messages[] contains all Q/A
# ✔ Works with Djongo JSONField
# ===========================

class Conversation(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="conversation")
    username = models.CharField(max_length=150)

    # embedded messages array
    messages = models.JSONField(default=list)
    # Each message = { "message": str, "is_bot": bool, "timestamp": iso-str }

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Conversation of {self.username}"


# ===========================
# PENDING QUESTIONS
# ===========================
# When bot can't answer → store here
# Admin fixes & retrains model later
# ===========================

class PendingQuestion(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    username = models.CharField(max_length=150, null=True, blank=True)

    question_text = models.TextField()
    model_tag = models.CharField(max_length=100, blank=True)
    confidence = models.FloatField(default=0.0)

    created_at = models.DateTimeField(auto_now_add=True)

    is_resolved = models.BooleanField(default=False)
    admin_note = models.TextField(blank=True)

    def __str__(self):
        return self.question_text[:60]
