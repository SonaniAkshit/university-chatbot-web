from django.db import models
from django.utils import timezone

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

    # New fields
    joined_date = models.DateTimeField(default=timezone.now)
    status = models.CharField(
        max_length=10,
        choices=[("active", "Active"), ("inactive", "Inactive")],
        default="inactive",   # Default inactive
    )

    def __str__(self):
        return self.username

class AdminUser(models.Model):
    username = models.CharField(max_length=150, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=128)  # Store plain text or hashed manually if needed

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.username

class ChatMessage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    username = models.CharField(max_length=150, null=True, blank=True)  # new field
    message = models.TextField()
    is_bot = models.BooleanField(default=False)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        sender = "Bot" if self.is_bot else "User"
        return f"{sender}: {self.message[:50]}"


# {
#   "_id": {
#     "$oid": "68dd11141e965a7e8cd97061"
#   },
#   "user_id": 13,
#   "username": "user_example", 
#   "messages": [
#     {
#       "message": "hi",
#       "is_bot": false,
#       "timestamp": {
#         "$date": "2025-10-01T11:31:32.400Z"
#       }
#     },
#     {
#       "message": "Greetings! I'm here to help you with any questions about Gujarat Vidyapith. 🙏",
#       "is_bot": true,
#       "timestamp": {
#         "$date": "2025-10-01T11:31:32.440Z"
#       }
#     },
#     {
#       "message": "hi",
#       "is_bot": false,
#       "timestamp": {
#         "$date": "2025-10-01T11:33:35.878Z"
#       }
#     },
#     {
#       "message": "Hi there! What information are you looking for? 🤔",
#       "is_bot": true,
#       "timestamp": {
#         "$date": "2025-10-01T11:33:35.946Z"
#       }
#     }
#   ]
# }
