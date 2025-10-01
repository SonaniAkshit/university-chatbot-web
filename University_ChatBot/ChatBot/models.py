from djongo import models

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
    profile_image = models.ImageField(upload_to='profile_images/', blank=True, null=True, default=None)

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
