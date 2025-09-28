# ChatBot/models.py
from djongo import models

class User(models.Model):
    full_name = models.CharField(max_length=100)
    username = models.CharField(max_length=50, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=255)  # store hashed password
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

    def __str__(self):
        return self.username
