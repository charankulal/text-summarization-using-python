from django.urls import path
from . import views

urlpatterns = [
    path("", views.home_page, name=""),
    path("user-login", views.user_login, name="user-login"),
    path("user-register", views.user_register, name="user-register"),
    path("dashboard", views.dashboard, name="dashboard"),
    path("user-logout", views.user_logout, name="user-logout"),
    path("ext", views.extractive_summary, name="ext"),
    path("abs", views.abstractive_summary, name="abs"),
]
