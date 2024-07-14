from django.shortcuts import render, redirect
from django.urls import reverse
from .forms import CreateUserForm, LoginForm
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import auth
from django.contrib import messages
from django.contrib.auth.decorators import login_required


def home_page(request):
    if request.user.is_authenticated:
        return redirect(reverse("dashboard"))
    return render(request, "base/index.html")


def user_register(request):
    if request.user.is_authenticated:
        return redirect(reverse("dashboard"))
    form = CreateUserForm()
    if request.method == "POST":
        form = CreateUserForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("user-login")
    context = {"registerform": form}
    return render(request, "base/user-register.html", context=context)


def user_login(request):
    if request.user.is_authenticated:
        return redirect(reverse("dashboard"))
    form = LoginForm()
    if request.method == "POST":
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = request.POST.get("username")
            password = request.POST.get("password")
            user = authenticate(request, username=username, password=password)

            if user is not None:
                auth.login(request, user)
                return redirect("dashboard")
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    context = {"loginform": form}
    return render(request, "base/user-login.html", context=context)


@login_required(login_url="user-login")
def dashboard(request):
    return render(request, "base/dashboard.html")


@login_required(login_url="user-login")
def user_logout(request):
    auth.logout(request)
    return redirect("")


@login_required(login_url="user-login")
def extractive_summary(request):
    pass


@login_required(login_url="user-login")
def abstractive_summary(request):
    pass
