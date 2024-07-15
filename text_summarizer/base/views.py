from django.shortcuts import render, redirect
from django.urls import reverse
from .forms import CreateUserForm, LoginForm
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import auth
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from summarizer_models.extractive_summary import extractive_summary_func
from summarizer_models.abstractive_summary import abstractive_summarization


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
def extractive_summary_function(request):
    summarized_text = None
    input_text = None
    custom_percentage = None
    if request.method == "POST":
        input_text = request.POST.get("input_text")
        custom_percentage = int(request.POST.get("custom_percentage"))
        summarized_text = extractive_summary_func(input_text, custom_percentage / 100)
        length=len(input_text.split(' '))
        length_summary=len(summarized_text.split(' '))

    return render(
        request,
        "base/ext.html",
        {
            "summarized_text": summarized_text,
            "input_text": input_text,
            "percentage": custom_percentage,
            "length":length,
            "length_summary":length_summary
        },
    )


@login_required(login_url="user-login")
def abstractive_summary(request):
    summarized_text = None
    input_text = None
    length=0
    length_summary=0
    if request.method == "POST":
        input_text = request.POST.get("input_text")
        length=len(input_text.split(' '))
        summarized_text = abstractive_summarization(input_text,min_length=int(length*0.4))
        
        length_summary=len(summarized_text.split(' '))

    return render(
        request,
        "base/abs.html",
        {
            "summarized_text": summarized_text,
            "input_text": input_text,
            "length":length,
            "length_summary":length_summary
        },
    )
