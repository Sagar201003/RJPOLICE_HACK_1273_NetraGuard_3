from django.shortcuts import render

from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login
from .models import CustomUser

from django.shortcuts import render

def home(request):
    return render(request,"home.html") 

def history(request):
    return render(request,"history.html")

def signup(request):
    if request.method == 'POST':
        full_name = request.POST.get('full_name')
        phone = request.POST.get('phone')
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = CustomUser.objects.create_user(
            full_name=full_name,
            phone=phone,
            username=username,
            password=password
        )

        login(request, user)

        return redirect('home') 
    else:
        return render(request, 'signup.html')

def signin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('home') 
        else:
            return render(request, 'signin.html', {'error_message': 'Invalid username or password'})

    return render(request, 'signin.html')
