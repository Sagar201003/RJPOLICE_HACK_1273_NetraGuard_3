# yourapp/urls.py
from . import views
from django.contrib import admin
from django.urls import URLPattern, include, path
# urls.py

urlpatterns = [
    path('', views.signin, name="signin"),
    path('signup', views.signup, name="signup"),
    path('home', views.home, name="home"),
    path('history', views.history, name="history"),
]

# if settings.DEBUG:
#     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
