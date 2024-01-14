# webcam_project/urls.py
from django.contrib import admin
from django.urls import path
from rajpoapp.views import index, video_feed,vid_inp
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('video_feed/', video_feed, name='video_feed'),
    path('vid_inp/', vid_inp, name='vid_inp'),
    path('', index, name='index'),
] + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
