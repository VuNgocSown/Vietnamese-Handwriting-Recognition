from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('recognize/', views.recognize_handwriting, name='recognize'),
]