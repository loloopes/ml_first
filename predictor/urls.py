from django.contrib import admin
from django.urls import path
from predictor.views import viewTest

urlpatterns = [
    path('predict/', viewTest),
]