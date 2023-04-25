from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def viewTest(request):
    return render(request, 'file.html')
