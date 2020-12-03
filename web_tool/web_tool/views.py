from django.shortcuts import render


# Create your views here.

def index(request):
    return render(request, 'index.html')

# def hyp_test(request):
#     return render(request, 'hyp_test.html')


