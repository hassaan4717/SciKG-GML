from django.urls import path
from authentication.views import signup, login

urlpatterns = [
    path('signup/', signup, name='signup'),
    path('login/', login, name='login'),
]