from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/chats/', include('core.urls')),
    path('api/v1/user/', include('authentication.urls')),
]