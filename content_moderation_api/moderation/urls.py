from django.urls import path
from .views import ModerateAPIView, moderate_form

urlpatterns = [
    path('api/moderate/', ModerateAPIView.as_view(), name='api-moderate'),
    path('moderate/', moderate_form, name='moderate-form'),
]
