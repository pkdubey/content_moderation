import sys
import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import render

# Ensure src is in the path for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from moderate import moderate_content

class ModerateAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        text = request.data.get('text', '')
        result = moderate_content(text)
        return Response({'result': result})

def moderate_form(request):
    result = None
    if request.method == 'POST':
        text = request.POST.get('text', '')
        result = moderate_content(text)
    return render(request, 'moderation/form.html', {'result': result})
