from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from .nlp.chatbot_logic import load_responses, get_response

# Load responses at server start
responses = load_responses()

def chat_view(request):
    return render(request, 'chat.html')

@csrf_exempt
def webhook(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_query = data.get('query', '').lower()

        # Get response using NLP logic
        response = get_response(user_query, responses)

        return JsonResponse({'response': response})
    return JsonResponse({'error': 'Invalid request'}, status=400)