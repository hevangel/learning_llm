from ollama import chat
from ollama import ChatResponse
from ollama import Client

client = Client(
  host='http://134.199.196.38:11434'
)

response : ChatResponse = client.chat(model='phi4', messages=[
  {
    'role': 'user',
    'content': 'How to train a dog?',
  },
])

print(response.message.content)