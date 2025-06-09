import ollama
response = ollama.chat(model='phi4', messages=[
  {
    'role': 'user',
    'content': 'How to train a dog?',
  },
])
print(response['message']['content'])