from transformers import pipeline
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')
print(generator("한국의 수도는 어디야?", max_length=50))


