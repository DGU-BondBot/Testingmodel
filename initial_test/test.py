# This function is not required, but it's nice to have
def get_completion(prompt, model="llama2", temperature=0.0):
     messages = [{"role": "user", "content": prompt}]
     response = client.chat.completions.create(
         model=model,
         messages=messages,
         temperature=temperature,
     )
     return response.choices[0].message.content

response = get_completion("What is the chief end of man?")
print(response)