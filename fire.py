from fireworks.client import Fireworks
import os
key = os.getenv("FIREWORKS_API_KEY")
client = Fireworks(api_key=key)

response = client.chat.completions.create(
model="accounts/fireworks/models/llama-v3p3-70b-instruct",
messages=[{
   "role": "user",
   "content": "Say this is a test",
}],
)

print(response.choices[0].message.content)