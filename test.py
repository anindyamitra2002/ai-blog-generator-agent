from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:20128/v1",
    api_key="sk-983eb88713250912-20bf5f-ddfa210e",  # from your dashboard
)

response = client.chat.completions.create(
    model="auto",  # swap for any model available on your router
    messages=[
        {"role": "user", "content": "What is Operation Sindoor"}
    ],
)

print(response.choices[0].message.content)