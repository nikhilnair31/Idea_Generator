import os
import openai

OPENAI_API_KEY = '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'
openai.api_key = OPENAI_API_KEY

response = openai.Completion.create(engine="davinci", prompt="Make a startup that", max_tokens=32)

print(f'\nresponse :\n{response}')