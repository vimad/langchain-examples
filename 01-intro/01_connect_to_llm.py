from langchain_google_genai import GoogleGenerativeAI
from shared import google_api_key

def simple_query():
    model = GoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=google_api_key)
    prompt = 'The sky is'
    completion = model.invoke(prompt)
    print(completion)

if '__main__' == __name__:
    simple_query()