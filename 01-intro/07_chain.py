from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_google_genai import ChatGoogleGenerativeAI

from shared import google_api_key


@chain
def chatbot(values):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=google_api_key)
    template = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful assistant.'),
        ('human', '{question}'),
    ])
    prompt = template.invoke(values)
    for token in model.invoke(prompt):
        yield token


if '__main__' == __name__:
    result = chatbot.invoke({
        "question": "Which model providers offer LLMs?"
    })
    print(result)