from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from shared import google_api_key


def chatbot():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key)
    template = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful assistant.'),
        ('human', '{question}'),
    ])
    chatbot = template | model
    result = chatbot.invoke({
        "question": "Which model providers offer LLMs?"
    })
    print(result)


if '__main__' == __name__:
    chatbot()