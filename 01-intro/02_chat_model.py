from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from shared import google_api_key


def simple_query_with_chat_model():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=google_api_key)
    system_msg = SystemMessage('You are a helpful assistant that responds to questions with three exclamation marks.')
    human_msg = HumanMessage('What is the capital of France?')
    completion = model.invoke([system_msg, human_msg])
    print(completion)

if '__main__' == __name__:
    simple_query_with_chat_model()