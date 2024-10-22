from langchain_google_genai import ChatGoogleGenerativeAI

from shared import google_api_key


def simple_query_using_batch_and_stream():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=google_api_key)
    completions = model.batch(['Hi there!', 'Bye!'])
    print(completions)
    for token in model.stream('Bye!'):
        print(token)


if '__main__' == __name__:
    simple_query_using_batch_and_stream()