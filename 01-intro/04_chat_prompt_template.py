from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from shared import google_api_key


def simple_query_with_chat_prompt_template():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=google_api_key)
    template = ChatPromptTemplate.from_messages([
        ('system',
         'Answer the question based on the context below. If the question cannot be answered using the information provided answer with "I don\'t know".'),
        ('human', 'Context: {context}'),
        ('human', 'Question: {question}'),
    ])
    prompt = template.invoke({
        "context": "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries respectively.",
        "question": "Which model providers offer LLMs?"
    })
    completion = model.invoke(prompt)
    print(completion)


if '__main__' == __name__:
    simple_query_with_chat_prompt_template()