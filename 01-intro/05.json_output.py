from langchain_core.pydantic_v1 import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

from shared import google_api_key


class AnswerWithJustification(BaseModel):
    '''An answer to the user question along with justification for the answer.'''
    answer: str
    '''The answer to the user's question'''
    justification: str
    '''Justification for the answer'''

def simple_query_to_json_output():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=google_api_key)
    structured_llm = llm.with_structured_output(AnswerWithJustification)
    completion = structured_llm.invoke("What weighs more, a pound of bricks or a pound of feathers")
    print(completion)


if '__main__' == __name__:
    simple_query_to_json_output()