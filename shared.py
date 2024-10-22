import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=google_api_key)
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
