from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_postgres.vectorstores import PGVector
import uuid

import streamlit as st
from streamlit_chat import message

from shared2 import embedding_model
from shared2 import llm

st.set_page_config(
    page_title="OrfanAI Chat",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
db = PGVector(embeddings=embedding_model, connection=connection, collection_name='mine')
retriever = db.as_retriever()

if 'ready' not in st.session_state:
    st.session_state['ready'] = False

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

def get_msg_content(msg):
    return msg.content
if st.session_state['ready'] == False:
    contextualize_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    contextualize_chain = (
            contextualize_prompt
            | llm
            | get_msg_content
    )
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
            qa_prompt
            | llm
            | get_msg_content
    )


    @chain
    def history_aware_qa(input):
        # rephrase question if needed
        if input.get('chat_history'):
            question = contextualize_chain.invoke(input)
        else:
            question = input['input']
        # get context from retriever
        context = retriever.invoke(question)
        # get answer
        return qa_chain.invoke({
            **input,
            "context": context
        })


    chat_history_for_chain = InMemoryChatMessageHistory()
    qa_with_history = RunnableWithMessageHistory(
        history_aware_qa,
        lambda _: chat_history_for_chain,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    st.session_state['qa'] = qa_with_history
    st.session_state['ready'] = True

st.title("OrfanAI Chat")
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


response_container = st.container()
container = st.container()

if st.session_state['ready'] == True:
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Ask about mine data", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = st.session_state['qa'].invoke(
                {"input": user_input},
                config={"configurable": {"session_id": st.session_state['session_id']}},
            )
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['history'].append((user_input, output))

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                if i < len(st.session_state['past']):
                    st.markdown(
                        "<div style='background-color: #90caf9; color: black; padding: 10px; border-radius: 5px; width: 70%; float: right; margin: 5px;'>" +
                        st.session_state["past"][i] + "</div>",
                        unsafe_allow_html=True
                    )
                    message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")
