import streamlit as st
from langchain_helper import get_few_shot_db_chain

st.title("Martians' T Shirts: Database Retriever Using GenAI 👕")

question = st.text_input("Question: ")

if question:
    chain = get_few_shot_db_chain()
    response = chain.invoke(question)

    st.header("Answer")
    print(response)
    st.write(response["result"])






