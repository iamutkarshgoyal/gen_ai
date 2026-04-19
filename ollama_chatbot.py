from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import dotenv
import os
import streamlit as st

dotenv.load_dotenv()

llm = OllamaLLM(model="llama3", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [("system", "You will answer all user queries."),
     ("user", "{input}")])

st.title("Ollama Translator")
# language = st.text_input("Enter the language you want to translate to:")
input_text = st.text_input("Enter the question:")

if st.button("Translate"):
    response = prompt | llm
    st.write("Translation:", response.invoke({"input": input_text}))

# response = prompt.invoke({"language": "French", "input": "Hello, how are you?"})
# response = prompt | llm
# ai_answer = response.invoke({"language": "French", "input": "Hello, how are you?"})
# print(ai_answer)