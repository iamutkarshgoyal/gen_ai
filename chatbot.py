from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv
import os
import streamlit as st

os.environ["LANGCHAIN_TRACING_V2"] = "false"
dotenv.load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

st.title("My Internet Mom!!")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

user_input = st.chat_input("Ask me anything!")
if user_input:
    if user_input.lower() in ["exit", "quit"]:
        st.write("Goodbye!")
    else:
        st.session_state.messages.append({"role": "User", "content": user_input})
        st.chat_message("User").markdown(user_input)
        response = llm.invoke(user_input)
        st.session_state.messages.append({"role": "Bot", "content": response.content[0]['text']})
        st.chat_message("Bot").markdown(response.content[0]['text'])