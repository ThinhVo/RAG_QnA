import streamlit as st
import requests

def get_LLM_response(user_prompt):
    result = requests.post(url='http://127.0.0.1:5000/answer', data={'prompt': user_prompt})
    # result = requests.post(url='http://llm-agent:5000/answer', data={'prompt': user_prompt})
    return result

st.title('Generative Agent - Q&A Chatbot')


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Type your question here"):
    # Display user message in chat message container
    st.chat_message("user").markdown(user_prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    response = get_LLM_response(user_prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        completion = response.json()['completion']
        print(response)
        st.markdown(completion)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": completion})