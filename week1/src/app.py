import os
from dotenv import load_dotenv
import streamlit as st

from graph.state import GraphState
from graph.builder import pipeline

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "smalltalk2rec"

# Streamlit App UI

st.set_page_config(page_title="SmallTalk2Rec")

# Replicate Credentials
with st.sidebar:
    st.title("대화형 영화 추천")

    # selectbox 레이블 공백 제거
    st.markdown(
        """
        <style>
        .stSelectbox label {  /* This targets the label element for selectbox */
            display: none;  /* Hides the label element */
        }
        .stSelectbox div[role='combobox'] {
            margin-top: -20px; /* Adjusts the margin if needed */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.title("SmallTalk2Rec")
st.subheader("영화 추천해드릴게요! 🎬")

st.write("")

# image_path = "https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.aitimes.com%2Fnews%2FarticleView.html%3Fidxno%3D159842&psig=AOvVaw2kQsd_hMp8onpGjbKf1YpW&ust=1729006825575000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCLiPysCajokDFQAAAAAdAAAAABAE"
# image_html = f"""
# <div style="display: flex; justify-content: center;">
#     <img src="{image_path}" alt="centered image" width="50%">
# </div>
# """
# st.markdown(image_html, unsafe_allow_html=True)

st.write("")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "안녕하세요 영화 추천 챗봇입니다. 무엇을 도와드릴까요?",
        }
    ]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "안녕하세요 제주도 맛집 추천 챗봇입니다. 무엇을 도와드릴까요?",
        }
    ]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


# User-provided prompt
if prompt := st.chat_input():  # (disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # response = generate_llama2_response(prompt)
            inputs = GraphState(question=prompt)
            response = pipeline.invoke(inputs)["answer"]

            response = response.replace("~", "\\~")

            placeholder = st.empty()
            placeholder.markdown(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
