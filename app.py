from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import LLMChain
import streamlit as st

import os
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="ZOMI AI",
    page_icon="🍕",
    layout="centered"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
/* Hide Streamlit Branding */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
/* Background */
.stApp {
    background-image: linear-gradient(
        rgba(0,0,0,0),
        rgba(0,0,0,0)
    ),
    url("https://www.zomato.com/webroutes/zoomBackgrounds/downloadImage?title=Everyday%20Sundae");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
/* Main Content */
.block-container{
    padding-top: 1rem;
    padding-bottom: 120px;
}
/* Title */
.main-title{
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:white;
}
.subtitle{
    text-align:center;
    font-size:18px;
    color:white;
    margin-bottom:25px;
}
/* Chat Messages */
[data-testid="stChatMessage"]{
    border-radius:18px;
    padding:15px;
    margin-top:10px;
    border:none;
}
/* Input Box */
[data-testid="stChatInput"]{
    position: fixed;
    bottom: 20px;
    left: 22%;
    width: 56%;
    z-index: 999;
}
/* Sidebar */
section[data-testid="stSidebar"]{
    background:#1f1f1f;
}
/* Sidebar Text */
section[data-testid="stSidebar"] *{
    color:white !important;
}
/* Chat Text */
[data-testid="stChatMessage"] *{
    color:white !important;
}
/* User Bubble */
.stChatMessage[data-testid="stChatMessage"]{
    background: rgba(255,255,255,0.08);
}
/* Scrollbar */
::-webkit-scrollbar{
    width:8px;
}
::-webkit-scrollbar-thumb{
    background:#ffffff55;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- BACKEND LOGIC ---------------- #

with open("template.txt", "r") as file:
    prompt_template = file.read()

prompt = ChatPromptTemplate(
    [
        ('system', prompt_template),
        MessagesPlaceholder(variable_name='history'),
        ('human', "{input}")
    ]
)

llm = ChatMistralAI(
    model="mistral-small-2506"
)

# ---------------- SESSION STATE ---------------- #

if "langchain_memory" not in st.session_state:
    st.session_state.langchain_memory = ConversationBufferMemory(
        memory_key='history',
        return_messages=True
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=st.session_state.langchain_memory
)

# ---------------- HEADER ---------------- #

st.markdown(
    "<div class='main-title'>🍕 ZOMI AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Your Personal Food Ordering Assistant</div>",
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("🍔 ZOMI AI")

    st.markdown("---")

    st.subheader("What I Can Help With")

    st.write("""
    🍕 Food Suggestions
    🍔 Restaurant Discovery
    🥗 Healthy Meals
    🍜 Cuisine Recommendations
    🍰 Desserts
    💸 Budget-Friendly Food
    """)

    st.markdown("---")

    if st.button("🗑️ Reset Conversation"):
        st.session_state.chat_history = []
        st.session_state.langchain_memory.clear()
        st.rerun()

# ---------------- DISPLAY CHAT HISTORY ---------------- #

for chat in st.session_state.chat_history:

    avatar = "🧑" if chat["role"] == "user" else "🍕"

    with st.chat_message(chat["role"], avatar=avatar):
        st.markdown(chat["content"])

# ---------------- CHAT INPUT ---------------- #

if user_input := st.chat_input("I am hungry... Ask ZOMI AI 🍔"):

    # Display User Message
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Generate Response
    with st.spinner("🍕 Finding delicious options..."):

        response = chain.invoke(
            {
                "input": user_input
            }
        )

        ai_text = response["text"]

    # Display Assistant Message
    with st.chat_message("assistant", avatar="🍔"):
        st.markdown(ai_text)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": ai_text
        }
    )
