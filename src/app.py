"""Main module"""
import streamlit as st
from streamlit_chat import message
from PIL import Image

from config.chatbot_config import ChatBotConfig
from chatbot.chatbot import ChatBot


if __name__ == "__main__":
    chatbot_config = ChatBotConfig()
    chatbot_config.bot_description = (
        "تو ربات هوشمند پشتیبانی شرکت همراه اول هستی که کاربران همراه اول را " \
        "برای یافتن پاسخ سوال‌هایشان راهنمایی می‌کنی. با صداقت کامل به سوال‌ها " \
        "پاسخ بده و اگر نمی‌دانستی بگو متاسفانه پاسخ سوال شما را نمیدانم."
    )

    chatbot = ChatBot(chatbot_config=chatbot_config)

    #Creating the chatbot interface
    img = Image.open("resources/favicon.png")
    menu_items = {
        'About': "## ربات هوشمند پشتیبانی همراه اول \n " \
        "این ربات از شبکه جی‌پی‌تی ۳.۵ قدرت گرفته است و " \
        "توسط امیرحسین داغستانی توسعه یافته است."
    }
    st.set_page_config(page_title='ربات هوشمند همراه', page_icon=img, menu_items=menu_items)
    st.title("ربات هوشمند پشتیبانی همراه اول 🤖")

    FOOTER_STYLE = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                content:'.توسعه یافته توسط امیرحسین داغستانی  .Streamlit و OpenAI قدرت گرفته از'; 
                visibility: visible;
                display: block;
                position: relative;
                #background-color: red;
                padding: 5px;
                top: 2px;
            }
            </style>
            """
    st.markdown(FOOTER_STYLE, unsafe_allow_html=True)

    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    # Getting user input
    def get_text():
        """Get the input text from user"""
        input_text = st.text_input(
            label=":شما",
            placeholder="به عنوان مثال: سلام، تو کی هستی؟ یا چطور میتونم ازت کمک بگیرم؟",
            key="input")
        return input_text

    user_input = get_text()

    if user_input:
        output = chatbot.generate_response(user_input)
        # store the output 
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True,
                    key=str(i) + '_user')
