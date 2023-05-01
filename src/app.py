"""Main module"""
import streamlit as st
from streamlit_chat import message
from PIL import Image
import random
import pandas as pd

from config.chatbot_config import ChatBotConfig
from chatbot.chatbot import ChatBot

FINETUNED_MODELS = [
    'davinci:ft-personal:rbt-25-1100-ca-2023-04-30-12-42-56'
]
if __name__ == "__main__":
    chatbot_config = ChatBotConfig()
    chatbot_config.bot_description = (
        "تو ربات هوشمند پشتیبانی شرکت همراه اول هستی که کاربران همراه اول را " \
        "برای یافتن پاسخ سوال‌هایشان راهنمایی می‌کنی. با صداقت کامل به سوال‌ها " \
        "پاسخ بده و اگر نمی‌دانستی بگو متاسفانه پاسخ سوال شما را نمیدانم."
    )

    if 'question_answers' not in st.session_state:
        test_questions = pd.read_csv('resources/RBT_questions_not_in_train.csv')
        question_answers = {}
        for i in range(len(test_questions)):
            sample_questions = list(eval(test_questions.iloc[i]['questions']))
            samples = random.sample(sample_questions, 1)
            question_answers[str(samples[0])] = test_questions.iloc[i]['answer']
        question_list = list(question_answers.keys())
        question_list.insert(0, "")
        st.session_state['question_answers'] = question_answers
        st.session_state['question_list'] = question_list

    # Creating the chatbot interface
    img = Image.open("resources/favicon.png")
    menu_items = {
        'About': "## ربات هوشمند پشتیبانی همراه اول \n " \
        "این ربات از شبکه جی‌پی‌تی ۳.۵ قدرت گرفته است و " \
        "توسط امیرحسین داغستانی توسعه یافته است."
    }
    st.set_page_config(page_title='ربات هوشمند همراه', page_icon=img, menu_items=menu_items)
    st.title(f"ربات هوشمند پشتیبانی همراه اول 🤖")
        
    if 'generated_chat_engine' not in st.session_state:
        st.session_state['generated_chat_engine'] = []
    
    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []
        
    chat_engine = st.selectbox(
    'مدل زبانی ربات را انتخاب کنید.',
    ('gpt-3.5-turbo', 'text-davinci-003', 'gpt-4', 
     'davinci:ft-personal:rbt-25-1100-ca-2023-04-30-12-42-56'))
    chatbot_config.chat_engine = chat_engine
    chatbot = ChatBot(chatbot_config=chatbot_config)

    show_chat_engine = st.checkbox("نام مدل زبانی در پاسخ نمایش داده شود.", 
                                   value=True)

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

    flag_is_button_pressed = False
    if chat_engine in FINETUNED_MODELS:
        cols = st.columns([1,0.1])
        example_question = cols[0].selectbox(
            'این مدل دارای سوالات پیش‌فرض به همراه پاسخ است. برای امتحان سوال نمونه را انتخاب کنید.',
            tuple(st.session_state['question_list'])
        )
        cols[1].markdown('##')
        if cols[1].button("تایید"):
            if example_question and example_question != "":
                st.session_state.generated_chat_engine.append('database')
                # store the output
                st.session_state.past.append(example_question)
                st.session_state.generated.append(
                    st.session_state['question_answers'][example_question])
                flag_is_button_pressed = True

    # Getting user input
    def get_text():
        """Get the input text from user"""
        input_text = st.text_input(
            label=":شما",
            placeholder="به عنوان مثال: سلام، تو کی هستی؟ یا چطور میتونم ازت کمک بگیرم؟",
            key="input")
        return input_text
    
    st.markdown('----')

    user_input = get_text()

    if flag_is_button_pressed:
        user_input = example_question

    if user_input:
        with st.spinner("لطفاً منتظر بمانید ..."):
            output = chatbot.generate_response(user_input)
            st.session_state.generated_chat_engine.append(chat_engine)
            # store the output
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

    if st.session_state['generated'] or \
       (len(st.session_state.generated_chat_engine) > 0 and \
        st.session_state.generated_chat_engine[-1] == 'Example'):
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            if show_chat_engine:
                message(st.session_state["generated_chat_engine"][i] + ":", 
                        key=str(i) + "_engine")
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True,
                    key=str(i) + '_user')
