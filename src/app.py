"""Main module"""
from PIL import Image
import base64
import random

import pandas as pd
import streamlit as st
from streamlit_chat import message

from config.chatbot_config import get_chat_config, ChatBotModel
from config.vector_config import EmbeddingModel
from chatbot.chatbot import ChatBot

FINETUNED_MODELS = [
    'davinci:ft-personal:rbt-25-1100-ca-2023-04-30-12-42-56'
]

CHAT_ENGINES = [
    'gpt-4-prompt-engineering',
    'gpt-4',
    'gpt-3.5-turbo-prompt-engineering-classifier-response-9-non-chitchat',
    'gpt-3.5-turbo-prompt-engineering',
    'gpt-3.5-turbo',
    # 'ada:ft-personal:classifier-chitchat-2023-07-08-11-00-41',
    'text-davinci-003-prompt-engineering',
    'text-davinci-003',
    'davinci:ft-personal:faq-9epoch-2023-05-13-17-23-45-prompt-engineering',
    'davinci:ft-personal:faq-9epoch-2023-05-13-17-23-45',
    'davinci:ft-personal:rbt-25-1100-ca-2023-04-30-12-42-56',
    'davinci:ft-personal:faq-2023-05-07-05-25-57',
    'davinci:ft-personal:chat-2023-05-13-20-08-50',
]

EMBEDDING_MODEL = [
    'text-embedding-ada-002',
    'zibert_v2',
]


if __name__ == "__main__":

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
    img = Image.open("resources/img/favicon.png")
    with open("resources/img/wait.gif", "rb") as file:
        contents = file.read()
        img_wait = base64.b64encode(contents).decode("utf-8")

    menu_items = {
        'About': "## ربات هوشمند پشتیبانی همراه اول \n " \
        "این ربات از شبکه جی‌پی‌تی ۳.۵ قدرت گرفته است و " \
        "توسط امیرحسین داغستانی توسعه یافته است."
    }
    st.set_page_config(page_title='ربات هوشمند همراه', page_icon=img, menu_items=menu_items)
    st.title("🤖 ربات هوشمند همراه اول")

    st.markdown("""
    <style>
    p, div, input, label {
    direction: RTL;
    text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)
        
    if 'generated_chat_engine' not in st.session_state:
        st.session_state['generated_chat_engine'] = []
    
    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []
        
    chat_engine = st.selectbox(
        'مدل زبانی ربات را انتخاب کنید.', tuple(CHAT_ENGINES))
    add_context = False
    embedding_model = EmbeddingModel.ZIBERT
    if chat_engine.find("-prompt-engineering") != -1:
        add_context = True
        chat_engine_model = chat_engine[:chat_engine.find("-prompt-engineering")]
        embedding_model = st.selectbox(
            'وکتورایزر جست‌وجوی کانتکست را انتخاب کنید.', tuple(EMBEDDING_MODEL))
    elif chat_engine.find("-classifier") != -1:
        chat_engine_model = chat_engine[:chat_engine.find("-classifier")]
    else:
        chat_engine_model = chat_engine

    if add_context:
        chat_engine += "-" + embedding_model

    with st.spinner("لطفاً منتظر بمانید ..."):
        chatbot_config = get_chat_config(ChatBotModel(chat_engine_model), 
                                        add_context=add_context,
                                        embedding_model=EmbeddingModel(embedding_model))
        if chat_engine.find("-classifier") != -1:
            chatbot_config.temperature = 1
            chatbot_config.num_responses = int(
                chat_engine[chat_engine.find("response-") + 9:chat_engine.find("response-") + 10])
            chatbot_config.num_retrieve_context = 5
            chatbot_config.max_tokens = 512
            chatbot_config.post_process = [
                'classification'
            ]
            chatbot_config.post_process_params = {
                'classification_threshold':-0.924
            }

        if chat_engine not in st.session_state:
            st.session_state[chat_engine] = ChatBot(chatbot_config=chatbot_config)

    chatbot = st.session_state[chat_engine]

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
        input_text = st.chat_input(
            placeholder="به عنوان مثال: سلام، تو کی هستی؟",
            key="input")
        return input_text
    
    st.markdown('----')

    user_input = get_text()

    if flag_is_button_pressed:
        user_input = example_question

    if st.session_state['generated'] or \
       (len(st.session_state.generated_chat_engine) > 0 and \
        st.session_state.generated_chat_engine[-1] == 'Example'):
        for i in range(len(st.session_state['generated'])):
            with st.chat_message("user"):
                st.write(st.session_state['past'][i])

            with st.chat_message("assistant"):
                if show_chat_engine:
                    st.write(st.session_state["generated_chat_engine"][i] + ":")
                st.write(st.session_state["generated"][i])


    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            wait_placeholder = st.empty()
            wait_placeholder.markdown(
                f'<img src="data:image/gif;base64,{img_wait}" alt="wait gif" width="40px">',
                unsafe_allow_html=True,
            )
            output = chatbot.generate_response(user_input)
            output = output.replace("\n", "  \n")
            output = output.replace("*", "\*")
            wait_placeholder.empty()
            message_to_print = output
            if show_chat_engine:
                message_to_print = (chat_engine + ":") + "  \n" + message_to_print
            wait_placeholder.write(message_to_print)

        # store the output
        st.session_state.generated_chat_engine.append(chat_engine)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)