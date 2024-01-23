from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st
from PIL import Image
import io
import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])


def load_config():
    load_dotenv(find_dotenv())


def img_to_text(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image_to_text = pipeline(
        'image-to-text', model='nlpconnect/vit-gpt2-image-captioning')
    text = image_to_text(image)[0]['generated_text']
    return text


def text_to_story(scenario):
    template = """
    Imagine you are an expert storyteller with the voice of Phua Chua Kang from Singapore. In the local Singlish dialect, generate a story
    of not more than 100 words.

    CONTEXT: {scenario}
    STORY:
    """

    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI()
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    story = chain.invoke({'scenario': scenario})
    return story


def log_call(caption, story):
    log_entry = f"Caption: {caption}|| Generated: {story}"
    logger.info(log_entry)


def draw_ui():
    st.set_page_config(
        page_title="PCK!", page_icon="🇸🇬")
    st.header("Eh PCK can help you tell story one!")
    "You send me picture, I help you tell story lor - very simple one!"
    st.image("phuachukang.jpg")
    uploaded_file = st.file_uploader(
        "Show me picture lah...", type=["jpg", "png"])

    if uploaded_file is not None:
        img_bytes = uploaded_file.read()
        st.image(uploaded_file, caption='This one you send one right?',
                 use_column_width=True)
        st.write("")
        with st.spinner("Wait ah, I thinking..."):
            with st.expander("PCK says:", expanded=True):
                caption = img_to_text(img_bytes)
                story = text_to_story(caption)
                log_call(caption, story, )
                st.write(story)


if __name__ == "__main__":
    load_config()
    draw_ui()
