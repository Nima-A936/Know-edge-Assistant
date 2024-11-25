import openai
import streamlit as st

# Make sure the OpenAI API key is set via Streamlit's secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

if openai.api_key is None:
    raise ValueError("API key is not set! Please set the OPENAI_API_KEY environment variable.")
