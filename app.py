import os
import faiss
import numpy as np
import openai
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import base64

# Retrieve API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is set
if openai_api_key is None:
    raise ValueError("API key is not set! Please set the OPENAI_API_KEY in GitHub Secrets.")

# Set the OpenAI client key
openai.api_key = openai_api_key

# Set up Streamlit app background
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url('data:image/jpeg;base64,{encoded_image}');
        background-size: cover;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("background.png")

with st.sidebar:
    st.title('🤖💬 Knowledge Assistant')
    st.markdown(
        """
        The Chatbot Assistant is an AI-powered application designed to help users interact with large datasets 
        and get intelligent responses. It uses advanced natural language processing (NLP) techniques and 
        machine learning models to retrieve relevant information from a dataset.
        """
    )

# Initialize the ChatOpenAI client
llm = ChatOpenAI(
    base_url="https://api.avalai.ir/v1",
    model="gpt-3.5-turbo",
    api_key=openai_api_key
)

# PDF text processing and embedding setup
pdf_path = "Engine-v61n61p73-en.pdf"
reader = PdfReader(pdf_path)

texts = []
for i, page in enumerate(reader.pages[:50]):  
    text = page.extract_text()
    if text.strip():  
        texts.append(Document(page_content=text, metadata={"page": i + 1}))

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_texts = []
for doc in texts:
    chunks = text_splitter.split_text(doc.page_content)
    for chunk in chunks:
        split_texts.append(Document(page_content=chunk, metadata=doc.metadata))

embeddings = []
for doc in split_texts:
    embedding_response = openai.Embedding.create(
        input=doc.page_content,
        model="text-embedding-ada-002"
    )
    embeddings.append(embedding_response.data[0].embedding)

embedding_dimension = len(embeddings[0])
index = faiss.IndexFlatL2(embedding_dimension)
index.add(np.array(embeddings))

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Title and Chat Messages
st.markdown('<h1 style="color:#f1c40f;text-align:center;">Knowledge Assistant</h1>', unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("Ask Here!"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
    
    query_response = openai.Embedding.create(
        input=prompt,
        model="text-embedding-ada-002"
    )
    query_embedding = np.array(query_response.data[0].embedding).reshape(1, -1)

    k = 5  # Top 5 results
    distances, indices = index.search(query_embedding, k)

    matching_docs = [split_texts[i] for i in indices[0]]
    retrieved_texts = "\n".join([doc.page_content for doc in matching_docs])

    prompt_with_context = f"""
    user_prompt:
    {prompt}

    retrieved_context:
    {retrieved_texts}
    """

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = llm.invoke([{"role": "user", "content": prompt_with_context}]).content
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
