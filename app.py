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


def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()  # Base64 encode the image
    css = f"""
    <style>
    .stApp {{
        background-image: url('data:image/jpeg;base64,{encoded_image}');
        background-size: cover;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set the background
set_background("background.png")

with st.sidebar:
    st.title('🤖💬 Knowledge Assistant')
    st.markdown(
        """
        The Chatbot Assistant is an AI-powered application designed to help users interact with large datasets and get intelligent responses. 
        It uses advanced natural language processing (NLP) techniques and machine learning models to understand user queries and retrieve relevant information          from a provided dataset, such as a PDF or CSV file.
        """
    )



openai.api_key = "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"

client = openai.Client(api_key="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")

llm = ChatOpenAI(
    base_url="https://api.avalai.ir/v1",
    model="gpt-3.5-turbo",
    api_key="aa-gAp2CUy0mkUSBkHkJ8HpVNnVW099QOZgkKG99LN8gpxc5fwT"
)

pdf_path = r"Engine-v61n61p73-en.pdf"
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
    embedding_response = client.embeddings.create(
        input=doc.page_content,
        model="text-embedding-ada-002"
    )
    embeddings.append(embedding_response.data[0].embedding)

embedding_dimension = len(embeddings[0])
index = faiss.IndexFlatL2(embedding_dimension)
index.add(np.array(embeddings))

if "messages" not in st.session_state:
    st.session_state.messages = []

# Darker Chat Window CSS
st.markdown("""
    <style>
        .user-message {
            background-color: #333333;  /* Dark background for user messages */
            color: white;                /* White text for readability */
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        
        .assistant-message {
            background-color: #444444;  /* Darker background for assistant messages */
            color: white;                /* White text for readability */
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        
        .stChatMessage {
            padding-left: 0;
            padding-right: 0;
        }
        
    </style>
""", unsafe_allow_html=True)

# Display Title
st.markdown('<div style="background-color: rgba(49, 48, 49, 0.8); padding: 21px; border-radius: 10px; text-align: center; color: #f1c40f;"><h1>Knowledge Assistant</h1></div>', unsafe_allow_html=True)

for message in st.session_state.messages:
    message_class = "user-message" if message["role"] == "user" else "assistant-message"
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="{message_class}">{message["content"]}</div>', unsafe_allow_html=True)

if prompt := st.chat_input("Ask Here!"):
    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
    
    query_response = client.embeddings.create(
        input=prompt,
        model="text-embedding-ada-002"
    )
    query_embedding = np.array(query_response.data[0].embedding).reshape(1, -1)

    k = 5  # Top 5 most relevant results
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
        full_response = ""
        
        llm = ChatOpenAI(
            base_url = "https://api.avalai.ir/v1",
            model = "gpt-3.5-turbo",
            api_key = "aa-gAp2CUy0mkUSBkHkJ8HpVNnVW099QOZgkKG99LN8gpxc5fwT"
        )
        response = llm.invoke([{"role" : "user" , "content" : prompt_with_context}])
        
        full_response = response.content
        
        message_placeholder.markdown(f'<div class="assistant-message">{full_response}</div>', unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
