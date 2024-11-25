import faiss
import numpy as np
import openai
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import base64

# Make sure the OpenAI API key is set via Streamlit's secrets
openai_api_key = st.secrets["openai"]["api_key"]
openai.api_key = openai_api_key

# Set background image for Streamlit app
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

# Define the LLM (Large Language Model)
llm = ChatOpenAI(
    base_url="https://api.avalai.ir/v1",
    model="gpt-3.5-turbo",
    api_key=openai.api_key
)

# Path to your PDF file
pdf_path = r"Engine-v61n61p73-en.pdf"
reader = PdfReader(pdf_path)

# Extract text from first 50 pages
texts = []
for i, page in enumerate(reader.pages[:50]):  
    text = page.extract_text()
    if text.strip():  
        texts.append(Document(page_content=text, metadata={"page": i + 1}))

# Split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_texts = []
for doc in texts:
    chunks = text_splitter.split_text(doc.page_content)
    for chunk in chunks:
        split_texts.append(Document(page_content=chunk, metadata=doc.metadata))

# Create embeddings for each chunk
embeddings = []
for doc in split_texts:
    embedding_response = openai.Embedding.create(
        input=doc.page_content,
        model="text-embedding-ada-002"
    )
    embeddings.append(embedding_response.data[0].embedding)

# Set up FAISS index
embedding_dimension = len(embeddings[0])
index = faiss.IndexFlatL2(embedding_dimension)
index.add(np.array(embeddings))

# Initialize session state for storing messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit app header
st.markdown('<div style="background-color: rgba(49, 48, 49, 0.8); padding: 21px; border-radius: 10px; text-align: center; color: #f1c40f;"><h1>IP-CO Teaching Assistant</h1></div>', unsafe_allow_html=True)

# Sidebar with app description
with st.sidebar:
    st.title('🤖💬 OpenAI Chatbot')
    st.success('API key successfully loaded!', icon='✅')
    st.write(
        "The Chatbot Assistant is an AI-powered application designed to help users interact with large datasets and get intelligent responses. "
        "It uses advanced natural language processing (NLP) techniques and machine learning models to understand user queries and retrieve "
        "relevant information from a provided dataset, such as a PDF or CSV file."
    )

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process user input and respond
if prompt := st.chat_input("Ask me anything about Data Science!"):
    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate embedding for the user prompt
    query_response = openai.Embedding.create(
        input=prompt,
        model="text-embedding-ada-002"
    )
    query_embedding = np.array(query_response.data[0].embedding).reshape(1, -1)

    # Retrieve the most relevant documents from FAISS
    k = 5  # Number of top results to fetch
    distances, indices = index.search(query_embedding, k)

    matching_docs = [split_texts[i] for i in indices[0]]
    retrieved_texts = "\n".join([doc.page_content for doc in matching_docs])

    # Combine the user prompt and retrieved context for the LLM
    prompt_with_context = f"""
    user_prompt:
    {prompt}

    retrieved_context:
    {retrieved_texts}
    """

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Use the LLM to generate a response with context
        response = llm.invoke([{"role": "user", "content": prompt_with_context}])
        
        full_response = response.content
        
        # Display the response in the chat
        message_placeholder.markdown(full_response)
    
    # Append assistant message to session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
