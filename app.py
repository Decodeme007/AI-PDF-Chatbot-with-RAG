# ==============================================================================
# AI PDF CHATBOT WITH RETRIEVAL-AUGMENTED GENERATION (RAG)
# ------------------------------------------------------------------------------
# This application allows users to upload multiple PDF documents and ask
# questions about their content.
#
# HOW IT WORKS (RAG Pipeline):
# 1. Extraction: Reads and extracts text from uploaded PDF files (PyPDF2).
# 2. Chunking: Splits long text into smaller overlapping chunks (LangChain TextSplitter).
# 3. Embedding: Converts text chunks into mathematical vectors (Hugging Face MiniLM).
# 4. Storage & Search: Stores vectors in a local FAISS index for fast similarity search.
# 5. Generation: When a user asks a question, FAISS finds the most relevant text chunks,
#    and Google Gemini (gemini-2.0-flash) generates an answer using that context.
# 6. UI: Built with Streamlit for a web-based chat interface.
# ==============================================================================

# --- UI and Core Utilities ---
import streamlit as st           # Web interface framework
from PyPDF2 import PdfReader     # Tool to read and extract text from PDF files
import pandas as pd              # Used to organize chat history into tables
import base64                    # Used for encoding chat history into downloadable CSV
import os                        # Operating system utilities
from datetime import datetime    # Timestamp tracking for chat messages and CSV export

# --- LangChain & RAG Components ---
# Splits large documents into smaller pieces with overlap so context isn't lost at borders
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store to store embeddings and perform fast similarity search
from langchain_community.vectorstores import FAISS

# Embedding model to convert text chunks into numerical vectors locally (free, no API key needed)
from langchain_huggingface import HuggingFaceEmbeddings

# LLM integration for Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

# Pre-built chain that feeds relevant context documents into an LLM along with the user question
from langchain.chains.question_answering import load_qa_chain

# Template to guide the LLM's behavior and response style
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

import asyncio

# ------------------------------------------------------------------------------
# Asyncio Event Loop Setup
# ------------------------------------------------------------------------------
# Streamlit runs scripts in worker threads. Some underlying asynchronous libraries
# (like Google's client SDK) expect an active asyncio event loop.
# This ensures an event loop is always available in the current thread to avoid errors.
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# ------------------------------------------------------------------------------
# STEP 1: Extract Text from Uploaded PDFs
# ------------------------------------------------------------------------------
def get_pdf_text(pdf_docs):
    """
    Reads all uploaded PDF files page by page and concatenates their text into one single string.
    
    Args:
        pdf_docs (list): List of uploaded PDF file objects from Streamlit file_uploader.
    
    Returns:
        str: Combined plain text of all pages across all uploaded PDFs.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            # Extract text from each page and append to our master text string
            text += page.extract_text()
    return text


# ------------------------------------------------------------------------------
# STEP 2: Split Text into Chunks
# ------------------------------------------------------------------------------
def get_text_chunks(text, model_name):
    """
    Breaks a large block of text into smaller, manageable chunks.
    
    Why chunking is needed:
    - LLMs have context limits and perform better with focused information.
    - Smaller chunks make search retrieval more precise.
    - 'chunk_overlap' ensures sentences split at boundaries still retain context.
    
    Args:
        text (str): The full extracted text from the PDFs.
        model_name (str): The selected model provider (e.g., 'Google AI').
    
    Returns:
        list[str]: A list of text chunks.
    """
    if model_name == "Google AI":
        # Split text into chunks of 100 characters with an overlap of 100 characters
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


# ------------------------------------------------------------------------------
# STEP 3: Create Vector Store (FAISS) using Embeddings
# ------------------------------------------------------------------------------
def get_vector_store(text_chunks, model_name, api_key=None):
    """
    Converts text chunks into numerical vectors (embeddings) and saves them
    into a local FAISS vector index on disk.
    
    How this works:
    - We use Hugging Face's 'sentence-transformers/all-MiniLM-L6-v2' model.
    - It maps text into a 384-dimensional vector space where semantically similar
      sentences sit closer together.
    - FAISS indexes these vectors for lightning-fast similarity lookups.
    
    Args:
        text_chunks (list[str]): List of text segments to embed.
        model_name (str): The model provider configuration.
        api_key (str, optional): API key if needed for cloud embeddings.
    
    Returns:
        FAISS: The initialized FAISS vector database object.
    """
    if model_name == "Google AI":
        # Using HuggingFace embeddings running locally on your CPU/GPU
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create the vector store from our text chunks
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    # Save the vector index locally in a folder named 'faiss_index'
    vector_store.save_local("faiss_index")
    return vector_store


# ------------------------------------------------------------------------------
# STEP 4: Build Question-Answering Prompt & LLM Chain
# ------------------------------------------------------------------------------
def get_conversational_chain(model_name, vectorstore=None, api_key=None):
    """
    Prepares the Google Gemini LLM with a strict prompt template that forces it
    to answer using ONLY the provided PDF context, reducing hallucinations.
    
    Args:
        model_name (str): Selected model provider.
        vectorstore (FAISS, optional): The vector store instance.
        api_key (str, optional): User's Google Gemini API key.
    
    Returns:
        Chain: A LangChain QA chain ready to answer questions.
    """
    if model_name == "Google AI":
        # Prompt instructs the model to only use the context and not invent facts
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        # Initialize Google Gemini Flash model with low temperature (0.3) for factual answers
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=api_key)
        
        # Define prompt variables
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # 'stuff' chain type takes all retrieved context documents and inserts ('stuffs') them into the prompt
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain


# ------------------------------------------------------------------------------
# STEP 5: Process User Question (RAG Retrieval & Answer Generation)
# ------------------------------------------------------------------------------
def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    """
    Orchestrates the entire query pipeline:
    1. Validates API key and uploaded PDFs.
    2. Chunks and indexes the PDFs into FAISS.
    3. Searches FAISS for the most relevant text chunks matching the user's question.
    4. Passes the chunks + question to Gemini to generate the answer.
    5. Displays the result and updates conversation history.
    
    Args:
        user_question (str): The question typed by the user.
        model_name (str): Selected model provider.
        api_key (str): User's Google API key.
        pdf_docs (list): Uploaded PDF files.
        conversation_history (list): Session chat history list.
    """
    # Guard check: Ensure required inputs are present
    if api_key is None or pdf_docs is None:
        st.warning("Please upload PDF files and provide API key before processing.")
        return

    # Extract text and re-index vector store with current documents
    text_chunks = get_text_chunks(get_pdf_text(pdf_docs), model_name)
    vector_store = get_vector_store(text_chunks, model_name, api_key)
    
    user_question_output = ""
    response_output = ""
    
    if model_name == "Google AI":
        # Load the saved FAISS vector store with HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Find the text chunks most semantically similar to the user question
        docs = new_db.similarity_search(user_question)
        
        # Get the Gemini QA chain and run inference
        chain = get_conversational_chain("Google AI", vectorstore=new_db, api_key=api_key)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        user_question_output = user_question
        response_output = response['output_text']
        
        # Record this turn in the conversation history
        pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
        conversation_history.append((
            user_question_output,
            response_output,
            model_name,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            ", ".join(pdf_names)
        ))

    # --- Render the Current Chat Exchange (User + Bot) with Custom HTML/CSS ---
    st.markdown(
        f"""
        <style>
            .chat-message {{
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
            }}
            .chat-message.user {{
                background-color: #2b313e;
            }}
            .chat-message.bot {{
                background-color: #475063;
            }}
            .chat-message .avatar {{
                width: 20%;
            }}
            .chat-message .avatar img {{
                max-width: 78px;
                max-height: 78px;
                border-radius: 50%;
                object-fit: cover;
            }}
            .chat-message .message {{
                width: 80%;
                padding: 0 1.5rem;
                color: #fff;
            }}
            .chat-message .info {{
                font-size: 0.8rem;
                margin-top: 0.5rem;
                color: #ccc;
            }}
        </style>
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
            </div>    
            <div class="message">{user_question_output}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
            </div>
            <div class="message">{response_output}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Manage history display: prevent duplicate rendering of the most recent exchange
    if len(conversation_history) == 1:
        conversation_history = []
    elif len(conversation_history) > 1:
        last_item = conversation_history[-1]
        conversation_history.remove(last_item)

    # Render previous conversation turns in reverse chronological order
    for question, answer, model_name, timestamp, pdf_name in reversed(conversation_history):
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>    
                <div class="message">{question}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
                </div>
                <div class="message">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Provide a CSV download button in the sidebar if conversation history exists
    if len(st.session_state.conversation_history) > 0:
        df = pd.DataFrame(
            st.session_state.conversation_history,
            columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"]
        )
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Convert CSV to base64 download string
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV file</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.markdown("To download the conversation, click the Download button on the left side at the bottom of the conversation.")

    # Visual celebration animation in Streamlit
    st.snow()


# ------------------------------------------------------------------------------
# STEP 6: Streamlit Main Interface & Sidebar Setup
# ------------------------------------------------------------------------------
def main():
    """
    Main function setting up page configuration, the sidebar controls,
    PDF upload widget, and user input box.
    """
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs (v1) :books:")

    # Initialize chat history in Streamlit session state so it persists across re-renders
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Social links
    github_profile_link = "https://github.com/Decodeme007"
    st.sidebar.markdown(
        f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_profile_link})"
    )

    # Sidebar: Model Selection
    model_name = st.sidebar.radio("Select the Model:", ("Google AI",))

    api_key = None

    # Sidebar: Google Gemini API Key Input
    if model_name == "Google AI":
        api_key = st.sidebar.text_input("Enter your Google API Key:", type="default")
        st.sidebar.markdown("Click [here](https://ai.google.dev/) to get an API key.")
        
        # Stop execution if user hasn't entered their API key yet
        if not api_key:
            st.sidebar.warning("Please enter your Google API Key to proceed.")
            return

    # Sidebar: Action buttons & PDF File Uploader
    with st.sidebar:
        st.title("Menu:")
        
        col1, col2 = st.columns(2)
        reset_button = col2.button("Reset")
        clear_button = col1.button("Rerun")

        # 'Reset' clears all state, API keys, and conversation history
        if reset_button:
            st.session_state.conversation_history = []  # Clear conversation history
            st.session_state.user_question = None       # Clear user question input 
            api_key = None                             # Reset Google API key
            pdf_docs = None                            # Reset PDF documents
        else:
            # 'Rerun' discards the last question or re-queries
            if clear_button:
                if 'user_question' in st.session_state:
                    st.warning("The previous query will be discarded.")
                    st.session_state.user_question = ""
                    if len(st.session_state.conversation_history) > 0:
                        st.session_state.conversation_history.pop()  # Remove last question from history
                else:
                    st.warning("The question in the input will be queried again.")

        # PDF File Uploader (supports uploading multiple files simultaneously)
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    st.success("Done")
            else:
                st.warning("Please upload PDF files before processing.")

    # Main area: Question input box for the user
    user_question = st.text_input("Ask a Question from the PDF Files")

    # When the user submits a question, trigger the RAG pipeline
    if user_question:
        user_input(user_question, model_name, api_key, pdf_docs, st.session_state.conversation_history)
        st.session_state.user_question = ""  # Clear user question input for next run


if __name__ == "__main__":
    main()
