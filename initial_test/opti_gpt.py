from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI()

# Load data from file
data_loader = UnstructuredFileLoader("../files/wiki.txt")

# Cache directory for storing embeddings
cache_dir = LocalFileStore("./.cache/")

# Splitter for breaking text into chunks
splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=500,
    chunk_overlap=50
)

# Load and split the text data
docs = data_loader.load_and_split(text_splitter=splitter)

# Initialize OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Initialize CacheBackedEmbeddings with cache directory
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

# Initialize Chroma vector store from documents with cached embeddings
vectorstore = Chroma.from_documents(docs, cached_embeddings)

# Convert vector store to retriever
retriever = vectorstore.as_retriever()

# Initialize RetrievalQA chain
chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="map_reduce",
    retriever=retriever,
)

# Run the retrieval QA chain
chain.run("한국의 집단주의에 대해 설명해줘")
