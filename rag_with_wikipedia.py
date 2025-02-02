# rag_with_wikipedia.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from functions import search_wikipedia  # Import the search_wikipedia function

def ingest_wikipedia_content(query):
    """
    Fetches content from Wikipedia, processes it, and stores it in a vector database.
    """
    # Fetch content from Wikipedia
    content = search_wikipedia(query)
    if content.startswith("I couldn't find") or content.startswith("An error occurred"):
        print(content)  # Handle errors
        return

    # Split the content into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_text(content)
    print(f"Split Wikipedia content into {len(chunks)} chunks.")

    # Create embeddings and store in Chroma
    embedding = FastEmbedEmbeddings()
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embedding,
        persist_directory="./wikipedia_chroma_db"  # Directory to store the vector database
    )
    print("Wikipedia content stored in Chroma vector database.")

# Example usage
if __name__ == "__main__":
    query = "Python (programming language)"  # Replace with your desired Wikipedia topic
    ingest_wikipedia_content(query)