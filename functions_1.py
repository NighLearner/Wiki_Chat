# functions.py

import wikipediaapi
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Initialize Wikipedia API with a custom user agent
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='MyWikipediaBot/1.0 (https://example.com; myemail@example.com)'
)

def search_wikipedia(query):
    """
    Searches Wikipedia for the given query, retrieves the full content of the page,
    and returns the content as a string.
    """
    try:
        # Fetch the page
        page = wiki_wiki.page(query)

        # Check if the page exists
        if not page.exists():
            return f"I couldn't find any information on '{query}'."

        # Get the full content of the page
        content = page.text
        return content  # Return the content for further use

    except Exception as e:
        return f"An error occurred: {str(e)}"

def ingest_wikipedia_content(query, persist_directory="./wikipedia_chroma_db"):
    """
    Fetches content from Wikipedia, processes it, and stores it in a vector database.
    """
    # Fetch content from Wikipedia
    content = search_wikipedia(query)
    if content.startswith("I couldn't find") or content.startswith("An error occurred"):
        print(content)  # Handle errors
        return False

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
        persist_directory=persist_directory  # Directory to store the vector database
    )
    print(f"Wikipedia content stored in Chroma vector database at {persist_directory}.")
    return True

def rag_chain(persist_directory="./wikipedia_chroma_db"):
    """
    Creates a RAG chain using two Ollama models:
    - llama3.2:1b as the retriever.
    - llama3.2:latest as the generator.
    """
    # Initialize the retriever model (llama3.2:1b)
    retriever_model = ChatOllama(model="llama3.2:1b")

    # Initialize the generator model (llama3.2:latest )
    generator_model = ChatOllama(model="llama3.2:latest ")

    # Define the prompt template for the generator
    prompt = PromptTemplate.from_template(
        """
        <s> [Instructions] You are a friendly assistant. Answer the question in a comprehensive and detailed manner based only on the following context. 
        If you don't know the answer, then reply, No Context available for this question {input}. [/Instructions] </s> 
        [Instructions] Question: {input} 
        Context: {context} 
        Answer: [/Instructions]
        """
    )

    # Load the vector store
    embedding = FastEmbedEmbeddings()
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    # Create the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.5,
        },
    )

    # Create the document chain and retrieval chain
    document_chain = create_stuff_documents_chain(generator_model, prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    return chain

def ask(query: str, persist_directory="./wikipedia_chroma_db"):
    """
    Queries the RAG chain and prints the results.
    """
    chain = rag_chain(persist_directory)
    result = chain.invoke({"input": query})
    print("Answer:", result["answer"])
    for doc in result["context"]:
        print("Source:", doc.metadata.get("source", "Unknown"))