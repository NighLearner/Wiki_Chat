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
from typing import Optional, Dict, Any

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='MyWikipediaBot/1.0 (https://example.com; myemail@example.com)'
)

def search_wikipedia(query: str) -> str:
    """
    Searches Wikipedia for the given query and returns the content.
    
    Args:
        query (str): Search term for Wikipedia
    
    Returns:
        str: Wikipedia page content or error message
    """
    try:
        page = wiki_wiki.page(query)
        if not page.exists():
            return f"No Wikipedia page found for '{query}'."
        return page.text
    except Exception as e:
        return f"Error accessing Wikipedia: {str(e)}"

def ingest_wikipedia_content(query: str, persist_directory: str = "./wikipedia_chroma_db") -> bool:
    """
    Processes Wikipedia content and stores it in a vector database.
    
    Args:
        query (str): Topic to search for
        persist_directory (str): Directory to store the vector database
    
    Returns:
        bool: True if ingestion was successful, False otherwise
    """
    try:
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Fetch content
        content = search_wikipedia(query)
        if content.startswith(("No Wikipedia page found", "Error accessing Wikipedia")):
            print(content)
            return False

        # Split content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,  # Smaller chunks for better retrieval
            chunk_overlap=50,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_text(content)
        
        if not chunks:
            print("No content chunks were generated.")
            return False

        print(f"Split content into {len(chunks)} chunks")

        # Create and store embeddings
        embedding = FastEmbedEmbeddings()
        vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=embedding,
            persist_directory=persist_directory,
            collection_metadata={"topic": query}  # Add metadata for tracking
        )
        vector_store.persist()
        print(f"Content stored in vector database at {persist_directory}")
        return True

    except Exception as e:
        print(f"Error during content ingestion: {str(e)}")
        return False

def create_rag_chain(persist_directory: str = "./wikipedia_chroma_db") -> Any:
    """
    Creates a RAG chain using llama3.2 models.
    
    Args:
        persist_directory (str): Directory where the vector database is stored
    
    Returns:
        Chain: The configured RAG chain
    """
    # Configure retriever model (llama3.2:1b)
    retriever_model = ChatOllama(
        model="llama3.2:1b",
        temperature=0.1,  # Lower temperature for more focused retrieval
        streaming=True
    )

    # Configure generator model (llama3.2:latest)
    generator_model = ChatOllama(
        model="llama3.2:latest",
        temperature=0.7,  # Higher temperature for more creative responses
        streaming=True
    )

    # Enhanced prompt template
    prompt = PromptTemplate.from_template("""
        <s>[INST] You are a knowledgeable assistant. Using only the provided context, 
        answer the question thoroughly and accurately. If the context doesn't contain 
        enough information, respond with "I don't have enough context to answer this 
        question completely."

        Question: {input}
        
        Context: {context}
        
        Please provide a detailed answer based on the above context. [/INST]</s>
    """)

    # Initialize vector store and retriever
    try:
        embedding = FastEmbedEmbeddings()
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )
        
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,  # Retrieve more chunks for better context
                "score_threshold": 0.3,  # Lower threshold for broader retrieval
            }
        )

        # Create and return the chain
        document_chain = create_stuff_documents_chain(generator_model, prompt)
        return create_retrieval_chain(retriever, document_chain)

    except Exception as e:
        print(f"Error creating RAG chain: {str(e)}")
        return None

def ask(query: str, persist_directory: str = "./wikipedia_chroma_db") -> None:
    """
    Queries the RAG chain and displays the results.
    
    Args:
        query (str): Question to ask about the topic
        persist_directory (str): Directory containing the vector database
    """
    try:
        chain = create_rag_chain(persist_directory)
        if chain is None:
            print("Error: Could not create RAG chain")
            return

        print("\nThinking...")
        result = chain.invoke({"input": query})
        
        print("\nAnswer:", result["answer"])
        print("\nSources used:")
        for i, doc in enumerate(result["context"], 1):
            print(f"{i}. Score: {doc.metadata.get('score', 'N/A'):.3f}")
            print(f"   Content: {doc.page_content[:150]}...")
            print()

    except Exception as e:
        print(f"Error processing question: {str(e)}")