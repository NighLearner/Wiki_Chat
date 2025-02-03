# functions_3.py

import wikipediaapi
import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from typing import Optional, Dict, Any, List

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='MyWikipediaBot/1.0 (https://example.com; myemail@example.com)'
)

def clear_database(persist_directory: str) -> None:
    """
    Clears the existing vector database directory.
    
    Args:
        persist_directory (str): Path to the directory containing the vector database
    """
    try:
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            print(f"Cleared previous database at {persist_directory}")
    except Exception as e:
        print(f"Error clearing database: {str(e)}")

def search_wikipedia(query: str) -> str:
    """
    Searches Wikipedia for the given query and returns the content.
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
    First clears any existing database to ensure clean data for the new topic.
    """
    try:
        # Clear existing database
        clear_database(persist_directory)
        
        # Create fresh directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Fetch new content
        content = search_wikipedia(query)
        if content.startswith(("No Wikipedia page found", "Error accessing Wikipedia")):
            print(content)
            return False

        # Split content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_text(content)
        
        if not chunks:
            print("No content chunks were generated.")
            return False

        print(f"Split content into {len(chunks)} chunks")

        # Create new embeddings and store
        embedding = FastEmbedEmbeddings()
        vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=embedding,
            persist_directory=persist_directory,
            collection_metadata={"topic": query}
        )
        vector_store.persist()
        print(f"New content stored in vector database at {persist_directory}")
        return True

    except Exception as e:
        print(f"Error during content ingestion: {str(e)}")
        return False

def create_rag_chain(persist_directory: str = "./wikipedia_chroma_db") -> Any:
    """
    Creates a RAG chain using llama3.2 models with balanced retrieval settings.
    """
    retriever_model = ChatOllama(
        model="llama3.2:1b",
        temperature=0.1,
        streaming=True
    )

    generator_model = ChatOllama(
        model="llama3.2:latest",
        temperature=0.7,
        streaming=True
    )

    prompt = PromptTemplate.from_template("""
        <s>[INST] You are a helpful assistant. Give direct, clear answers using the information below.
        If you don't have enough information to answer, simply say:
        "I don't have enough information to answer this question."
        
        Chat History:
        {chat_history}
        
        Question: {input}
        
        Context: {context}
        
        Answer: [/INST]</s>
    """)

    try:
        embedding = FastEmbedEmbeddings()
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )
        
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 4,
                "score_threshold": 0.3,
            }
        )

        document_chain = create_stuff_documents_chain(generator_model, prompt)
        return create_retrieval_chain(retriever, document_chain)

    except Exception as e:
        print(f"Error creating RAG chain: {str(e)}")
        return None

def ask(query: str, chat_history: List[Dict[str, str]], persist_directory: str = "./wikipedia_chroma_db") -> Dict[str, Any]:
    """
    Queries the RAG chain with chat history and returns the result.
    """
    try:
        chain = create_rag_chain(persist_directory)
        if chain is None:
            print("Error: Could not create RAG chain")
            return {"answer": "Error: Could not create RAG chain", "context": []}

        print("\nSearching for relevant information...")
        result = chain.invoke({"input": query, "chat_history": chat_history})
        
        if not result.get("context"):
            print("\nNo relevant information found to answer this question.")
            return {"answer": "No relevant information found.", "context": []}

        print("\nAnswer:", result["answer"])
        print("\nSources used:")
        for i, doc in enumerate(result["context"], 1):
            score = doc.metadata.get('score', 'N/A')
            if isinstance(score, (int, float)):
                print(f"{i}. Relevance Score: {score:.3f}")
            print(f"   Content: {doc.page_content[:150]}...")
            print()

        return result

    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return {"answer": f"Error: {str(e)}", "context": []}