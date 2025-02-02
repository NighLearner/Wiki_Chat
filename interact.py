# interact.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from functions import search_wikipedia  # Import the search_wikipedia function

def ingest_wikipedia_content(query):
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
        chunk_size=2048,
        chunk_overlap=256,
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
    return True

def rag_chain():
    """
    Creates a RAG chain for querying the vector database using the Ollama model.
    """
    # Initialize the Ollama model
    model = ChatOllama(model="llama3.2:1b")

    # Define the prompt template
    prompt = PromptTemplate.from_template(
        """
        <s> [Instructions] You are a friendly assistant. Answer the question based only on the following context. 
        If you don't know the answer, then reply, No Context available for this question {input}. [/Instructions] </s> 
        [Instructions] Question: {input} 
        Context: {context} 
        Answer: [/Instructions]
        """
    )

    # Load the vector store
    embedding = FastEmbedEmbeddings()
    vector_store = Chroma(
        persist_directory="./wikipedia_chroma_db",
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
    document_chain = create_stuff_documents_chain(model, prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    return chain

def ask(query: str):
    """
    Queries the RAG chain and prints the results.
    """
    chain = rag_chain()
    result = chain.invoke({"input": query})
    print("Answer:", result["answer"])
    for doc in result["context"]:
        print("Source:", doc.metadata.get("source", "Unknown"))

def main():
    """
    Main function to interact with the user.
    """
    while True:
        # Prompt the user for a topic
        topic = input("Enter a topic to search on Wikipedia (or type 'exit' to quit): ")
        if topic.lower() == "exit":
            break

        # Ingest the Wikipedia content
        if not ingest_wikipedia_content(topic):
            continue  # Skip to the next iteration if ingestion fails

        # Chat loop for the topic
        while True:
            query = input(f"Ask a question about '{topic}' (or type 'back' to search a new topic): ")
            if query.lower() == "back":
                break
            ask(query)

if __name__ == "__main__":
    main()