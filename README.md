# Wikipedia RAG with Ollama and Chroma

This project allows you to search Wikipedia, store the content in a vector database (Chroma), and interact with the content using a Retrieval-Augmented Generation (RAG) chain powered by the Ollama language model.

## Features

- **Fetch Wikipedia Content**: Retrieve the full content of any Wikipedia page.
- **Store in Vector Database**: Process and store the content in a Chroma vector database for efficient querying.
- **Query with RAG**: Use an Ollama model to answer questions based on the stored Wikipedia content.
- **Interactive Interface**: A command-line interface (CLI) for searching topics and asking questions.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- Required Python libraries (install via `pip install -r requirements.txt`)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/wikipedia-rag.git
   cd wikipedia-rag
