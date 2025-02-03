# interact.py

import os
from functions import ingest_wikipedia_content, ask
from typing import List, Dict

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the application header."""
    print("\n=== Wikipedia RAG Chat ===")
    print("Chat with Wikipedia articles using LLM!\n")

def print_commands():
    """Print available commands."""
    print("\nAvailable commands:")
    print("- 'back': Return to topic selection")
    print("- 'exit': Quit the application")
    print("- 'help': Show these commands")
    print("- 'clear': Clear chat history\n")

def main():
    """
    Main function to interact with the RAG system.
    """
    clear_screen()
    print_header()
    
    # Initialize chat history
    chat_history: List[Dict[str, str]] = []
    current_topic = None

    while True:
        try:
            # Topic selection
            print("\nEnter a topic to search on Wikipedia")
            print("(or type 'exit' to quit)")
            print("-" * 40)
            topic = input("Topic: ").strip()
            
            if topic.lower() == 'exit':
                print("\nThank you for using Wikipedia RAG Chat!")
                break
                
            if not topic:
                print("\nPlease enter a valid topic.")
                continue

            # Clear chat history when switching to a new topic
            if current_topic != topic:
                chat_history.clear()
                current_topic = topic
                print("\nChat history cleared for new topic.")

            # Ingest Wikipedia content
            print(f"\nFetching and processing information about '{topic}'...")
            if not ingest_wikipedia_content(topic):
                print("\nFailed to process the topic. Please try another one.")
                current_topic = None  # Reset current topic on failure
                continue

            print(f"\nSuccess! You can now ask questions about '{topic}'")
            print_commands()

            # Question-answering loop
            while True:
                print("\nAsk a question or use a command")
                print(f"Current topic: {current_topic}")
                print(f"Chat history length: {len(chat_history)//2} exchanges")  # Each exchange has 2 messages
                print("-" * 40)
                query = input("Question: ").strip()
                
                if not query:
                    continue
                    
                query_lower = query.lower()
                if query_lower == 'back':
                    clear_screen()
                    print_header()
                    break
                elif query_lower == 'exit':
                    print("\nThank you for using Wikipedia RAG Chat!")
                    return
                elif query_lower == 'help':
                    print_commands()
                    continue
                elif query_lower == 'clear':
                    chat_history.clear()
                    print("\nChat history cleared.")
                    continue
                
                # Process the question
                result = ask(query, chat_history)
                
                # Only add to history if we got a meaningful response
                if result["context"] and not result["answer"].startswith(("I don't have enough information", "No relevant information")):
                    chat_history.append({"role": "user", "content": query})
                    chat_history.append({"role": "assistant", "content": result["answer"]})

        except KeyboardInterrupt:
            print("\n\nExiting gracefully...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        print("The application will now exit.")