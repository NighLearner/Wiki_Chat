# interact.py

import os
from functions_3 import ingest_wikipedia_content, ask

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
    print("- 'help': Show these commands\n")

def main():
    """
    Main function to interact with the RAG system.
    """
    clear_screen()
    print_header()
    
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

            # Ingest Wikipedia content
            print(f"\nFetching and processing information about '{topic}'...")
            if not ingest_wikipedia_content(topic):
                print("\nFailed to process the topic. Please try another one.")
                continue

            print(f"\nSuccess! You can now ask questions about '{topic}'")
            print_commands()

            # Question-answering loop
            while True:
                print("\nAsk a question or use a command")
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
                
                # Process the question
                ask(query)

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