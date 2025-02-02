# interact.py

from functions import ingest_wikipedia_content, ask

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