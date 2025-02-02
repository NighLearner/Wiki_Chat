# search.py

import wikipediaapi
import os

# Initialize Wikipedia API with a custom user agent
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='MyWikipediaBot/1.0 (https://example.com; myemail@example.com)'
)

# Define the folder to store Wikipedia content
OUTPUT_FOLDER = "wikipedia_content"

# Create the output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def search_wikipedia(query):
    """
    Searches Wikipedia for the given query, retrieves the full content of the page,
    saves it to a text file in the specified folder, and returns the content.
    """
    try:
        # Fetch the page
        page = wiki_wiki.page(query)

        # Check if the page exists
        if not page.exists():
            return f"I couldn't find any information on '{query}'."

        # Get the full content of the page
        content = page.text

        # Save the content to a text file in the output folder
        filename = f"{query.replace(' ', '_')}.txt"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(content)

        print(f"Wikipedia content saved to {filepath}")
        return content  # Return the content for further use

    except Exception as e:
        return f"An error occurred: {str(e)}"