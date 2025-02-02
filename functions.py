# functions.py

import wikipediaapi
import os

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