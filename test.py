# test.py

from search import search_wikipedia  # Import the optimized function

# Simple loop to interact with the user
while True:
    user_input = input("Enter a topic to search on Wikipedia (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break

    # Call the search_wikipedia function
    result = search_wikipedia(user_input)
    print("Result:", result[:500])  # Print the first 500 characters of the result for brevity