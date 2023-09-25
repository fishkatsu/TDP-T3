from flask import Flask, render_template, request
from webscraping.main import chatbot_function  # Adjust the import path as needed
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# Load and process FAQ data (move this part from main.py to app.py)
url = "https://www.swinburneonline.edu.au/faqs/"
response = requests.get(url)
faq_data = []

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find all the FAQ cards
    faq_cards = soup.find_all("div", class_="card")

    # Collect FAQ questions and answers
    for card in faq_cards:
        card_text = card.get_text(separator=" ")
        parts = card_text.split("A.", 1)
        if len(parts) == 2:
            question = parts[0].strip().replace("Q.", "").strip()
            answer = parts[1].strip()
            faq_data.append({"Question": question, "Answer": answer})
else:
    print("Failed to retrieve the web page. Status code:", response.status_code)

# Define a route for the homepage
@app.route('/')
def home():
    # Check if there is an answer to display
    greeting = "Hello! How may I assist you with Swinburne Online's FAQ today?" 
    # Render the homepage template and pass the answer to it
    return render_template('index.html', answer=greeting)

# Define a route to handle user input and display responses
@app.route('/ask', methods=['POST'])
def ask():
    # Get the user's question from the form
    user_question = request.form.get('user_question')

    # Check if user selected a suggestion
    selected_suggestion = request.form.get('selected_suggestion')

    if selected_suggestion:
        # User selected a suggestion, find the corresponding answer
        selected_answer = next(faq["Answer"] for faq in faq_data if faq["Question"] == selected_suggestion)
        return render_template('index.html', answer=selected_answer, suggestions=None)

    # If no suggestion is selected, pass the question and FAQ data to the chatbot function
    response, suggestions = chatbot_function(user_question, faq_data)

    if suggestions:
        # Define the default message if suggestions are available
        default_message = "Here are some suggestions for you."
    else:
        # If no suggestions, set the default message to an empty string
        default_message = ""

    # Render the homepage template with the default message, response, and suggestions
    return render_template('index.html', answer=default_message if default_message else response, suggestions=suggestions)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
