from flask import Flask, render_template, request
from webscraping.main import chatbot_function  # Adjust the import path as needed
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# Load and process FAQ data (move this part from main.py to app.py)
url = "https://www.swinburneonline.edu.au/faqs/"
response = requests.get(url)
faq_data = []
# Initialize a list to store the conversation history
conversation_history = []

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
    user_question = request.form.get('user_question', '')
    
    # If the user has selected a suggestion, use it as the user_question
    selected_suggestion = request.form.get('selected_suggestion')
    if selected_suggestion:
        user_question = selected_suggestion
    
    # Call chatbot_function to get a response and suggestions
    response, suggestions = chatbot_function(user_question, faq_data)
    
    # If a suggestion was selected, find the corresponding answer
    if selected_suggestion:
        response = next(faq["Answer"] for faq in faq_data if faq["Question"] == selected_suggestion)
        suggestions = None  # No need to display suggestions if one is selected
    
    # Update conversation history
    conversation_history.append({"User Query": user_question, "Response": response or suggestions})
    
    # Render the template with all necessary variables
    return render_template(
        'index.html', 
        conversation_history=conversation_history,
        answer=response or "Here are some suggestions for you.", 
        suggestions=suggestions, 
        user_question=user_question
    )

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
