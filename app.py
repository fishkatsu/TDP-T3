from flask import Flask, render_template, request
from webscraping.main import chatbot_function  # Adjust the import path as needed

app = Flask(__name__)

# Define a route for the homepage
@app.route('/')
def home():
    # Render the homepage template
    return render_template('index.html', answer=None)

# Define a route to handle user input and display responses
@app.route('/ask', methods=['POST'])
def ask():
    # Get the user's question from the form
    user_question = request.form.get('user_question')
    # Pass the question to the chatbot function
    response = chatbot_function(user_question)
    # Render the homepage template with the response
    return render_template('index.html', answer=response)  # Pass the response to the template

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
