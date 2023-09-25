from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer, util
import csv

# Load a pre-trained Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Modify the chatbot_function to handle suggestions when scores are close
def chatbot_function(user_query, faq_data):
    # Retrieve FAQ data from the website
    url = "https://www.swinburneonline.edu.au/faqs/"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all the FAQ cards
        faq_cards = soup.find_all("div", class_="card")

        # Initialize a list to store FAQ data
        faq_data = []

        # Collect FAQ questions and answers
        for card in faq_cards:
            card_text = card.get_text(separator=" ")
            parts = card_text.split("A.", 1)
            if len(parts) == 2:
                question = parts[0].strip().replace("Q.", "").strip()
                answer = parts[1].strip()
                faq_data.append({"Question": question, "Answer": answer})

        # Define the threshold for a high similarity score
        threshold = 0.8 
        
        # Calculate similarity scores for all FAQ questions
        similarity_scores = []

        # Loop through all FAQ questions
        for faq in faq_data:
            # Calculate the similarity score between the user query and the FAQ question
            similarity_score = util.pytorch_cos_sim(
                model.encode(user_query, convert_to_tensor=True),
                model.encode(faq["Question"], convert_to_tensor=True),
            )
            similarity_scores.append(
                {
                    "Question": faq["Question"],
                    "Similarity Score": similarity_score.item(),
                }
            )

        # Output similarity scores to a CSV file (debugging)
        csv_file_path = "faq_similarity_scores.csv"
        with open(csv_file_path, "w", newline="", encoding="utf-8") as csv_file:
            fieldnames = ["Question", "Similarity Score"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(similarity_scores)

        # Sort similarity scores
        sorted_similarity_scores = sorted(similarity_scores, key=lambda x: x["Similarity Score"], reverse=True)
        top_three_questions = sorted_similarity_scores[:3]

        # Define a flag to track whether an answer has been displayed
        answer_displayed = False

        # Check if there is at least one high similarity score above the threshold
        top_score = top_three_questions[0]["Similarity Score"]
        if top_score >= threshold and not answer_displayed:
            # Filter questions with scores close to the top score
            close_scores = [faq for faq in top_three_questions if top_score - faq["Similarity Score"] <= 0.2]

            if len(close_scores) == 1:
                # If there's only one, return its answer
                selected_answer = next(faq["Answer"] for faq in faq_data if faq["Question"] == close_scores[0]["Question"])
                answer_displayed = True
                return selected_answer, []

        # Check if suggestions should be displayed
        if not answer_displayed:
            # If an answer hasn't been displayed, return suggestions
            suggestions = [faq["Question"] for faq in top_three_questions]
            return None, suggestions
        else:
            # If an answer has been displayed, return an empty suggestion list
            return None, []

    else:
        return "Failed to retrieve the web page. Status code:", response.status_code
