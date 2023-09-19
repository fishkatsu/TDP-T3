from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer, util
import csv

# Load a pre-trained Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Function to retrieve FAQ data and calculate similarity scores
def chatbot_function(user_query):
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

        # Calculate similarity scores for all FAQ questions
        similarity_scores = []

        # Loop through all FAQ questions
        for faq in faq_data:
            # Calculate the similarity score between the user query and the FAQ question
            similarity_score = util.pytorch_cos_sim(
                # Convert the user query and FAQ question to embeddings
                model.encode(user_query, convert_to_tensor=True),
                model.encode(faq["Question"], convert_to_tensor=True),
            )
            similarity_scores.append(
                {
                    "Question": faq["Question"],
                    "Similarity Score": similarity_score.item(),
                }
            )

        # Find the FAQ question with the highest similarity score
        top_question = max(similarity_scores, key=lambda x: x["Similarity Score"])
        selected_answer = next(
            faq["Answer"]
            for faq in faq_data
            if faq["Question"] == top_question["Question"]
        )

        # Output similarity scores to a CSV file (debugging)
        csv_file_path = "faq_similarity_scores.csv"
        with open(csv_file_path, "w", newline="", encoding="utf-8") as csv_file:
            fieldnames = ["Question", "Similarity Score"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(similarity_scores)

        return selected_answer

    else:
        return "Failed to retrieve the web page. Status code:", response.status_code
