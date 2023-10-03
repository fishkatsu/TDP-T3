import csv
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim

# Load a pre-trained Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

<<<<<<< HEAD
# Load FAQ data and compute embeddings
faq_data = []
faq_embeddings = []

with open("faq_data.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        faq_data.append({"Question": row["Question"], "Answer": row["Answer"]})
        faq_embeddings.append(model.encode(row["Question"]))

# Convert faq_embeddings list to tensor for faster operation
faq_embeddings = torch.tensor(faq_embeddings)

=======
# Maintain a list to store the history of user queries and corresponding responses
conversation_history = []
>>>>>>> main

def chatbot_function(user_query, faq_data):
    threshold = 0.8

    user_embedding = model.encode(user_query, convert_to_tensor=True)

    # Compute cosine similarities in batch
    similarities = pytorch_cos_sim(user_embedding, faq_embeddings)[0]  # <-- 修复这里
    top_indices = similarities.argsort(descending=True)[:3]

    # Check the top similarity score
    top_score = similarities[top_indices[0]].item()

    if top_score >= threshold:
        # Return the answer corresponding to the highest similarity
        return faq_data[top_indices[0]]["Answer"], []

<<<<<<< HEAD
    # Return suggestions
    return None, [faq_data[idx]["Question"] for idx in top_indices]
=======
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
        top_score = top_three_questions[0]["Similarity Score"] if sorted_similarity_scores else 0
        if top_score >= threshold and not answer_displayed:
            # Filter questions with scores close to the top score
            close_scores = [faq for faq in top_three_questions if top_score - faq["Similarity Score"] <= 0.2]

            if len(close_scores) == 1:
                # If there's only one, return its answer
                selected_answer = next(faq["Answer"] for faq in faq_data if faq["Question"] == close_scores[0]["Question"])
                answer_displayed = True
                return selected_answer, []


         # Check if the user query is totally out of scope
        if top_score < 0.5:
            return "Sorry, I don't understand your question. Please try again.", []

        # Check if suggestions should be displayed
        if not answer_displayed:
            # Return the top three questions as suggestions
            suggestions = [faq["Question"] for faq in top_three_questions]
            # Store user query and suggestions in conversation history
            conversation_history.append({"User Query": user_query, "Response": suggestions})
            return None, suggestions
        else:
            # If an answer has been displayed, return an empty suggestion list
            return None, []
            
    else:
        return "Failed to retrieve the web page. Status code:", response.status_code

        
# Function to display the conversation history
def display_conversation_history():
    print("Conversation History:")
    for i, interaction in enumerate(conversation_history, 1):
        print(f"{i}. User: {interaction['User Query']}")
        print(f"   Response: {interaction['Response']}")
    print("\n")
>>>>>>> main
