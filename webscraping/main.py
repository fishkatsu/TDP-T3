# from bs4 import BeautifulSoup
# import requests
from sentence_transformers import SentenceTransformer, util
import csv
import torch
from sentence_transformers.util import pytorch_cos_sim
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from .labelquestion import process_faq_csv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load a pre-trained Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Maintain a list to store the history of user queries and corresponding responses
conversation_history = []

# Load FAQ data and compute embeddings
faq_data = []
with open("faq_data.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    faq_embeddings = []
    for row in reader:
        faq_data.append({"Question": row["Question"], "Answer": row["Answer"]})
        faq_embeddings.append(model.encode(row["Question"]))

# Convert faq_embeddings list to tensor for faster operation
faq_embeddings = torch.tensor(np.array(faq_embeddings))


# Modify the chatbot_function to handle suggestions when scores are close
def chatbot_function(user_query, faq_data):
    # # Retrieve FAQ data from the website
    # url = "https://www.swinburneonline.edu.au/faqs/"
    # response = requests.get(url)

    # if response.status_code == 200:
    #     soup = BeautifulSoup(response.text, "html.parser")

    #     # Find all the FAQ cards
    #     faq_cards = soup.find_all("div", class_="card")

    #     # Initialize a list to store FAQ data
    #     faq_data = []

    #     # Collect FAQ questions and answers
    #     for card in faq_cards:
    #         card_text = card.get_text(separator=" ")
    #         parts = card_text.split("A.", 1)
    #         if len(parts) == 2:
    #             question = parts[0].strip().replace("Q.", "").strip()
    #             answer = parts[1].strip()
    #             faq_data.append({"Question": question, "Answer": answer})

    # Define the threshold for a high similarity score
    threshold = 0.8

    user_embedding = model.encode(user_query, convert_to_tensor=True)

    # Compute cosine similarities in batch
    similarities = pytorch_cos_sim(user_embedding, faq_embeddings)[0]
    similarities = (
        similarities.cpu().numpy()
    )  # Convert tensor to numpy for easier handling

    # Generate similarity scores list using the pre-computed batch similarities
    similarity_scores = [
        {"Question": faq["Question"], "Similarity Score": score}
        for faq, score in zip(faq_data, similarities)
    ]

    # # Calculate similarity scores for all FAQ questions
    # similarity_scores = []

    # # Loop through all FAQ questions
    # for faq in faq_data:
    #     # Calculate the similarity score between the user query and the FAQ question
    #     similarity_score = util.pytorch_cos_sim(
    #         model.encode(user_query, convert_to_tensor=True),
    #         model.encode(faq["Question"], convert_to_tensor=True),
    #     )
    #     similarity_scores.append(
    #         {
    #             "Question": faq["Question"],
    #             "Similarity Score": similarity_score.item(),
    #         }
    #     )

    # Output similarity scores to a CSV file (debugging)
    csv_file_path = "faq_similarity_scores.csv"
    with open(csv_file_path, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["Question", "Similarity Score"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(similarity_scores)

    # Sort similarity scores
    sorted_similarity_scores = sorted(
        similarity_scores, key=lambda x: x["Similarity Score"], reverse=True
    )
    top_three_questions = sorted_similarity_scores[:3]

    # Define a flag to track whether an answer has been displayed
    answer_displayed = False

    # Check if there is at least one high similarity score above the threshold
    top_score = (
        top_three_questions[0]["Similarity Score"] if sorted_similarity_scores else 0
    )
    if top_score >= threshold and not answer_displayed:
        # Filter questions with scores close to the top score
        close_scores = [
            faq
            for faq in top_three_questions
            if top_score - faq["Similarity Score"] <= 0.2
        ]

        if len(close_scores) == 1:
            # If there's only one, return its answer
            selected_answer = next(
                faq["Answer"]
                for faq in faq_data
                if faq["Question"] == close_scores[0]["Question"]
            )
            # Comment because Evaluation lack the performance
#             selected_question = next(
#                 faq["Question"]
#                 for faq in faq_data
#                 if faq["Question"] == close_scores[0]["Question"]
#             )
#
#             # Evaluate_model function
#             # Define a list of language models to compare
#             compare_model_names = ["all-MiniLM-L6-v2", "bert-base-uncased", "roberta-base"]
#             dataset = [{"user_query": user_query, "response_question": selected_question, "correct_label": 1}]
#             # accuracy, f1, precision, recall = evaluate_model(model, dataset)
#             # # Print or use the evaluation results
#             # print(f"Accuracy: {accuracy}")
#             # print(f"F1 Score: {f1}")
#             # print(f"Precision: {precision}")
#             # print(f"Recall: {recall}")
#
#             # Iterate over the language models and evaluate each one
#             results = {}
#             for model_name in compare_model_names:
#                 test_model = SentenceTransformer(model_name)
#                 accuracy, f1, precision, recall = evaluate_model(test_model, dataset)
#                 results[model_name] = {
#                     "Accuracy": accuracy,
#                     "F1 Score": f1,
#                     "Precision": precision,
#                     "Recall": recall,
#                 }
#
#             # Print or save the results for model comparison
#             for model_name, metrics in results.items():
#                 print(f"Model: {model_name}")
#                 for metric, value in metrics.items():
#                     print(f"{metric}: {value}")
#                 print()
            answer_displayed = True
            return selected_answer, []

    # df = process_faq_csv()
    # print(df["Topic"])

    # Check if the user query is totally out of scope
    if top_score < 0.5:
        # topics = process_faq_csv()
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

    # else:
    #     return "Failed to retrieve the web page. Status code:", response.status_code

# Function to display the conversation history
def display_conversation_history():
    print("Conversation History:")
    for i, interaction in enumerate(conversation_history, 1):
        print(f"{i}. User: {interaction['User Query']}")
        print(f"   Response: {interaction['Response']}")
    print("\n")

# Function to evaluates the model's performance on the dataset
# def evaluate_classification_model(model, tokenizer, dataset):
#     from transformers import TrainingArguments, Trainer

#     training_args = TrainingArguments(
#         output_dir="./results",
#         evaluation_strategy="steps",
#         eval_steps=500,
#         per_device_eval_batch_size=32,
#         save_steps=500,
#         num_train_epochs=3,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         data_collator=default_data_collator,
#         compute_metrics=compute_metrics,
#         train_dataset=dataset["train"],
#         eval_dataset=dataset["test"],
#     )

#     results = trainer.evaluate()
#     return results
def evaluate_model(model, dataset):
    predictions = []
    ground_truth = []

    for row in dataset:
        user_query = row["user_query"]
        bot_response = row["response_question"]

        # Use your evaluation pipeline to compare user_query and bot_response
        # similarity_score = model.encode(row["Question"]), (row["Answer"])
        # similarity_score = model.encode([user_query], [bot_response])
        user_query_embedding = model.encode(user_query)
        response_question_embedding = model.encode(bot_response)

        # Calculate the similarity score between the embeddings
        similarity_score = util.pytorch_cos_sim(user_query_embedding, response_question_embedding)

        threshold = 0.8
        # You can set a threshold for similarity_score to classify responses
        if similarity_score > threshold:
            predicted_label = 1  # Correct response
        else:
            predicted_label = 0  # Incorrect response

        predictions.append(predicted_label)
        ground_truth.append(row['correct_label'])

    # Calculate evaluation metrics
    accuracy = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)

    return accuracy, f1, precision, recall