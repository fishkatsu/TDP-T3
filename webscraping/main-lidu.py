import csv
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim

# Load a pre-trained Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

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

    # Return suggestions
    return None, [faq_data[idx]["Question"] for idx in top_indices]
