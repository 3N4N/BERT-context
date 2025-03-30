import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# Dataset of semantically equivalent word pairs in English and Spanish
word_pairs = [
    # Format: (english_word, spanish_word, english_context, spanish_context, category)
    # Synonyms
    ("happy", "feliz", "I am happy today.", "Estoy feliz hoy.", "synonym"),
    ("sad", "triste", "She looks sad.", "Ella parece triste.", "synonym"),
    ("big", "grande", "That is a big house.", "Esa es una casa grande.", "synonym"),
    ("small", "pequeño", "The small dog barked.", "El perro pequeño ladró.", "synonym"),
    (
        "beautiful",
        "hermoso",
        "What a beautiful view!",
        "¡Qué vista tan hermosa!",
        "synonym",
    ),
    ("fast", "rápido", "He runs very fast.", "Él corre muy rápido.", "synonym"),
    ("slow", "lento", "The turtle is slow.", "La tortuga es lenta.", "synonym"),
    ("hot", "caliente", "The soup is hot.", "La sopa está caliente.", "synonym"),
    ("cold", "frío", "It's cold outside.", "Hace frío afuera.", "synonym"),
    (
        "old",
        "viejo",
        "That is an old building.",
        "Ese es un edificio viejo.",
        "synonym",
    ),
    # Homonyms with different contexts to capture different meanings
    (
        "bank",
        "banco",
        "I went to the bank to withdraw money.",
        "Fui al banco a sacar dinero.",
        "homonym_financial",
    ),
    (
        "bank",
        "orilla",
        "We sat on the bank of the river.",
        "Nos sentamos en la orilla del río.",
        "homonym_riverside",
    ),
    (
        "light",
        "luz",
        "Turn on the light please.",
        "Enciende la luz por favor.",
        "homonym_illumination",
    ),
    (
        "light",
        "ligero",
        "This backpack is very light.",
        "Esta mochila es muy ligera.",
        "homonym_weight",
    ),
    (
        "spring",
        "primavera",
        "Spring is my favorite season.",
        "La primavera es mi estación favorita.",
        "homonym_season",
    ),
    (
        "spring",
        "resorte",
        "The spring in the mechanism broke.",
        "El resorte del mecanismo se rompió.",
        "homonym_coil",
    ),
    (
        "bat",
        "murciélago",
        "The bat flew in the night.",
        "El murciélago voló en la noche.",
        "homonym_animal",
    ),
    (
        "bat",
        "bate",
        "He hit the ball with the bat.",
        "Golpeó la pelota con el bate.",
        "homonym_sports",
    ),
    (
        "book",
        "libro",
        "I read a book yesterday.",
        "Leí un libro ayer.",
        "homonym_reading",
    ),
    (
        "book",
        "reservar",
        "I need to book a hotel room.",
        "Necesito reservar una habitación de hotel.",
        "homonym_reserve",
    ),
]


def get_bert_embedding(text, word, model, tokenizer, device="cpu"):
    """
    Get BERT embedding for a specific word in a context.
    """
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Find the position of the word in the tokenized text
    word_tokens = tokenizer.tokenize(word)
    text_tokens = tokenizer.tokenize(text)

    # Find where the word starts in the tokenized text
    word_indices = []
    for i in range(len(text_tokens) - len(word_tokens) + 1):
        if text_tokens[i : i + len(word_tokens)] == word_tokens:
            word_indices.extend(
                range(i + 1, i + len(word_tokens) + 1)
            )  # +1 because of the CLS token
            break

    if not word_indices:
        raise ValueError(f"Word '{word}' not found in tokenized text")

    # Get the model output
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the hidden states
    hidden_states = outputs.last_hidden_state.squeeze(0).cpu().numpy()

    # Average the embeddings of the word tokens
    word_embedding = hidden_states[word_indices].mean(axis=0)

    return word_embedding


def analyze_cross_lingual_representations():
    """
    Analyze the consistency of multilingual BERT's representations across languages.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load pre-trained multilingual BERT model
    print("Loading multilingual BERT model...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = BertModel.from_pretrained("bert-base-multilingual-cased").to(device)
    model.eval()

    # Process all word pairs
    results = {"embeddings": [], "similarities": [], "categories": [], "words": []}

    print("Processing word pairs...")
    for en_word, es_word, en_context, es_context, category in tqdm(word_pairs):
        try:
            # Get embeddings using the same model for both languages
            en_embedding = get_bert_embedding(
                en_context, en_word, model, tokenizer, device
            )
            es_embedding = get_bert_embedding(
                es_context, es_word, model, tokenizer, device
            )

            # Calculate cosine similarity
            similarity = cosine_similarity([en_embedding], [es_embedding])[0][0]

            # Store results
            results["embeddings"].append(
                {"english": en_embedding.tolist(), "spanish": es_embedding.tolist()}
            )
            results["similarities"].append(
                float(similarity)
            )  # Convert to native Python float
            results["categories"].append(category)
            results["words"].append(
                {
                    "english": en_word,
                    "spanish": es_word,
                    "english_context": en_context,
                    "spanish_context": es_context,
                }
            )

        except Exception as e:
            print(f"Error processing {en_word}/{es_word}: {e}")

    # Calculate statistics by category
    category_stats = {}
    for category in set(results["categories"]):
        indices = [i for i, cat in enumerate(results["categories"]) if cat == category]
        category_stats[category] = {
            "mean_similarity": float(
                np.mean([results["similarities"][i] for i in indices])
            ),
            "std_similarity": float(
                np.std([results["similarities"][i] for i in indices])
            ),
            "count": len(indices),
        }

    results["category_stats"] = category_stats

    # Compute overall statistics
    results["overall_stats"] = {
        "mean_similarity": float(np.mean(results["similarities"])),
        "std_similarity": float(np.std(results["similarities"])),
        "min_similarity": float(np.min(results["similarities"])),
        "max_similarity": float(np.max(results["similarities"])),
    }

    # Perform PCA for visualization
    print("Performing PCA for visualization...")
    all_embeddings = []
    for pair in results["embeddings"]:
        all_embeddings.append(pair["english"])
        all_embeddings.append(pair["spanish"])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(np.array(all_embeddings))

    # Store PCA results
    results["pca"] = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "components": pca.components_.tolist(),
        "projections": [],
    }

    for i in range(len(results["embeddings"])):
        en_proj = pca_result[i * 2].tolist()
        es_proj = pca_result[i * 2 + 1].tolist()
        results["pca"]["projections"].append(
            {
                "english": en_proj,
                "spanish": es_proj,
                "category": results["categories"][i],
                "english_word": results["words"][i]["english"],
                "spanish_word": results["words"][i]["spanish"],
            }
        )

    # Create visualization data
    results["visualizations"] = create_visualizations(results)

    return results


def create_visualizations(results):
    """
    Create visualization data for the web app.
    """
    vis_data = {}

    # Bar chart of similarities by category
    categories = list(results["category_stats"].keys())
    mean_similarities = [
        results["category_stats"][cat]["mean_similarity"] for cat in categories
    ]

    vis_data["category_similarity_bar"] = {
        "type": "bar",
        "x": categories,
        "y": mean_similarities,
        "error_y": [
            results["category_stats"][cat]["std_similarity"] for cat in categories
        ],
    }

    # Scatter plot of PCA projections
    en_x = [proj["english"][0] for proj in results["pca"]["projections"]]
    en_y = [proj["english"][1] for proj in results["pca"]["projections"]]
    es_x = [proj["spanish"][0] for proj in results["pca"]["projections"]]
    es_y = [proj["spanish"][1] for proj in results["pca"]["projections"]]
    categories = [proj["category"] for proj in results["pca"]["projections"]]
    en_words = [proj["english_word"] for proj in results["pca"]["projections"]]
    es_words = [proj["spanish_word"] for proj in results["pca"]["projections"]]

    vis_data["pca_scatter"] = {
        "type": "scatter",
        "data": [
            {
                "x": en_x,
                "y": en_y,
                "mode": "markers+text",
                "marker": {"size": 10, "color": "blue"},
                "text": en_words,
                "name": "English",
            },
            {
                "x": es_x,
                "y": es_y,
                "mode": "markers+text",
                "marker": {"size": 10, "color": "red"},
                "text": es_words,
                "name": "Spanish",
            },
        ],
        "categories": categories,
    }

    # Heatmap of similarities
    vis_data["similarity_heatmap"] = {
        "type": "heatmap",
        "z": results["similarities"],
        "x": [
            pair["english_word"] + "/" + pair["spanish_word"]
            for pair in results["pca"]["projections"]
        ],
        "y": ["Similarity"],
    }

    return vis_data


def save_results_to_json(results, filename="bert_cross_lingual_analysis.json"):
    """
    Save the analysis results to a JSON file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {filename}")


def main():
    """
    Main function to run the analysis and save results.
    """
    print("Analyzing multilingual BERT's cross-lingual representations...")
    results = analyze_cross_lingual_representations()

    print("Saving results to JSON...")
    save_results_to_json(results, "mbert_cross_lingual_analysis.json")

    print("Analysis complete!")


if __name__ == "__main__":
    main()
