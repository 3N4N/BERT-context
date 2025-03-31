import torch
import numpy as np
import pandas as pd
import json
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import re
from tqdm import tqdm
import os


class BertLayerWiCEvaluator:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()  # Set model to evaluation mode
        self.num_layers = (
            self.model.config.num_hidden_layers + 1
        )  # +1 for embedding layer

    def load_wic_data(self, data_df):
        """
        Load WiC dataset from a pandas DataFrame
        """
        wic_data = []

        for _, row in data_df.iterrows():
            # Extract target word and contexts
            target_word = row["Target"]
            context1 = row["context1"]
            context2 = row["context2"]
            label = 1 if row["label"] == "T" else 0  # Convert T/F to 1/0

            # Clean the contexts by removing HTML tags
            clean_context1 = re.sub(r"</?[^>]+>", "", context1)
            clean_context2 = re.sub(r"</?[^>]+>", "", context2)

            # Extract the exact form of the word from the <b> tags if present
            word1 = re.search(r"<b>([^<]+)</b>", context1)
            word1 = word1.group(1) if word1 else target_word

            word2 = re.search(r"<b>([^<]+)</b>", context2)
            word2 = word2.group(1) if word2 else target_word

            wic_data.append(
                {
                    "target_word": target_word,
                    "word1": word1,
                    "word2": word2,
                    "context1": clean_context1,
                    "context2": clean_context2,
                    "original_context1": context1,
                    "original_context2": context2,
                    "label": label,
                }
            )

        return wic_data

    def get_target_word_embedding(self, sentence, target_word):
        """
        Extract the embedding of the target word in the given sentence from all layers
        """
        # Tokenize the sentence
        inputs = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
        token_ids = inputs["input_ids"][0]
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        # Get all hidden states from the model
        with torch.no_grad():
            outputs = self.model(**inputs)
            # hidden_states includes embedding layer + all transformer layers
            hidden_states = outputs.hidden_states

        # Find the target word's token indices
        target_indices = []
        target_word_pieces = self.tokenizer.tokenize(target_word)

        for i in range(len(tokens) - len(target_word_pieces) + 1):
            if tokens[i : i + len(target_word_pieces)] == target_word_pieces:
                target_indices = list(range(i, i + len(target_word_pieces)))
                break

        if not target_indices:
            # If exact match not found, look for individual pieces
            for i, token in enumerate(tokens):
                # Skip special tokens
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue

                # Check if token is part of the target word (handles subword tokenization)
                if (
                    token.replace("##", "") in target_word.lower()
                    or target_word.lower() in token
                ):
                    target_indices.append(i)

        if not target_indices:
            print(f"Warning: Target word '{target_word}' not found in '{sentence}'")
            print(f"Tokens: {tokens}")
            # Default to first non-special token
            for i, token in enumerate(tokens):
                if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                    target_indices = [i]
                    break

        print("Sentence", sentence)
        print("Target Word", target_word)
        print("Indexes", target_indices)

        # Extract embeddings from each layer for target token(s)
        all_layers_embeddings = []

        for layer_hidden_state in hidden_states:
            # Average embeddings if multiple tokens for the target word
            token_embeddings = [
                layer_hidden_state[0, idx].numpy() for idx in target_indices
            ]
            avg_embedding = np.mean(token_embeddings, axis=0)
            all_layers_embeddings.append(avg_embedding)

        return all_layers_embeddings

    def extract_all_embeddings(self, wic_data):
        """
        Extract embeddings for all WiC instances from all layers
        """
        all_embeddings = []
        labels = []

        for idx, instance in enumerate(tqdm(wic_data, desc="Extracting embeddings")):
            # Get embeddings for both contexts
            context1_embeddings = self.get_target_word_embedding(
                instance["context1"], instance["word1"]
            )
            context2_embeddings = self.get_target_word_embedding(
                instance["context2"], instance["word2"]
            )

            # For each layer
            for layer in range(self.num_layers):
                # Store the embedding pair and label
                embedding_pair = {
                    "id": idx,
                    "target_word": instance["target_word"],
                    "context1": instance["context1"],
                    "context2": instance["context2"],
                    "original_context1": instance.get(
                        "original_context1", instance["context1"]
                    ),
                    "original_context2": instance.get(
                        "original_context2", instance["context2"]
                    ),
                    "layer": layer,
                    "emb1": context1_embeddings[layer],
                    "emb2": context2_embeddings[layer],
                    "label": instance["label"],
                }
                all_embeddings.append(embedding_pair)

                if layer == 0:  # Only add label once
                    labels.append(instance["label"])

        return all_embeddings, labels

    def compute_cosine_similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two vectors
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)

    def evaluate_layers(self, all_embeddings, method="cosine"):
        """
        Evaluate each layer's performance on the WiC task

        Methods:
        - 'cosine': Use cosine similarity with a threshold
        - 'lr': Train a logistic regression classifier
        """
        results = []

        # Group embeddings by layer
        embeddings_by_layer = {}
        for emb in all_embeddings:
            layer = emb["layer"]
            if layer not in embeddings_by_layer:
                embeddings_by_layer[layer] = []
            embeddings_by_layer[layer].append(emb)

        # Evaluate each layer
        for layer, layer_embeddings in sorted(embeddings_by_layer.items()):
            if method == "cosine":
                # Compute cosine similarities
                similarities = []
                true_labels = []
                instance_ids = []

                for emb in layer_embeddings:
                    similarity = self.compute_cosine_similarity(
                        emb["emb1"], emb["emb2"]
                    )
                    similarities.append(similarity)
                    true_labels.append(emb["label"])
                    instance_ids.append(emb["id"])

                # Try different thresholds to find the best one
                best_accuracy = 0
                best_threshold = 0
                best_predictions = []

                for threshold in np.arange(0.7, 1.0, 0.01):
                    pred_labels = [1 if sim >= threshold else 0 for sim in similarities]
                    accuracy = accuracy_score(true_labels, pred_labels)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_threshold = threshold
                        best_predictions = pred_labels

                # Final evaluation with best threshold
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_labels, best_predictions, average="binary"
                )

                # Store instance-level results for this layer
                instance_results = []
                for i, emb in enumerate(layer_embeddings):
                    instance_results.append(
                        {
                            "id": emb["id"],
                            "target_word": emb["target_word"],
                            "similarity": similarities[i],
                            "prediction": best_predictions[i],
                            "true_label": true_labels[i],
                            "correct": best_predictions[i] == true_labels[i],
                        }
                    )

                results.append(
                    {
                        "layer": layer,
                        "method": method,
                        "threshold": best_threshold,
                        "accuracy": best_accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "instance_results": instance_results,
                    }
                )

            elif method == "lr":
                # Use logistic regression
                X = []
                y = []
                instance_ids = []

                for emb in layer_embeddings:
                    # Create feature vector as abs difference
                    feature_vec = np.abs(emb["emb1"] - emb["emb2"])
                    X.append(feature_vec)
                    y.append(emb["label"])
                    instance_ids.append(emb["id"])

                # Perform cross-validation (simplified here - just train/test split)
                unique_ids = list(set(instance_ids))
                np.random.shuffle(unique_ids)
                train_ids = set(unique_ids[: int(0.8 * len(unique_ids))])

                X_train = [
                    X[i] for i, idx in enumerate(instance_ids) if idx in train_ids
                ]
                y_train = [
                    y[i] for i, idx in enumerate(instance_ids) if idx in train_ids
                ]
                X_test = [
                    X[i] for i, idx in enumerate(instance_ids) if idx not in train_ids
                ]
                y_test = [
                    y[i] for i, idx in enumerate(instance_ids) if idx not in train_ids
                ]
                test_indices = [
                    i for i, idx in enumerate(instance_ids) if idx not in train_ids
                ]

                if (
                    len(np.unique(y_train)) > 1
                ):  # Ensure we have both classes in training
                    clf = LogisticRegression(random_state=42, max_iter=1000)
                    clf.fit(X_train, y_train)

                    y_pred = clf.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_test, y_pred, average="binary"
                    )

                    # Store instance-level results for test set
                    instance_results = []
                    for i, test_idx in enumerate(test_indices):
                        instance_results.append(
                            {
                                "id": instance_ids[test_idx],
                                "target_word": layer_embeddings[test_idx][
                                    "target_word"
                                ],
                                "prediction": int(y_pred[i]),
                                "true_label": y_test[i],
                                "correct": int(y_pred[i]) == y_test[i],
                            }
                        )
                else:
                    accuracy = precision = recall = f1 = 0
                    instance_results = []

                results.append(
                    {
                        "layer": layer,
                        "method": method,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "instance_results": instance_results,
                    }
                )

        return results

    def generate_embedding_visualization_data(
        self, all_embeddings, layers_to_include=None
    ):
        """
        Generate t-SNE visualization data for selected layers
        """
        # If no layers specified, use first, middle, and last layer
        if layers_to_include is None:
            layers_to_include = [0, self.num_layers // 2, self.num_layers - 1]

        visualization_data = []

        # Group embeddings by layer
        embeddings_by_layer = {}
        for emb in all_embeddings:
            layer = emb["layer"]
            if layer in layers_to_include:
                if layer not in embeddings_by_layer:
                    embeddings_by_layer[layer] = []
                embeddings_by_layer[layer].append(emb)

        # Get all unique target words
        target_words = set(emb["target_word"] for emb in all_embeddings)

        # Process each layer
        for layer, layer_embeddings in sorted(embeddings_by_layer.items()):
            print(f"Generating t-SNE for layer {layer}...")

            # Extract embeddings
            X1 = np.array([emb["emb1"] for emb in layer_embeddings])
            X2 = np.array([emb["emb2"] for emb in layer_embeddings])
            X = np.vstack([X1, X2])

            # If we have a lot of examples, subsample for t-SNE
            max_samples = 300  # Avoid too many points for t-SNE
            if len(X) > max_samples:
                indices = np.random.choice(len(X), max_samples, replace=False)
                X_subset = X[indices]
            else:
                X_subset = X
                indices = range(len(X))

            # Apply t-SNE
            try:
                tsne = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=min(30, len(X_subset) - 1),
                )
                X_tsne = tsne.fit_transform(X_subset)

                # Map back to original indices if subsampled
                tsne_by_index = {}
                for i, idx in enumerate(indices):
                    if idx < len(X):
                        tsne_by_index[idx] = X_tsne[i].tolist()

                # Create visualization data points
                layer_data = {"layer": layer, "points": []}

                # Add context pairs with t-SNE coordinates
                for i, emb in enumerate(layer_embeddings):
                    # Check if we have t-SNE coordinates for this point
                    emb1_idx = i
                    emb2_idx = i + len(layer_embeddings)

                    if emb1_idx in tsne_by_index and emb2_idx in tsne_by_index:
                        point_data = {
                            "id": emb["id"],
                            "target_word": emb["target_word"],
                            "label": "same" if emb["label"] == 1 else "different",
                            "context1": emb["original_context1"],
                            "context2": emb["original_context2"],
                            "coordinates": [
                                {
                                    "context": 1,
                                    "x": tsne_by_index[emb1_idx][0],
                                    "y": tsne_by_index[emb1_idx][1],
                                },
                                {
                                    "context": 2,
                                    "x": tsne_by_index[emb2_idx][0],
                                    "y": tsne_by_index[emb2_idx][1],
                                },
                            ],
                            "similarity": self.compute_cosine_similarity(
                                emb["emb1"], emb["emb2"]
                            ),
                        }
                        layer_data["points"].append(point_data)

                visualization_data.append(layer_data)

            except Exception as e:
                print(f"Error generating t-SNE for layer {layer}: {str(e)}")

        return {"target_words": list(target_words), "layers": visualization_data}

    def analyze_examples(self, all_embeddings, results, top_n=5):
        """
        Analyze specific examples where higher layers significantly improve or worsen performance
        """
        # Find the best and worst performing layers
        layer_results = sorted(results, key=lambda x: x["accuracy"])
        worst_layer = layer_results[0]["layer"]
        best_layer = layer_results[-1]["layer"]

        print(
            f"Best performing layer: {best_layer} with accuracy {layer_results[-1]['accuracy']:.4f}"
        )
        print(
            f"Worst performing layer: {worst_layer} with accuracy {layer_results[0]['accuracy']:.4f}"
        )

        # Extract embeddings for these layers
        best_layer_embs = [emb for emb in all_embeddings if emb["layer"] == best_layer]
        worst_layer_embs = [
            emb for emb in all_embeddings if emb["layer"] == worst_layer
        ]

        # Compute cosine similarities
        best_sims = [
            (emb["id"], self.compute_cosine_similarity(emb["emb1"], emb["emb2"]), emb)
            for emb in best_layer_embs
        ]
        worst_sims = [
            (emb["id"], self.compute_cosine_similarity(emb["emb1"], emb["emb2"]), emb)
            for emb in worst_layer_embs
        ]

        # Sort by similarity difference between layers
        improved = []
        worsened = []

        for (id1, sim1, emb1), (id2, sim2, emb2) in zip(best_sims, worst_sims):
            if id1 == id2:  # Same example
                sim_diff = abs(sim1 - sim2)
                label = emb1["label"]

                # Check if the predictions would be different
                best_pred = 1 if sim1 >= 0.85 else 0  # Use threshold
                worst_pred = 1 if sim2 >= 0.85 else 0

                if best_pred == label and worst_pred != label:
                    # Best layer correct, worst layer wrong
                    improved.append(
                        {
                            "id": id1,
                            "target_word": emb1["target_word"],
                            "context1": emb1["original_context1"],
                            "context2": emb1["original_context2"],
                            "label": "same" if label == 1 else "different",
                            "best_sim": float(sim1),
                            "worst_sim": float(sim2),
                            "sim_diff": float(sim_diff),
                            "best_layer": int(best_layer),
                            "worst_layer": int(worst_layer),
                        }
                    )
                elif best_pred != label and worst_pred == label:
                    # Best layer wrong, worst layer correct
                    worsened.append(
                        {
                            "id": id1,
                            "target_word": emb1["target_word"],
                            "context1": emb1["original_context1"],
                            "context2": emb1["original_context2"],
                            "label": "same" if label == 1 else "different",
                            "best_sim": float(sim1),
                            "worst_sim": float(sim2),
                            "sim_diff": float(sim_diff),
                            "best_layer": int(best_layer),
                            "worst_layer": int(worst_layer),
                        }
                    )

        # Sort by similarity difference
        improved.sort(key=lambda x: x["sim_diff"], reverse=True)
        worsened.sort(key=lambda x: x["sim_diff"], reverse=True)

        # Get top examples
        improved = improved[:top_n]
        worsened = worsened[:top_n]

        return {
            "improved": improved,
            "worsened": worsened,
            "best_layer": best_layer,
            "worst_layer": worst_layer,
        }

    def save_results_as_json(self, output_path, data):
        """
        Save all analysis results as a single JSON file
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert numpy values to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            else:
                return obj

        # Save the combined data to a single file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(convert_to_serializable(data), f, indent=2)

        print(f"All results saved to {output_path}")


# Main function to run the analysis
def analyze_wic_with_bert_layers(
    data_df, model_name="bert-base-uncased", output_path="wic_bert_analysis.json"
):
    # Initialize the evaluator
    evaluator = BertLayerWiCEvaluator(model_name)

    # Load data
    print("Loading WiC data...")
    wic_data = evaluator.load_wic_data(data_df)

    # Extract embeddings
    print("Extracting embeddings from all BERT layers...")
    all_embeddings, labels = evaluator.extract_all_embeddings(wic_data)

    # Evaluate using cosine similarity
    print("Evaluating layers using cosine similarity...")
    cosine_results = evaluator.evaluate_layers(all_embeddings, method="cosine")

    # Evaluate using logistic regression
    print("Evaluating layers using logistic regression...")
    lr_results = evaluator.evaluate_layers(all_embeddings, method="lr")

    # Generate embedding visualization data for selected layers
    print("Generating embedding visualization data...")
    vis_layers = [
        0,
        evaluator.num_layers // 2,
        evaluator.num_layers - 1,
    ]  # First, middle, last
    vis_data = evaluator.generate_embedding_visualization_data(
        all_embeddings, layers_to_include=vis_layers
    )

    # Analyze specific examples
    print("Analyzing specific examples...")
    example_analysis = evaluator.analyze_examples(all_embeddings, cosine_results)

    # Prepare data for JSON export
    target_words = list(set(emb["target_word"] for emb in all_embeddings))

    # Extract layer performance metrics for easy plotting
    layer_performance = {
        "cosine": {
            "layers": [r["layer"] for r in cosine_results],
            "accuracy": [r["accuracy"] for r in cosine_results],
            "precision": [r["precision"] for r in cosine_results],
            "recall": [r["recall"] for r in cosine_results],
            "f1": [r["f1"] for r in cosine_results],
            "thresholds": [r["threshold"] for r in cosine_results],
        },
        "lr": {
            "layers": [r["layer"] for r in lr_results],
            "accuracy": [r["accuracy"] for r in lr_results],
            "precision": [r["precision"] for r in lr_results],
            "recall": [r["recall"] for r in lr_results],
            "f1": [r["f1"] for r in lr_results],
        },
    }

    # Combine all results into a single object
    combined_data = {
        "model_info": {
            "name": model_name,
            "num_layers": evaluator.num_layers,
            "timestamp": pd.Timestamp.now().isoformat(),
        },
        "target_words": target_words,
        "layer_performance": layer_performance,
        "embedding_visualization": vis_data,
        "example_analysis": example_analysis,
        "cosine_results": cosine_results,
        "lr_results": lr_results,
    }

    # Save to a single JSON file
    evaluator.save_results_as_json(output_path, combined_data)

    print(f"\nAnalysis complete. Results saved to {output_path}")

    return combined_data
