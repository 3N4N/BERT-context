import torch

def tokenize_sentence(tokenizer, sentence):
    input_tokens = sentence.split()
    tokens = ['[CLS]']
    map_tokens = {}
    for t in input_tokens:
        map_tokens[t] = len(tokens)
        tokens.extend(tokenizer.tokenize(t))
    tokens.append('[SEP]')
    return tokens, map_tokens

def get_word_embedding_from_sentence(model,tokenizer,sentence,word):
    assert word in sentence, f"word '{word}' not in sentence"
    tokens, map_tokens = tokenize_sentence(tokenizer, sentence)
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    segments_ids = [0] * len(tokens)
    tokens_tensor = torch.tensor([tokens_ids])
    segments_tensor = torch.tensor([segments_ids])
    with torch.no_grad():
        encoded_layers = model(tokens_tensor, segments_tensor)
    hidden_states = encoded_layers.hidden_states
    token_embeddings = []
    for token_i in range(len(tokens)):
        embeddings = []
        for layer_i in range(len(hidden_states)):
            vec = hidden_states[layer_i][0][token_i]
            embeddings.append(vec)
        token_embeddings.append(embeddings)
    assert word in map_tokens, f"word '{word}' not in sentence"
    token_idx = map_tokens[word]
    # print(tokens)
    # print(word)
    # print(token_idx)
    return token_embeddings[token_idx][-1]
