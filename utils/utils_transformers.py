import re
import torch

def tokenize_sentence(tokenizer, sentence):
    input_tokens = sentence.split()
    tokens = ['[CLS]']
    map_tokens = {}
    prev_token = None
    for token in input_tokens:
        token = re.sub(r'[^\w\s]','',token)
        map_tokens[token] = [len(tokens)]
        if prev_token is not None:
            map_tokens[prev_token].append(len(tokens) - 1)
        tokens.extend(tokenizer.tokenize(token))
        prev_token = token
    max_idx = -1
    max_token = None
    for k,v in map_tokens.items():
        # map_tokens[k] = list(set(v))
        map_tokens[k] = list(range(v[0], v[-1]+1))
        if v[-1] > max_idx:
            max_idx = v[-1]
            max_token = k
    if max_idx < len(tokens):
        v = map_tokens[max_token][-1]
        new_v = list(range(v,len(tokens)))
        # print(map_tokens[max_token])
        map_tokens[max_token].extend(list(range(v,len(tokens))))
        map_tokens[max_token] = list(set(map_tokens[max_token]))
        # print(v, new_v)
        # print(map_tokens[max_token])
    tokens.append('[SEP]')
    # with open('file.txt', 'a') as f:
    #     print(sentence, file=f)
    #     print(tokens, file=f)
    #     print(map_tokens, file=f)
    return tokens, map_tokens

def get_word_embedding_from_sentence(model,tokenizer,sentence,word):
    assert word in sentence, f"word '{word}' not in sentence"
    tokens, map_tokens = tokenize_sentence(tokenizer, sentence)
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    segments_ids = [0] * len(tokens)
    tokens_tensor = torch.tensor([tokens_ids])
    segments_tensor = torch.tensor([segments_ids])
    tok = {
        'input_ids': tokens_tensor,
        'token_type_ids': torch.tensor([ [0] * len(tokens) ]),
        'attention_mask': torch.tensor([ [1] * len(tokens) ]),
    }
    with torch.no_grad():
        # encoded_layers = model(tokens_tensor, segments_tensor)
        encoded_layers = model(**tok)
    hidden_states = encoded_layers.hidden_states
    # token_embeddings = []
    # for token_i in range(len(tokens)):
    #     embeddings = []
    #     for layer_i in range(len(hidden_states)):
    #         vec = hidden_states[layer_i][0][token_i]
    #         embeddings.append(vec)
    #     token_embeddings.append(embeddings)
    assert word in map_tokens, f"word '{word}' not in '{sentence}'"
    token_idx = map_tokens[word]
    token_embeddings = hidden_states[-1].squeeze()
    # print(tokens)
    # print(word)
    # print(token_idx)
    # return token_embeddings[token_idx][-1]
    return token_embeddings[token_idx].mean(axis=0)
