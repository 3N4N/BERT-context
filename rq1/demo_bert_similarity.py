import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

from utils.utils_transformers import *



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

vec0 = get_word_embedding_from_sentence(model, tokenizer,  "we're going there by coach", "coach")
vec1 = get_word_embedding_from_sentence(model, tokenizer,  "we're going there by bus", "bus")
vec2 = get_word_embedding_from_sentence(model, tokenizer,  "he was appointed as our coach", "coach")

cosim1 = F.cosine_similarity(vec0, vec1, dim=0)
cosim2 = F.cosine_similarity(vec0, vec2, dim=0)

print(f'{cosim1:0.4}, {cosim2:0.4}')
