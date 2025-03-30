import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

from utils.utils_transformers import *
import utils.visualization as viz

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

vec0 = get_word_embedding_from_sentence(model, tokenizer,  "we're going there by coach", "coach")
vec1 = get_word_embedding_from_sentence(model, tokenizer,  "we're going there by bus", "bus")
vec2 = get_word_embedding_from_sentence(model, tokenizer,  "he was appointed as our coach", "coach")


plt.figure(figsize=(6, 6))
ax = plt.gca()

cutoff = 100
viz.plot_vectors(ax, vec0[:cutoff], vec1[:cutoff], ['b','g','red'])
viz.plot_vectors(ax, vec0[:cutoff], vec2[:cutoff], ['b','g','blue'])

plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.savefig( 'plot.pdf')
