from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import pandas as pd

from utils.utils_transformers import *
import utils.visualization as viz



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

vec0 = get_word_embedding_from_sentence(model, tokenizer,  "we're going there by coach", "coach")
vec1 = get_word_embedding_from_sentence(model, tokenizer,  "we're going there by bus", "bus")
vec2 = get_word_embedding_from_sentence(model, tokenizer,  "he was appointed as our coach", "coach")


cutoff=100
x1,y1, x2,y2 = viz.get_vector_points(vec0[:cutoff],vec1[:cutoff])
_x1,_y1, x3,y3 = viz.get_vector_points(vec0[:cutoff],vec2[:cutoff])

fig = go.Figure()

# Add scatter plot for both lists
fig.add_trace(go.Scatter(x=x1, y=y1, mode='markers', marker=dict(size=5, color='black'), name='List 1'))
fig.add_trace(go.Scatter(x=x2, y=y2, mode='markers', marker=dict(size=5, color='blue'), name='List 2'))
fig.add_trace(go.Scatter(x=x3, y=y3, mode='markers', marker=dict(size=5, color='red'), name='List 2'))

# Add lines connecting each point from List 1 to each point from List 2
line_x = []
line_y = []
for px, py, tx, ty in zip(x1, y1, x2, y2):
    line_x.extend([px, tx, None])  # None breaks the line
    line_y.extend([py, ty, None])
fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', line=dict(color='blue', width=1), name='Connections'))

line_x = []
line_y = []
for px, py, tx, ty in zip(x1, y1, x3, y3):
    line_x.extend([px, tx, None])  # None breaks the line
    line_y.extend([py, ty, None])
fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', line=dict(color='red', width=1), name='Connections'))



app = Dash()

# Requires Dash 2.17.0 or later
app.layout = [
    html.H1(children='Title of Dash App', style={'textAlign':'center'}),
    dcc.Graph(figure=fig)
]

if __name__ == '__main__':
    app.run(debug=True)

