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


app = Dash()

# Requires Dash 2.17.0 or later
app.layout = [
    html.H1(children='Title of Dash App', style={'textAlign':'center'}),

    dcc.Input(
        id='sentence1',
        type='text',
        placeholder='Enter sentence',
        value="we're going there by coach"
    ),
    dcc.Input(
        id='sentence2',
        type='text',
        placeholder='Enter sentence',
        value="we're going there by bus"
    ),
    dcc.Input(
        id='sentence3',
        type='text',
        placeholder='Enter sentence',
        value="he was appointed as our coach"
    ),
    dcc.Input(
        id='word1',
        type='text',
        placeholder='Enter word',
        value="coach"
    ),
    dcc.Input(
        id='word2',
        type='text',
        placeholder='Enter word',
        value="bus"
    ),
    dcc.Input(
        id='word3',
        type='text',
        placeholder='Enter word',
        value="coach"
    ),

    dcc.Graph(id='scatter-plot')
]

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('sentence1', 'value'),
    Input('sentence2', 'value'),
    Input('sentence3', 'value'),
    Input('word1', 'value'),
    Input('word2', 'value'),
    Input('word3', 'value'),
)
def update_graph(sentence1,sentence2,sentence3,word1,word2,word3):
    vec0 = get_word_embedding_from_sentence(model, tokenizer, sentence1, word1)
    vec1 = get_word_embedding_from_sentence(model, tokenizer, sentence2, word2)
    vec2 = get_word_embedding_from_sentence(model, tokenizer, sentence3, word3)

    cutoff=100
    x1,y1, x2,y2 = viz.get_vector_points(vec0[:cutoff],vec1[:cutoff])
    _x1,_y1, x3,y3 = viz.get_vector_points(vec0[:cutoff],vec2[:cutoff])

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x1, y=y1, mode='markers', marker=dict(size=5, color='black'), name='List 1'))
    fig.add_trace(go.Scatter(x=x2, y=y2, mode='markers', marker=dict(size=5, color='blue'), name='List 2'))
    fig.add_trace(go.Scatter(x=x3, y=y3, mode='markers', marker=dict(size=5, color='red'), name='List 2'))

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

    return fig



if __name__ == '__main__':
    app.run(debug=True)
