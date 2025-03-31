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
app.layout = html.Div([
    html.H1(children='Visualization of BERT token embeddings', style={'textAlign':'center'}),

    html.Div(children=[
        html.Div(
            dcc.RadioItems(
                id='cutoff',
                options=[ 0, 1,2,3,4, ],
                value=0,
                labelStyle={'display': 'inline-block'}
            ), style = {'textAlign': 'center'},
        ),
        html.Div(
            dcc.Graph(id='scatter-plot', responsive=False),
            id='graph',
        ),
        html.Div(children=[
            html.Div(children=[
                html.Label("Sentence", id='sentence-label'),
                html.Label("Word", id='word-label'),
            ], className='label sentence-word',),
            html.Div(children=[
                dcc.Input(
                    id='sentence1',
                    className='sentence',
                    type='text',
                    placeholder='Enter sentence',
                    value="we're going there by coach",
                    debounce=True,
                ),
                dcc.Input(
                    id='word1',
                    className='word',
                    type='text',
                    placeholder='Enter word',
                    value="coach",
                    debounce=True,
                ),
            ], className='sentence-word',),
            html.Div([
                dcc.Input(
                    id='sentence2',
                    className='sentence',
                    type='text',
                    placeholder='Enter sentence',
                    value="we're going there by bus",
                    debounce=True,
                ),
                dcc.Input(
                    id='word2',
                    className='word',
                    type='text',
                    placeholder='Enter word',
                    value="bus",
                    debounce=True,
                ),
            ], className='sentence-word',),
            html.Div([
                dcc.Input(
                    id='sentence3',
                    className='sentence',
                    type='text',
                    placeholder='Enter sentence',
                    value="he was appointed as our coach",
                    debounce=True,
                ),
                dcc.Input(
                    id='word3',
                    className='word',
                    type='text',
                    placeholder='Enter word',
                    value="coach",
                    debounce=True,
                ),
            ], className='sentence-word',),
        ], id='textinput',),
    ], id='main-part')
])

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('sentence1', 'value'),
    Input('sentence2', 'value'),
    Input('sentence3', 'value'),
    Input('word1', 'value'),
    Input('word2', 'value'),
    Input('word3', 'value'),
    Input('cutoff', 'value'),
)
def update_graph(sentence1,sentence2,sentence3,word1,word2,word3,quarter):
    fig = go.Figure()
    # fig.update_layout(width=600, height=400)

    # print(sentence1)
    # print(sentence2)
    # print(sentence3)

    has_sent1 = len(sentence1) > 0
    has_sent2 = len(sentence2) > 0
    has_sent3 = len(sentence3) > 0

    if has_sent1 + has_sent2 + has_sent3 < 2:
        return Fig

    try:
        if has_sent1:
            vec0 = get_word_embedding_from_sentence(model, tokenizer, sentence1, word1)
        if has_sent2:
            vec1 = get_word_embedding_from_sentence(model, tokenizer, sentence2, word2)
        if has_sent3:
            vec2 = get_word_embedding_from_sentence(model, tokenizer, sentence3, word3)
    except AssertionError:
        return fig

    # assert len(vec0) == len(vec1) == len(vec2)

    if quarter == 0:
        left = 0
        right = len(vec0)
    else:
        left = (quarter-1)*200
        right = (quarter)*200

    if has_sent3:
        _x1,_y1, x3,y3 = viz.get_vector_points(vec0[left:right],vec2[left:right])
        fig.add_trace(go.Scatter(x=x3, y=y3, mode='markers', marker=dict(size=5, color='red'),      name=word3))

    x1,y1, x2,y2 = viz.get_vector_points(vec0[left:right],vec1[left:right])
    fig.add_trace(go.Scatter(x=x1, y=y1, mode='markers', marker=dict(size=5, color='black'),    name=word1))
    fig.add_trace(go.Scatter(x=x2, y=y2, mode='markers', marker=dict(size=5, color='blue'),     name=word2))


    line_x = []
    line_y = []
    for px, py, tx, ty in zip(x1, y1, x2, y2):
        line_x.extend([px, tx, None])  # None breaks the line
        line_y.extend([py, ty, None])
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', line=dict(color='blue', width=1), name=f'{word1} - {word2}'))

    if has_sent3:
        line_x = []
        line_y = []
        for px, py, tx, ty in zip(x1, y1, x3, y3):
            line_x.extend([px, tx, None])  # None breaks the line
            line_y.extend([py, ty, None])
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', line=dict(color='red', width=1), name=f'{word1} - {word3}'))

    return fig



if __name__ == '__main__':
    app.run(debug=True)
