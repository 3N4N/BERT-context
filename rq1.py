import plotly.express as px
import plotly.graph_objects as go

import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import json

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP

import utils.utils_transformers as util
import utils.visualization as viz

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

def _helper():
    df_pairs = pd.read_csv("datasets/dataset_pairs_en.tsv", sep='\t')
    df_pairs.columns = ['Sent1', 'Sent2', 'Synonym']

    synonyms = {}
    for idx, row in df_pairs.iterrows():
        word1 = re.search(r"<b>(.*)</b>", row['Sent1']).group(1)
        word2 = re.search(r"<b>(.*)</b>", row['Sent2']).group(1)
        if word1 != word2 and row['Synonym'] == 'T':
            synonyms.setdefault(word1, []).append(word2)
            synonyms.setdefault(word2, []).append(word1)
    synonyms = { k: list(set(v)) for k,v in synonyms.items() }

    df = pd.read_csv("datasets/dataset_wic_en.tsv", sep='\t')
    df.columns = [ 'Target', 'Sent1', 'Sent2', 'Synonym' ]
    # df = df[:50]

    targets = []
    sentences = []
    for row, item in df.iterrows():
        targets.append(item['Target'])
        targets.append(item['Target'])
        sentences.append(item['Sent1'])
        sentences.append(item['Sent2'])

    words = [
        re.search(r"<b>(.*)</b>", sentence).group(1) for sentence in sentences
    ]
    _sentences = sentences.copy()
    sentences = [ sentence.replace("<b>","").replace("</b>","") for sentence in sentences ]

    targets_uniq = sorted(set(targets), key=len)
    targets_base = []
    targets_map = {}
    for target in targets_uniq:
        flag = False
        for target_base in targets_base:
            if target_base in target:
                targets_map[target] = target_base
                flag = True
                break
        if not flag:
            targets_base.append(target)
            targets_map[target] = target

    targets_new = []
    for target in targets:
        targets_new.append(targets_map[target])

    df_uniq = pd.DataFrame({
        'Target': targets_new,
        'Sentence': sentences,
        '_Sentence': _sentences,
        'Word': words,
    })
    df_uniq.drop_duplicates(subset='Sentence', inplace=True)
    df_uniq = df_uniq[df_uniq['Word'].apply(lambda x: len(x.split()) == 1)]


    word_embeddings = [
        util.get_word_embedding_from_sentence(model, tokenizer, sentence, word)
        for sentence,word in zip(df_uniq['Sentence'],df_uniq['Word'])
    ]

    return df_uniq, synonyms, word_embeddings

def _scatter():
    df = pd.read_csv("datasets/dataset_pairs_en.tsv", sep='\t')
    df.columns = [ 'Sent1', 'Sent2', 'Synonym' ]

    df['Word1'] = df['Sent1'].apply(lambda x: re.search(r"<b>(.*)</b>",x).group(1))
    df['Word2'] = df['Sent2'].apply(lambda x: re.search(r"<b>(.*)</b>",x).group(1))
    df['Sent1'] = df['Sent1'].apply(lambda x: x.replace("<b>","").replace("</b>",""))
    df['Sent2'] = df['Sent2'].apply(lambda x: x.replace("<b>","").replace("</b>",""))
    df = df[df['Word1'].apply(lambda x: len(x.split()) == 1)]
    df = df[df['Word2'].apply(lambda x: len(x.split()) == 1)]

    cosims = []
    l2sims = []
    dotsims = []
    for idx, row in df.iterrows():
        embed1 = util.get_word_embedding_from_sentence(model, tokenizer, row['Sent1'], row['Word1'])
        embed2 = util.get_word_embedding_from_sentence(model, tokenizer, row['Sent2'], row['Word2'])
        cosim = F.cosine_similarity(embed1, embed2, dim=0).item()
        cosims.append(cosim)
        # l2sim = norm(embed1 - embed2) # L2 norm
        # l2sims.append(l2sim)
        # dotsim = np.dot(embed1, embed2) # dot prod sim
        # dotsims.append(dotsim)

    df['CoSim'] = cosims
    # df['L2Sim'] = l2sims
    # df['DPSim'] = dotsims

    df['Color'] = df['Synonym'].apply(lambda x: 'red' if x == 'F' else 'blue' )
    df['Size'] = df.apply(lambda row: abs(len(row['Word1']) - len(row['Word2'])), axis=1)
    df['Size'] = df.apply(lambda row: len(row['Word1']) + len(row['Word2']), axis=1)

    df['Error'] = df.apply(lambda row: 1-row['CoSim'] if row['Synonym']=='T' else row['CoSim'], axis=1)
    df['Same'] = df.apply(lambda row: len(set(row['Sent1'].split()).intersection(set(row['Sent2'].split()))), axis=1)

    return df



df, synonym_map, word_embeddings = _helper()
embeddings = torch.stack(word_embeddings)
pca_embeddings = PCA(n_components=2, random_state=42).fit_transform(embeddings)
tsne_embeddings = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
df['PCA_X'] = pca_embeddings[:,0]
df['PCA_Y'] = pca_embeddings[:,1]
df['tSNE_X'] = tsne_embeddings[:,0]
df['tSNE_Y'] = tsne_embeddings[:,1]
pca_graph_data = json.loads(df[['Target','_Sentence','Word','PCA_X','PCA_Y']].to_json(orient='records'))
print(json.dumps(pca_graph_data, indent=2))
tsne_graph_data = json.loads(df[['Target','_Sentence','Word','tSNE_X','tSNE_Y']].to_json(orient='records'))
print(json.dumps(tsne_graph_data, indent=2))


df_ = _scatter()
cosim_data = {
    'synonym': {
        'min': df_[df_['Synonym']=='T']['CoSim'].min().item(),
        'max': df_[df_['Synonym']=='T']['CoSim'].max().item(),
        'avg': df_[df_['Synonym']=='T']['CoSim'].mean().item(),
        'std': df_[df_['Synonym']=='T']['CoSim'].std().item(),
    },
    'homonym': {
        'min': df_[df_['Synonym']=='F']['CoSim'].min().item(),
        'max': df_[df_['Synonym']=='F']['CoSim'].max().item(),
        'avg': df_[df_['Synonym']=='F']['CoSim'].mean().item(),
        'std': df_[df_['Synonym']=='F']['CoSim'].std().item(),
    }
}

_df = pd.DataFrame({
    'X':        df_.index,
    'Y':        df_['CoSim'],
    'Size':     df_['Size'],
    'Color':    df_['Color'],
})
scatter_graph_data = json.loads( _df.to_json(orient='records') )
print(json.dumps(scatter_graph_data, indent=2))

_df = pd.DataFrame({
    'X':        df_['Same'],
    'Y':        df_['Error'],
    'Size':     5,
    'Color':    df_['Color'],
})
common_word_effect_data = json.loads( _df.to_json(orient='records') )
print(json.dumps(common_word_effect_data, indent=2))


json_data = {
    'similarity scores': cosim_data,
    'PCA_data': {
        'title': 'PCA of BERT token embeddings',
        'description': 'This graph highlights selected word and its synonyms on the scatter plot',
        'points': pca_graph_data,
    },
    'tSNE_data': {
        'title': 'tsNE of BERT token embeddings',
        'description': 'This graph highlights selected word and its synonyms on the scatter plot',
        'points': tsne_graph_data,
    },
    'scatter_data': {
        'title': 'Scatter plot of cosine similarities',
        'description': 'Scatter plot of cosine similarities where X-axis is dataframe indices and Y-axis is similarity',
        'conclusion': 'Cosine similarity is not a reliable metric for determining homonyms and synonyms from BERT token embeddings',
        'points': scatter_graph_data,
    },
    'Realiability of cosine similarity with #common words': {
        'title': 'Scatter plot of error with respect to number of common words',
        'description': 'X-axis is the number of common words in two sentences and Y-axis is the error as measured by the difference from [0,1] values.',
        'conclusion': 'Cosine similarity fails to detect homonyms when two contexts are too similar',
        'points': common_word_effect_data,
    }
}

with open('rq1.json', 'w') as f:
    json.dump(json_data, f)
