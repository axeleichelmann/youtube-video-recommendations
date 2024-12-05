from fastapi import FastAPI
import pandas as pd
import os

from sentence_transformers import SentenceTransformer, util
import numpy as np
from .functions import returnSearchResults

# define model info
model_name = "all-mpnet-base-v2"
model_path = 'data/' + model_name

# Load model
if os.path.exists(model_path):
    model = SentenceTransformer(model_path)
else:
    model = SentenceTransformer(model_name)

# Load video index
df = pd.read_parquet('data/video-index.parquet')

# Create similarity metric object
metric = util.cos_sim

# Create fastAPI object
app = FastAPI()

@app.get('/')
def health_check():
    return {'health_check' : 'OK'}

@app.get('/info')
def info():
    return {'name' : 'YT Search', 'Description' : "Search API for Sylvie von Duuglas-Ittu's YouTube Videos"}

@app.get('/search')
def search(query : str):

    idx_result = returnSearchResults(query, [df[[f'title_embedding-{i}' for i in range(768)]], df[[f'transcript_embedding-{i}' for i in range(768)]]], model, metric, threshold=0.3)
    response = df[['video_id','title']].iloc[idx_result].to_dict(orient='list')

    return response