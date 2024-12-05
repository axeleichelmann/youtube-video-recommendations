import requests
import json
from sentence_transformers import SentenceTransformer, util
from youtube_transcript_api import YouTubeTranscriptApi
import os

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple





def returnSearchResults(query : str, 
                        df_embedding : Tuple[pd.DataFrame,pd.DataFrame],
                        model : SentenceTransformer,
                        dist : util,
                        threshold : float=0.5, top_k : int=5) -> np.ndarray:
    
    # Embed Query
    query_embedding = model.encode(query).reshape(1,-1)

    # Compute distance between query and titles
    dist_arr = dist(df_embedding[0].values, query_embedding) + dist(df_embedding[1].values, query_embedding)

    # Identify videos that are close to query based on threshold
    idx_above_threshold = np.argwhere(dist_arr.flatten()>threshold).flatten()
    idx_sorted = np.argsort((-dist_arr[idx_above_threshold]), axis=0).flatten()

    # return indexes of top k search results
    return idx_above_threshold[idx_sorted][:top_k]



def getVideoRecords(response):
    """
        Function to extract YouTube video data from GET request response
    """

    video_record_list = []
    
    for raw_item in json.loads(response.text)['items']:
    
        # only execute for youtube videos
        if raw_item['id']['kind'] != "youtube#video":
            continue
        
        video_record = {}
        video_record['video_id'] = raw_item['id']['videoId']
        video_record['datetime'] = raw_item['snippet']['publishedAt']
        video_record['title'] = raw_item['snippet']['title']
        
        video_record_list.append(video_record)

    return video_record_list






def getVideoIDs():
    # Define Channel ID of Sylvie's YouTube channel
    channel_id = 'UCgFe05f-DrPpaunE4Gaz3cQ'

    # Define the url for the API to use when you make a request
    url = 'https://www.googleapis.com/youtube/v3/search'
    yt_api_key = os.getenv('YT_API_KEY')

    # Initialize page token
    page_token = None

    # Initialize list to store video data
    video_record_list = []

    # Extract video data across multiple search result pages
    while page_token != 0:
        # define parameters for API call
        params = {'key' : yt_api_key, 'channelId' : channel_id,
                'part' : ["snippet", "id"], 'order' : "date",
                'maxResults' : 50, 'pageToken' : page_token}
        
        # Make get request
        response = requests.get(url, params=params)
        video_record_list += getVideoRecords(response)

        try:
            # Get next page token
            page_token = json.loads(response.text)["nextPageToken"]
        except:
            # If no next page token, kill while loop
            page_token = 0

    # Store data in a Pandas DataFrame
    pd.DataFrame(video_record_list).to_parquet('data/video-ids.parquet')





def extract_text(transcript: list) -> str:
    """
        Function to extract text from transcript dictionary
    """
    
    text_list = [transcript[i]['text'] for i in range(len(transcript))]
    return ' '.join(text_list)





def getVideoTranscripts():
    
    # Import video ID data
    df = pd.read_parquet('data/video-ids.parquet')

    # Initialize a list to store video captions
    transcript_text_list = []

    # Loop through each row of videos dataframe
    for i in range(len(df)):
        # Try to extract captions
        try:
            # get transcript
            transcript = YouTubeTranscriptApi.get_transcript(df['video_id'][i])
            transcript_text = extract_text(transcript)
        except:
            # If no captions available set transcript text to "n/a"
            transcript_text = "n/a"
        # Append transcript text to list
        transcript_text_list.append(transcript_text)

    df['transcript'] = pd.Series(transcript_text_list)

    df.to_parquet('data/video-transcripts.parquet')

    



def cleanData():
    
    # Import video transcript data
    df = pd.read_parquet('data/video-transcripts.parquet')

    # Reformat datetime
    df['datetime'] = df.datetime.apply(lambda x : datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))

    # Replace each strings that appears in title and transcript columns
    string_replacements = {'&quot;' : '"', '&#39;' : "'", "[Music]" : ""}

    for string in string_replacements.keys():
        df['title'] = df.title.apply(lambda x : x.replace(string, string_replacements[string]))
        df['transcript'] = df.transcript.apply(lambda x : x.replace(string, string_replacements[string]))

    df.to_parquet('data/video-transcripts.parquet')




def createTextEmbeddings():

    # Import video transcript data
    df = pd.read_parquet('data/video-transcripts.parquet')

    # Define model to use for embeddings & column(s) to embed
    model_name = "all-mpnet-base-v2"

    # Create model
    model  = SentenceTransformer(model_name)

    for column in ['title','transcript']:
        # generate embeddings
        embedding_arr = model.encode(df[column].to_list())

        # store embeddings in a dataframe
        df_embedding = pd.DataFrame(embedding_arr)
        df_embedding.columns = [column+'_embedding-'+str(i) for i in range(embedding_arr.shape[1])]

        df = pd.concat([df,df_embedding], axis=1)

    df.to_parquet('data/video-index.parquet')
