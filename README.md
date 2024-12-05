# YouTube Muay Thai Video Recommendation Web App

Sylvie von Duuglas-Ittu has one of the greatest Muay Thai Youtube Channel's with over 3,000 videos worth of incredible content. Oftentimes the video recommendations produced by searching for videos directly on her channel can leave out valuable content due to the fact that this search is based only on the similarity of the search query to the video title.

In this project I attempt to create a more comprehensive search algorithm which also takes into account the similarity of the search query to the video transcript, by calculating the distance between their respective embeddings as done by the 'all-mpnet-base-v2' LLM available with the SentenceTransformer python library. The final search function is served as an API (built using FastAPI) and deployed to Google Cloud Run so that it could be used by the Gradio app which I created and deployed as a HuggingFace Space to serve as the front-end of the web app.

In order to use the app just type in a Muay Thai related topic you would like to know about (e.g. switch kick, footwork, timing) and the API will return a list of up to 5 video recommendations from Sylvie' YouTube channel that are most related to the search query.

Web App - https://huggingface.co/spaces/axeleichelmann/yt-semantic-search-app
