from functions import getVideoIDs, getVideoTranscripts, cleanData, createTextEmbeddings
import time
import datetime

print("Starting Data Pipeline at ", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("----------------------------------------------------------------------")


# Step 1 : extract video ID's
t_start = time.time()
getVideoIDs()
t_end = time.time()

print("Step 1 : Done")
print("-----> Video IDs downloaded in", str(t_end - t_start), "seconds", "\n")


# Step 2 : extract video transcripts
t_start = time.time()
getVideoTranscripts()
t_end = time.time()

print("Step 2 : Done")
print("-----> Video Transcripts downloaded in", str(t_end - t_start), "seconds", "\n")



# Step 3 : Clean & Transform data
t_start = time.time()
cleanData()
t_end = time.time()

print("Step 3 : Done")
print("-----> Data cleaned & transformed in", str(t_end - t_start), "seconds", "\n")





# Step 4 : Generate text embeddings
t_start = time.time()
createTextEmbeddings()
t_end = time.time()

print("Step 4 : Done")
print("-----> Text Embeddings Generated in", str(t_end - t_start), "seconds", "\n")
