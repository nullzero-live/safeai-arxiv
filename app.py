# import libraries
import os 
import json
import requests

#Extra
import openai
from dotenv import load_dotenv
import tiktoken
import tqdm
import conv_ag


#Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()   

openai.api_key = os.getenv("OPENAI_API_KEY")

'''#Chunking the Text
tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

# Langchain APIs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,  # number of tokens overlap between chunks
    length_function=tiktoken_len,
    separators=['\n\n', '\n', ' ', '']
)

chunks = text_splitter.split_text(docs[5].page_content)
len(chunks)'''

from datasets import load_dataset

data = load_dataset('squad', split='train')
data

'''data = data.to_pandas()
data.head()

data.drop_duplicates(subset='context', keep='first', inplace=True)
data.head()'''

from getpass import getpass
from langchain.embeddings.openai import OpenAIEmbeddings

OPENAI_API_KEY = getpass("OpenAI API Key: ")  # platform.openai.com
model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)



import pinecone

# find API key in console at app.pinecone.io
YOUR_API_KEY = getpass("Pinecone API Key: ")
# find ENV (cloud region) next to API key in console
YOUR_ENV = input("Pinecone environment: ")

index_name = 'langchain-retrieval-agent'
pinecone.init(
    api_key=YOUR_API_KEY,
    environment=YOUR_ENV
)

if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='dotproduct',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )


index = pinecone.Index(index_name)
index.describe_index_stats()


# INDEXING

from tqdm.auto import tqdm
from uuid import uuid4

batch_size = 100

texts = []
metadatas = []

for i in tqdm(range(0, len(data), batch_size)):
    # get end of batch
    i_end = min(len(data), i+batch_size)
    batch = data.iloc[i:i_end]
    # first get metadata fields for this record
    metadatas = [{
        'title': record['title'],
        'text': record['context']
    } for j, record in batch.iterrows()]
    # get the list of contexts / documents
    documents = batch['context']
    # create document embeddings
    embeds = embed.embed_documents(documents)
    # get IDs
    ids = batch['id']
    # add everything to pinecone
    index.upsert(vectors=zip(ids, embeds, metadatas))
    

index.describe_index_stats()

# Creating a Vector Store and Querying

from langchain.vectorstores import Pinecone

text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)


query = "when was the college of engineering in the University of Notre Dame established?"

vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
)



## DELETING AN INDEX
#pinecone.delete_index(index_name)