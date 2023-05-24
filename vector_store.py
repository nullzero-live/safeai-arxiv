# Vector Store

# Storage - Update to include query but primarily use logic to store the outputs as .txt files initially.

import os
from dotenv import load_dotenv
import pinecone
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
from dotenv import load_dotenv
from tqdm.auto import tqdm  # this is our progress bar

#Extra modules
from tqdm.auto import tqdm
from uuid import uuid4
import tiktoken

#LangChain Modules
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader


load_dotenv()
openai_api_key=os.environ.get("OPENAI_API_KEY")

#Initialize pinecone client
pinecone.init(
        api_key=os.environ.get("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.environ.get("PINECONE_ENV")  # next to api key in console
    )




tokenizer = tiktoken.get_encoding('p50k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

# create the embedding model
embed = OpenAIEmbeddings(openai_api_key=openai_api_key, chunk_size=1)

def embed_upsert(data, filename, business_name, industry, index_name="langchain-demo"): #add data later
       
    # Open the document in read mode
    with open(filename, 'r') as file:
    # Read the contents of the document into a string
        documents = file.read()
    
    #loader = TextLoader(os.path.join("./chats", "chat_logs.txt"))
    #documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""])
    
    doc_string = text_splitter.split_text(documents)
   
    #Initialize index (Pinecone client out of function globally)
    index = pinecone.Index(index_name)
    
    batch_limit=5
    texts = []
    metadatas = []

    
    for i in enumerate(doc_string):
        # first get metadata fields for this record
        metadata = {
            'id': str(i),
            'business_name': business_name,
            'industry': industry
        }
        #if len(texts) >= batch_limit: Include a batch limit for large file sizes
        # now we create chunks from the record text
        record_texts = text_splitter.split_text(doc_string)
        # create individual metadata dicts for each chunk
        record_metadatas = [{
            "chunk": j, "text": text, **metadata
        } for j, text in enumerate(record_texts)]
        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        
        # if we have reached the batch_limit we can add texts
    
        idx = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)    
        index.upsert(vectors=zip(idx, embeds, metadatas), namespace="business") #add back idx
        texts = []
        metadatas = []
        

    

    
    
def query_vec(index_name="mlqai"):
    #Open file and read in data
    #with open(data, 'r') as f:
    
    loader = TextLoader(os.path.join("./chats", "chat_logs.txt"))
    print(f"Loader is:\n\n {loader}")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    

    embeddings = OpenAIEmbeddings()
    
    index_name = "langchain-demo"

    docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    
    query = "What is the marketing strategy used for this company?"
    docs = docsearch.similarity_search(query)
    
    return print()


def embed_folder():    
    # Specify the folder path
    folder_path = '/Users/nullzero/Documents/lang-chain/bus-agent/chats'

    # List all files in the folder and sort them
    file_list = sorted(os.listdir(folder_path))

    # Iterate through the files by index
    for index in range(len(tqdm(file_list))):
        file_name = file_list[index]
        file_path = os.path.join(folder_path, file_name)
        
        # Perform your desired operation on the file
        with open(file_path, 'r') as file:
            documents = file.read()
        
        print(f"Processing file {index}: {file_path}")
        embed_upsert(documents, file_path, file_path.split()[-1], file_path.split()[-2])



    
    



