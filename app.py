# import libraries
import os 
import json
import requests

#Extra
import openai
from dotenv import load_dotenv

load_dotenv()   

openai.api_key = os.getenv("OPENAI_API_KEY")

# Langchain APIs



