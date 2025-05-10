#!/usr/bin/env python
# coding: utf-8

# In[1]:


from helper import get_openai_api_key
import nest_asyncio
nest_asyncio.apply()

OPENAI_API_KEY = get_openai_api_key()  # Use helper method (do not hardcode key)


# In[4]:


import os
import requests

# Create 'data' folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Dictionary of filenames and their URLs
urls = {
    "metagpt.pdf": "https://openreview.net/pdf?id=VtmBAGCN7o",
    "longlora.pdf": "https://openreview.net/pdf?id=6PmJoRfdaK",
    "selfrag.pdf": "https://openreview.net/pdf?id=hSyW5go0v8"
}

# Function to download and save files
for filename, url in urls.items():
    print(f"Downloading {filename}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(f"data/{filename}", "wb") as f:
            f.write(response.content)
        print(f"Saved {filename} to data/")
    else:
        print(f"Failed to download {filename}. HTTP Status: {response.status_code}")



# In[5]:


from llama_index.core import SimpleDirectoryReader

# Load all documents from the 'data' folder
documents = SimpleDirectoryReader(input_dir="data").load_data()
print(f"{len(documents)} documents loaded.")


# In[6]:


from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)


# In[7]:


from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")


# In[8]:


from llama_index.core import VectorStoreIndex, SummaryIndex

vector_index = VectorStoreIndex(nodes)
summary_index = SummaryIndex(nodes)


# In[9]:


vector_engine = vector_index.as_query_engine(similarity_top_k=3)
summary_engine = summary_index.as_query_engine(response_mode="tree_summarize", use_async=True)


# In[10]:


from llama_index.core.tools import QueryEngineTool

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_engine,
    description="Retrieve specific info from multiple research papers."
)

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_engine,
    description="Summarize content across multiple documents."
)


# In[11]:


from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[vector_tool, summary_tool],
    verbose=True
)


# In[12]:


response1 = router_engine.query("What are the main contributions of each paper?")
print(str(response1))

response2 = router_engine.query("Compare the results of the methods discussed.")
print(str(response2))

