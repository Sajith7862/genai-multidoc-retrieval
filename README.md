## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
To design and implement a multi-document retrieval agent using LlamaIndex, capable of extracting and synthesizing information from multiple research papers, and to evaluate its performance by testing diverse queries for accuracy, relevance, and clarity.

### DESIGN STEPS:

#### STEP 1:Setup Environment and Install Dependencies
Initialize the Jupyter environment and install required Python packages like llama-index, openai, and requests. Configure API keys and imports.

#### STEP 2:Download and Load Multiple Research Papers
Programmatically download three PDF research papers into a local data/ folder and load them using SimpleDirectoryReader from LlamaIndex.

#### STEP 3:Preprocess and Chunk Text into Nodes
Use SentenceSplitter to convert the loaded documents into manageable text chunks (nodes) for better retrieval and embedding.

#### STEP 4: Create Vector Index and Define Tools
Build a VectorStoreIndex from the document nodes and define tools like vector_query_tool or summary_tool to enable semantic querying and summarization.

#### STEP 5:Perform Retrieval and Summarization
Use an LLM (like gpt-3.5-turbo) with the defined tools to run queries and retrieve summarized or page-filtered content from the papers.

### PROGRAM:
```
from helper import get_openai_api_key
import nest_asyncio
nest_asyncio.apply()

OPENAI_API_KEY = get_openai_api_key()  # Use helper method (do not hardcode key)

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


from llama_index.core import SimpleDirectoryReader

# Load all documents from the 'data' folder
documents = SimpleDirectoryReader(input_dir="data").load_data()
print(f"{len(documents)} documents loaded.")


from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

from llama_index.core import VectorStoreIndex, SummaryIndex

vector_index = VectorStoreIndex(nodes)
summary_index = SummaryIndex(nodes)

vector_engine = vector_index.as_query_engine(similarity_top_k=3)
summary_engine = summary_index.as_query_engine(response_mode="tree_summarize", use_async=True)

from llama_index.core.tools import QueryEngineTool

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_engine,
    description="Retrieve specific info from multiple research papers."
)

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_engine,
    description="Summarize content across multiple documents."
)

from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[vector_tool, summary_tool],
    verbose=True
)

response1 = router_engine.query("What are the main contributions of each paper?")
print(str(response1))

response2 = router_engine.query("Compare the results of the methods discussed.")
print(str(response2))


```

### OUTPUT:

![image](https://github.com/user-attachments/assets/98718f1b-2d03-43b5-982e-59c97aec1384)


### RESULT:

The system successfully answered semantic queries such as:
“What are the high-level results of MetaGPT?” and
“What are the MetaGPT comparisons with ChatDev described on page 8?”

It returned concise, context-aware responses by retrieving the most relevant chunks from the research papers.
