#Load the necessary libraries
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import ServiceContext, StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.anthropic import Anthropic
import os

#Defining environment variable for OPENAI API KEY
os.environ['OPENAI_API_KEY'] = "your_openai_api_key"

#Defining the environment variable for ANTHROPIC API KEY
os.environ['ANTHROPIC_API_KEY'] = "your_anthropic_api_key"

#Defining Anthropic LLM - claude-instant-1.2 model from LlamaIndex
llm = Anthropic(model="claude-instant-1.2",
                   max_tokens=2000,
                   temperature=1)

#OPENAI embedding model taken from LlamaIndex
embed_model=OpenAIEmbedding(model="text-embedding-3-large")

#Load the pdf's from GST_data folder
documents = SimpleDirectoryReader("GST_data").load_data()

#Embedding folder name
embedding_name = "GST_embeddings"


service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model
)

PERSIST_DIR =f"{embedding_name}"

storage_context = StorageContext.from_defaults()

#build index
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context,storage_context=storage_context
)

#save index
index.storage_context.persist(persist_dir=PERSIST_DIR)

