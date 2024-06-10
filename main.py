#Import necessary libraries
from create_embeddings import service_context, storage_context
from llama_index.core import load_index_from_storage, set_global_service_context
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


#Load the index from storage_context
set_global_service_context(service_context)
index = load_index_from_storage(storage_context)

response_synthesizer = get_response_synthesizer(service_context=service_context)
vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=7)
vector_query_engine = RetrieverQueryEngine(
    retriever=vector_retriever,
    response_synthesizer=response_synthesizer,
)

#Ask any GST related question.
user_input = "What is GST?"

#Prompt engineering
prompt = f"""
    <|system|>
        you are a law and finance assistant, try to provide information with including the articles and acts according to the Indian Laws and Indian Finance.
        Do not consider anything on your own.
        focus on user query.
        Generate complete and meaningful answer.
    </s>
    <|user|>
        {user_input}
    </s>
    <|assistant|>
    """

response = vector_query_engine.query(prompt)
print(response)
