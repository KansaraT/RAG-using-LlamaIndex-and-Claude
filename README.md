**OVERVIEW**

There are many challenges when working with LLMs such as domain knowledge gaps and factuality issues, Retrieval Augmented Generation (RAG) provides a solution to mitigate some of these issues by augmenting LLMs with external knowledge such as databases.

We are using the power of Claude LLM from Anthropic to generate response of our queries and OPENAI to create embeddings of the dataset using llamaIndex. 

Dataset : consists of [GST acts](https://taxinformation.cbic.gov.in/)

STEPS TO BE FOLLOWED: 

1. Run requirements.txt file ```pip install -r requirements.txt```
  
2. Run create_embeddings.py file to create embeddings of your dataset. (like I am using GST_data)
  ```python create_embeddings.py```

3. Run main.py file to load the embeddings and genarate response to queries.
   ```python main.py```
