from dotenv import load_dotenv
import json
import os
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from weaviate_module.weaviate_wrapper import WeaviateWrapper
from weaviate_module.Cross_Encoder_Reranking import Reranker
from genai import Credentials, Client
from genai.text.generation import (
    DecodingMethod,
    TextGenerationParameters,
    TextGenerationReturnOptions,
)
import sys
import numpy as np
import re
# must include this otherwise with some embedding models, weaviate do numpy array truncation
np.set_printoptions(threshold=sys.maxsize)

class ChatBackend:
    '''
        Creating a Generalised Chat Wrapper for Watson X LLMs
        model_id: e.g. fmeta-llama/llama-2-70b-chat
        generation Config, etc.
    '''
    def __init__(self,model_id = "meta-llama/llama-2-70b-chat",generation_params = {
                               "decoding_method": "sample",
                                "max_new_tokens": 512,
                                "min_new_tokens":1,
                                "stop_sequences": ["\n\n\n"],
                                "temperature": 0.1,
                                "top_k": 50,
                                "top_p": 1,
                                "repetition_penalty": 1
    },
                 weaviate_host_url = "http://localhost:8080",
    embedding_model_name = "intfloat/e5-large-v2",
    genai_model = 'mistralai/mistral-7b-instruct-v0-2'
                 ):
        load_dotenv()
        api_key = os.getenv("API_KEY",None)
        ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
        project_id = os.getenv("PROJECT_ID", None)
        creds = None
        if api_key is None or ibm_cloud_url is None or project_id is None:
            print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
        else:
            creds = {
                "url": ibm_cloud_url,
                "apikey": api_key 
            }
        self.model = Model(model_id, params=generation_params, credentials=creds,project_id=project_id)
        self.model_id=model_id
        self.ww = WeaviateWrapper(host = weaviate_host_url,embedding_model_name = embedding_model_name)
        self.ce=Reranker()

        #GenAI section
        self.genai_client = Client(credentials=Credentials.from_env())
        self.genai_model = genai_model
        self.genai_params = TextGenerationParameters(
            decoding_method=DecodingMethod.GREEDY,
            max_new_tokens=generation_params['max_new_tokens'],
            min_new_tokens=generation_params['min_new_tokens'],
            return_options= TextGenerationReturnOptions(generated_tokens=True),
            # temperature=generation_params['temperature'],
            # repetition_penalty=generation_params['repetition_penalty'],
            stop_sequences=['<|endoftext|>', '\n\n\n'],
            include_stop_sequence=False,
            random_seed=3293482354,
            )

    def hypothetical_prompt(self, query):
#         prompt = f'''You have to answer a given query. Please generate an answer under 100 tokens.
# Query: {query}
# Answer:'''
        prompt = f'''<|system|>
You are Granite Chat, an AI language model developed by IBM. 
You are a cautious assistant. 
You carefully follow instructions. 
You are helpful and harmless and you follow ethical guidelines and promote positive behavior. 
Please do not say anything else apart from the answer and do not start a conversation.
<|user|>
You have to answer a given query. Please generate an answer under 100 tokens.
Query: {query}
<|assistant|>\n'''
        return prompt
    
    def qna_prompt(self, query:str, context):
    #     prompt = f'''You are a helpful, respectful and honest assistant. You will be provided with a context and a query. Use the context to answer the query. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something incorrectly.
    # If you don't know the answer to a question, please don't share false information. Do not provide any extra information
    # Context: {context}
    # Query: {query}
    # Answer:
    # '''
        prompt = f'''<|system|>
 'You are Granite Chat, an AI language model developed by IBM. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.
<|user|>
You are a AI language model designed to function as a specialized Retrieval Augmented Generation (RAG) assistant. 
When generating responses, prioritize correctness, i.e., ensure that your response is correct given the context and user query, and that it is grounded in the context. 
Furthermore, make sure that the response is supported by the given document or context.
Always make sure that your response is relevant to the question and then give the final answer.
[Document]
{context}
[End]
Query: {query}
<|assistant|>\n'''
        return prompt


    def build_prompt(self,question,context):
        prompt = f'''
[INST]
Context: {context}
- Take the context above and use that to answer questions in a detailed and professional way..
- if you dont know the answer just say ” i dont know “.
- refrain from using any other knowledge other than the text provided.
- don't mention that you are answering from the text, impersonate as if this is coming from your knowledge
- For the questions whose answer is not available in the provided context, just say please ask a relevant question or frame the question more clearly
Question: {question}?
[/INST]
'''
        return prompt

    def parse_json_response_for_LLM(self,data,classname):
        chunks= data["data"]["Get"][classname] 
        context = []
        if "Parent_child" in classname :
            for chunk in chunks:
                context.append(chunk["text"])
        else:
            for chunk in chunks:
                text = chunk["text"]
                pdf = chunk["pdf"]
                distance = chunk["_additional"]["distance"]
                certainty = chunk["_additional"]["certainty"]
                context.append({
                    "pdf":pdf,
                    "text":text,
                    "distance":distance,
                    "certainty":certainty,
                })
        return context
    
    def return_context_for_query(self,query,pdf="",classname="Rag",top_k=5,autocut=2):
        if pdf =="":
            data= self.ww.search_relevant_objects(user_query = query,classname=classname,parameters=["pdf","text","chunk_number"],top_k=top_k,autocut=autocut)
            #new_data = self.ww.get_unique_data(json.loads(data),"Rag")
            print(data)
            return self.parse_json_response_for_LLM(json.loads(data),classname)
        else:
            data= self.ww.hybrid_search_relevant_objects_where(user_query = query,classname=classname,parameters=["pdf","text","chunk_number"],pdf=pdf,top_k=top_k,autocut=autocut)
            #new_data = self.ww.get_unique_data(json.loads(data),"Rag")
            return self.parse_json_response_for_LLM(json.loads(data),classname)
    
    def return_context(self,query,classname="Parent_child",top_k=5,autocut=2):
        
        data= self.ww.search_relevant_objects(user_query = query,classname=classname,parameters=["text"],top_k=top_k,autocut=autocut)
            
        return self.parse_json_response_for_LLM(json.loads(data),classname)
        
    
        
    def send_to_watsonxai(self,prompt):
        for _, response in enumerate(self.genai_client.text.generation.create(model_id=self.genai_model, inputs = prompt, parameters=self.genai_params)):
            return response.results[0].generated_text
    
  
    def watsonxml_generate(self, prompt):
        """
        Generates text using the Watson AI model based on the given prompt.
        Parameters:
            prompt (str): The prompt to be processed by the Watson AI model.
        Returns:
            str: The generated text from the Watson AI model.
        """
        for _, response in enumerate(self.genai_client.text.generation.create(model_id=self.genai_model, inputs = prompt, parameters=self.genai_params)):
            return response.results[0].generated_text
        
    def answer_based_on_context(self,query,context):
        prompt = self.build_prompt(query,json.dumps(context,indent=4))
        print(prompt)
        result = self.watsonxml_generate(prompt = prompt)
        return result
    
    def answer_based_on_context_hyde(self,query:str,context:str):
        '''
        Generates LLM response for a query based upon a context.
        Parameters:
        query: str
        context: str
        '''
        prompt = self.qna_prompt(query,context)
        # print(prompt)
        result = self.watsonxml_generate(prompt = prompt)
        return result
    
    def generate_hypothetical_answer(self, query):
        prompt = self.hypothetical_prompt(query)
        result = self.watsonxml_generate(prompt)
        return result
    
    def get_unique_data(self,data,classname):
        # data's text value we will do chunk simlarity
        chunks= data["data"]["Get"][classname] # gettin chunks
        dictionaries = [] # a list of chunks with unique id and text embedding
        id = 0
        for chunk in chunks:
            text = chunk["text"]
            pdf = chunk["pdf"]
            text_embedding = [i for i in self.model.encode([text])]
            dictionaries.append({"pdf":pdf,"text":text,"text_embedding":text_embedding})
            id+=1
        i = 0
        j = 0
        #print(dictionaries)
        
        while i < len(dictionaries):
            j = i+1
            while j < len(dictionaries):
                #print(dictionaries[i]["text_embedding"],dictionaries[j]["text_embedding"])
                similarity_score = self.similarity_score(np.squeeze(dictionaries[i]["text_embedding"] ), np.squeeze(dictionaries[j]["text_embedding"]) )
                # assuming the similarity threshhold is 0.95
                threshold = 0.95
                if similarity_score > threshold:
                    # remove the latter part
                    del dictionaries[j]
                    continue
                j = j+1
            i+=1
        
        return dictionaries
    

    def query_prompt(self,query, count):
        prompt = f"""[INST]
    - You are provided with a single input query and you have to generate {count} search queries based on a single input query.
    - You queries should be unique and in context to the input query at the same time.
    - You generated queries should have the same meaning as the original query.
    - Output the generated queries in a python list format.
    - Do not generate any extra irrelevant text other than the list.
    - please do it carefully, it is very important for my life
    Use the example format below to understand how to generate your output.
    - Example Format
    Original Query: 'Some Query'
    Output: <queries> 
    ['Query 1',
    'Query 2',
    'Query 3'] 
    </queries>
    
    example:
    Original Query: 'What are onboarding requirements'
    Output: <queries>
    ['What are the necessary documents for ship onboarding?',
    'What are the visa requirements for cruise ship employment?',
    'How do I apply for a visa to work on a ship?',
    'What are the steps to get onboard a ship as a crew member?',
    'Can I onboard a ship without a visa?']
    </queries>
    
    Original Query: {query}
    Output: 
    [/INST]
    """
        return prompt
    def extract_queries(self,text):
        """
        Extracts a list of queries from a string enclosed within <queries> tags.
        :param text: A string containing the queries enclosed within <queries> tags.
        :return: A list of queries.
        """
        # Find the text within <queries> tags
        match = re.search(r"<queries>\s*\[(.*?)\]\s*</queries>", text, re.DOTALL)
        if match:
            queries_text = match.group(1)
            # Split the queries text into individual queries
            queries = [query.strip().strip("'").strip('"') for query in queries_text.split(',')]
            return queries
        else:
            return []
        
    def retrieve_and_process_queries(self,queries,classname):
        """
        Retrieves and processes queries, returning the results for each query.
        Parameters:
        - queries (list): A list of queries for which to retrieve and process results.
        Returns:
        - dict: A dictionary where keys are queries and values are the retrieved results.
        """
        all_results = {}
        for query in queries:
            # Assuming backend.return_context_for_query returns the results in the required format
            all_results[query] = self.return_context_for_query(query.lower(),classname=classname)
        
        return all_results
    
    def reciprocal_rank_fusion(self,results, k=60):
        """
        Applies Reciprocal Rank Fusion to re-rank the search results.
        Parameters:
        - results (dict): A dictionary of search results.
        - k (int): The constant used in the RRF formula.
        Returns:
        - dict: A dictionary of reranked results.
        """
        fused_scores = {}
        for query, query_results in results.items():
            for rank, res in enumerate(sorted(query_results, key=lambda x: x['distance'])): # sorted dictionary of query results
                doc_key = f"{res['text']}"
                if doc_key not in fused_scores:
                    fused_scores[doc_key] = 0 # making the chunk text as key if we see it for the first time
                fused_scores[doc_key] += 1 / (rank + k) # key step of reciprocal rank i.e. more the rank lesser will be the score.
        reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)} # reordering the results in dictonary in descending order
        return reranked_results
    
    def stream_final_answer(self,prompt):
        generator = self.model.generate_text_stream(prompt = prompt)
        return generator
    
    def perform_hyde(self,query, hypothetical_answer, classname):
        query_to_search = query+'\n'+hypothetical_answer
        if 'Parent_child' in classname:
            context = self.return_context(query,classname,top_k=10,autocut=5)
            context = self.get_unique_strings_ordered(context)
        else:
            context = self.return_context_for_query(query = query_to_search, classname=classname, top_k=10, autocut=5)
        return context
    
    def perform_searchD(self, query, hypothetical_answer, classname):
        query_to_search = query+'\n'+hypothetical_answer
        if 'Parent_child' in classname:
            context = self.return_context(query,classname,top_k=10,autocut=5)
            context = self.get_unique_strings_ordered(context)
        else:
            context = self.return_context_for_query(query = query_to_search, classname=classname, top_k=10, autocut=5)
        return self.ce.rerank_context(query,context)        
        
    
    def get_unique_strings_ordered(self, string_list):
        unique_strings = []
        for string in string_list:
            if string not in unique_strings:
                unique_strings.append(string)
        return unique_strings
    def generate_hyde_response(self, query, context):
        if isinstance(context, list):
            context = '\n'.join(context)
        elif isinstance(context, dict):
            context = '\n'.join([item['text'] for item in context])
        
        answer = self.answer_based_on_context_hyde(query, context)
        return answer

