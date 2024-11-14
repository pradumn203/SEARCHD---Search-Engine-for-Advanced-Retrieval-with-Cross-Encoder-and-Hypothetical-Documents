from sentence_transformers import SentenceTransformer # word2vec DL model
import numpy as np
from numpy.linalg import norm
import sys
'''
     Sentence transformer id s word2vec deep learning MODEl 
     which uses unsupervised clustering algorithms to cluster words in a higher dimension space
     384, 784, 1024, 2048, 4096
     how to pick text embeding model, Look at MTEB: massive text embedding hugging face
     
     tlimitation is the sequence length, which is 512 by default
     
'''
import weaviate # weaviate-client
import json
np.set_printoptions(threshold=sys.maxsize)
class WeaviateWrapper:
    '''
    host: is the address of the weaviate instance, e.g. http://localhost:8090 
        for local docker instance
    embedding_model_name: name of the sentence transformer model
    '''
    def __init__(self,host="http://localhost:8090",embedding_model_name="intfloat/e5-large-v2",device="cpu"):
        '''
            embedding model can be selected from Huggingface based on the Language(s) involved and
            task i.e. classification, retreival, etc.
            see https://huggingface.co/blog/mteb [Massive Text Embedding Benchmark]
            
            device by default I have selected to be CPU on deployment, if a GPU(cuda) is available select that
        '''
        self.host = host
        self.client = weaviate.Client(host)
        self.model = self.load_model(embedding_model_name,device=device)
        
    def load_model(self,embedding_model_name,device="cpu"):
        model = SentenceTransformer(embedding_model_name,device=device)
        model.max_seq_length = 512
        return model
    
    def add_schema(self,schema):
        self.client.schema = schema
    
    def add_class_to_schema(self,classconfiguration):
        '''
                class_obj = {
                'class': 'rag',
                'properties': [
                    {
                        'name': 'pdf',
                        'dataType': ['text'],
                    },
                    {
                        'name': 'text',
                        'dataType': ['text'],
                    },
                    {
                        'name': 'chunk_number',
                        'dataType': ['int'],
                    }
                ],
            }
        '''
        self.client.schema.create_class(classconfiguration)
        
    def delete_weaviate_class(self,classname):
        self.client.schema.delete_class(classname)
    
    def add_object_to_schema(self,classname,pdf,text,chunk_number):
        embeds = self.model.encode(text)
        self.client.data_object.create(
            data_object={
                'pdf':pdf,
                'text':text,
                'chunk_number':chunk_number
            },
            class_name=classname,
            vector=embeds
        )
        
    def hybrid_search_relevant_objects(self,user_query,classname,parameters,top_k=3,autocut=1):
        input_embedding = [i for i in self.model.encode([user_query])]
        response = (
                self.client.query
                .get(classname, parameters)
                .with_hybrid(
                    query=user_query,
                    vector=input_embedding[0]
                )
                .with_limit(top_k)
                .with_autocut(autocut) 
                .with_additional(["score"])
                .do()
            )
        return json.dumps(response,indent=4)

    def similarity_score(self,a,b):
        cos_sim = np.dot(a, b)/(norm(a)*norm(b))
        return cos_sim
    
    def search_relevant_objects(self,user_query,classname,parameters,top_k=3,autocut=3):
        input_embedding = [i for i in self.model.encode([user_query])]
        '''
        if the distances for six objects returned by nearText were [0.1899, 0.1901, 0.191, 0.21, 0.215, 0.23]
                then autocut: 1 would return the first three objects,
                autocut: 2 would return all but the last object,
                and autocut: 3 would return all objects.
                basically it indicated the number of jumps after which it will not take any chunks
                jump means jump in distance metrics
        '''
        response = (
                self.client.query
                .get(classname, parameters)
                .with_near_vector(
                    {
                        "vector":input_embedding[0]
                    }
                )
                .with_additional(["distance","certainty"])
                .with_autocut(autocut) 
                .with_limit(top_k)
                .do()
                
            )
        return json.dumps(response,indent=4)
    
    def hybrid_search_relevant_objects_where(self,user_query,classname,parameters,pdf,top_k=3,autocut=1):
        input_embedding = [i for i in self.model.encode([user_query])]
        search_string = f"*{pdf}*"
        response = (
                self.client.query
                .get(classname, parameters)
                .with_hybrid(
                    query=user_query.lower(),
                    vector=input_embedding[0],
                    # https://weaviate.io/developers/weaviate/search/hybrid#weight-boost-searched-properties
                    properties=["text"]
                ).with_where({
                    "path": ["pdf"],
                    "operator": "Like",
                    "valueText": search_string
                }
                )
                
                .with_limit(top_k)
                .with_additional(["score"])
                .with_autocut(autocut) 
                .do()
            )
        return json.dumps(response,indent=4)
    
    def search_relevant_objects_where(self,input_text,classname,parameters,pdf,top_k=3,autocut=3):
        input_embedding = [i for i in self.model.encode([input_text])]
        search_string = f"*{pdf}*"
        '''
        if the distances for six objects returned by nearText were [0.1899, 0.1901, 0.191, 0.21, 0.215, 0.23]
                then autocut: 1 would return the first three objects,
                autocut: 2 would return all but the last object,
                and autocut: 3 would return all objects.
        '''
        response = (
                self.client.query
                .get(classname, parameters)
                .with_near_vector(
                    {
                        "vector":input_embedding[0]
                    }
                )
                .with_where({
                    "path": ["pdf"],
                    "operator": "Like",
                    "valueText": search_string
                }
                )
                .with_additional(["distance","certainty"])
                .with_autocut(autocut) 
                .with_limit(top_k)
                .do()
                
            )
        return json.dumps(response,indent=4)

    def get_schema(self):
        return self.client.schema.get()
    
    def delete_all_objects_of_document(self,classname, document_name):
        self.client.batch.delete_objects(
        class_name=classname,
        # Same `where` filter as in the GraphQL API
        where={
            'path': ['pdf'],
            'operator': 'Equal',
            'valueText': document_name
        },
    )
    
        
    def get_unique_data(self,data,classname):
        # data's text value we will do chunk simlarity
        chunks= data["data"]["Get"][classname] # getting chunks
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
                
                
    def add_parent_child_object_to_schema(self,classname,parent_text,child_text,chunk_number):
        #encoding the child text only
        embeds = self.model.encode(child_text)
        # but storing the full text
        self.client.data_object.create(
            data_object={
                'child_text':child_text,
                'text':parent_text,
                'chunk_number':chunk_number
            },
            class_name=classname,
            vector=embeds
        )
        