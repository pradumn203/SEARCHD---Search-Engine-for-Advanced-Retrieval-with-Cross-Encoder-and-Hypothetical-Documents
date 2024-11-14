import os
import sys
sys.path.insert(0, os.path.abspath(".."))
from weaviate_module.weaviate_wrapper import WeaviateWrapper
import fitz
import pandas as pd
import ast
from langchain.text_splitter import SpacyTextSplitter

def text_to_chunks(texts: str,
                   chunk_length: int = 100,
                   chunk_overlap: int = 25) -> list:
    """
    Splits the text into equally distributed chunks with 25-word overlap.
    Args:
        texts (str): Text to be converted into chunks.
        chunk_length (int): Maximum number of words in each chunk.
        chunk_overlap (int): Number of words to overlap between chunks.
    """
    words = texts.split(' ')
    n = len(words)
    chunks = []
    chunk_number = 1
    i = 0
    while i < n:  # Corrected the length check
        chunk = words[i: min(i + chunk_length, n)]
        i = i + chunk_length - chunk_overlap
        #print(len(chunk))
        chunk = ' '.join(chunk).strip()
        chunks.append({"text": chunk, "chunk_number": chunk_number})
        chunk_number += 1
    return chunks

def parent_child_splitting(text: str, number_of_children: int, child_overlap: int = 10) -> list:
    """
    Splits the parent text into 'number_of_children' chunks, each chunk containing a portion of the full text.
    There will be an overlap of 'child_overlap' words between consecutive chunks.

    Args:
        text (str): Parent text to be split into chunks.
        number_of_children (int): Number of chunks to split the text into.
        child_overlap (int): Number of words to overlap between consecutive chunks.

    Returns:
        list: List containing each chunk of text.
    """
    words = text.split(' ')
    total_words = len(words)
    
    # Calculate the length of each chunk (excluding overlap)
    if number_of_children <= 1:
        chunk_length = total_words
    else:
        chunk_length = (total_words + (child_overlap * (number_of_children - 1))) // number_of_children

    chunks = []
    i = 0
    for _ in range(number_of_children):
        start_index = max(0, i)
        end_index = min(i + chunk_length, total_words)
        chunk = words[start_index:end_index]
        i = end_index - child_overlap
        chunk_text = ' '.join(chunk).strip()
        chunks.append(chunk_text)

    return chunks

def parent_to_child(parent_chunk:str)->list:
    text_splitter = SpacyTextSplitter(pipeline="en_core_web_sm",separator='##')
    return [sent for sent in  text_splitter.split_text(parent_chunk)[0].split('##')]

ww = WeaviateWrapper(host="http://localhost:8080")
class_obj = {
                'class': 'Parent_child_chunks',
                'properties': [
                    {
                        'name': 'child_text',
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

if not ww.client.schema.exists('Parent_child_chunks'):
    ww.add_class_to_schema(classconfiguration=class_obj)
    
df = pd.read_csv("../data/ms-marco-200-rows.csv")
df.head()

texxt =""
for context in df["contexts"]:
    context = ast.literal_eval(context)
    text = '\n'.join(context)
    texxt += text + "\n"

len(texxt)

chunks = text_to_chunks(texxt)
    
for chunk in chunks:
    chunk_number = chunk["chunk_number"]
    chunk_text = chunk["text"]
    print("-"*100)
    # parent 300 words, each children will have 50 words with an overlap of 5 words
    #child_texts = parent_child_splitting(text=chunk_text,number_of_children=6,child_overlap=5)
    child_texts = parent_to_child(chunk_text)
    for child_text in child_texts:
        ww.add_parent_child_object_to_schema(classname="Parent_child_chunks",
                                             parent_text=chunk_text,
                                             chunk_number=chunk_number,
                                             child_text=child_text)
    print(f"added {chunk_number} of rag_dataset")