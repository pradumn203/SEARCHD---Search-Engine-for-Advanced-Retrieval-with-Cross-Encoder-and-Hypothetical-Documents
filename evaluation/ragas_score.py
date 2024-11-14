from evaluation.ragas_langchain_customization import CustomizedLangchainLLM
from ragas.metrics import faithfulness,context_precision,context_recall,context_relevancy,AnswerCorrectness
from ragas import evaluate
from datasets import Dataset

from genai.text.generation import (
    DecodingMethod,
    TextGenerationParameters,
)
from genai.extensions.langchain import LangChainInterface
from genai import Client, Credentials

from dotenv import load_dotenv
from typing import Dict, Union
import os
import sys

def get_wml_creds():
    load_dotenv()
    api_key = os.getenv("API_KEY",None)
    ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
    project_id = os.getenv("PROJECT_ID", None)
    if api_key is None or ibm_cloud_url is None or project_id is None:
        print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
        if sys.stdout.isatty():
            print(" Above creds are required.")
    else:
        creds = {
            "url": ibm_cloud_url,
            "apikey": api_key 
        }
    return project_id, creds


def get_watsonxllm_wrapper(
        model_id:str = "meta-llama/llama-2-70b-chat",
        parameters: Dict = {
            'decoding_method':DecodingMethod.GREEDY
        }
):
    """
    This Function will return the watsonx Lanchain wrapper.
    """
    llm = LangChainInterface(
    model_id=model_id,
    client=Client(credentials=Credentials.from_env()),
    parameters=TextGenerationParameters(
        decoding_method = parameters['decoding_method']
    ),
)
    return llm

def _get_ragas_score(dataset:Union[Dataset,None]=None, sample:Union[int,None]=None):
    llm = get_watsonxllm_wrapper()
    #faithfulness,context_precision,context_recall,context_relevancy,AnswerRelevancy,AnswerCorrectness,conciseness,AnswerSimilarity
    ragas_model = CustomizedLangchainLLM(llm=llm)
    faithfulness.llm = ragas_model
    context_precision.llm = ragas_model
    context_recall.llm = ragas_model
    context_relevancy.llm = ragas_model
    AnswerCorrectness.llm = ragas_model
    
    if sample:
        print(f"Dataset is not passed Creating sample results {sample=} from explodinggradients/fiqa")
        result = evaluate(
        dataset.select(range(sample)),  # showing only 5 for demonstration
            metrics=[faithfulness, context_precision, context_relevancy],
        )
    else:
        result = evaluate(dataset, metrics=[faithfulness, context_precision, context_relevancy ])
    return result