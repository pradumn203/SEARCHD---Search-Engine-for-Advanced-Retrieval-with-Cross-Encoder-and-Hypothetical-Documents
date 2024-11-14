# S.E.A.R.CH.D - Search Engine for Advanced Retrieval with Cross Encoder and Hypothetical Documents

Search Engine with Advanced Retrieval using Cross-encoding and Hypothetical Documents (SEARCHD) enhances the existing information retrieval mechanism and reduces the latency of LLM-based retrievers. This framework generates a partially correct document using a LLM which is clubbed along with the original query for context retrieval. The initial context which has a lower context precision is re-ranked by cross encoding and lower-ranked documents are eliminated based on a set threshold depending on the use case. This framework outperforms LLM-based retrievers such as HyDE in both accuracy and latency and re-ranking-based retrievers like RAG Fusion in accuracy on the MS-Marco Question-Answering Dataset with a significant enhancement of 12%.

## Steps to Run
After cloning the repo, install requirements by running.

```pip install -r requirements.txt```

You will need IBM cloud credentials as enviorment variables. Create a .env file in the root directory of this repo and fill the details as shown in ```.env-example``` file.

I am using weaviate as vectordb. The docker-compose file is given in the ```weaviate_module``` folder. Make sure you have one of the compatible docker daemon installed in your system then run:

```docker-compose up -d``` 

The first step is to generate the embeddings of your data. 
Follow the steps in the ```generate_embeddings_parent_child.ipynb``` notebook. You can modify the code and structure of your data to suit your needs.

Once setup is complete, you can run the SEARCHD notebook for context retrieval. 

## Citation

If you find our work helpul, Please cite our work in your research:

```apa
Mishra, P., Mahakali, A., & Venkataraman, P. S. (2024, August). SEARCHD-Advanced Retrieval with Text Generation using Large Language Models and Cross Encoding Re-ranking. In 2024 IEEE 20th International Conference on Automation Science and Engineering (CASE) (pp. 975-980). IEEE.
```

```bibtex
@INPROCEEDINGS{10711642,
  author={Mishra, Pradumn and Mahakali, Aditya and Venkataraman, Prasanna Shrinivas},
  booktitle={2024 IEEE 20th International Conference on Automation Science and Engineering (CASE)}, 
  title={SEARCHD - Advanced Retrieval with Text Generation using Large Language Models and Cross Encoding Re-ranking}, 
  year={2024},
  volume={},
  number={},
  pages={975-980},
  keywords={Adaptation models;Accuracy;Computer aided software engineering;Automation;Large language models;Search engines;Benchmark testing;Rendering (computer graphics);Encoding},
  doi={10.1109/CASE59546.2024.10711642}}
```