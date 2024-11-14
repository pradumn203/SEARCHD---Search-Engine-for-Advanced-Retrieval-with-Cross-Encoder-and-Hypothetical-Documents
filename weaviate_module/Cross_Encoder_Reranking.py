from sentence_transformers import CrossEncoder
import os
from typing import Dict,List

class Reranker:
    
    def __init__(self,cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
    
        self.cs_model = CrossEncoder(cross_encoder_model, max_length=512)
        
    
    def rerank(self, pairs: List[List[str]]) -> List[Dict]:
        
        ce_scores = self.cs_model.predict(pairs, show_progress_bar=True)
        
        return ce_scores
    
    def rerank_context(self, query, contexts:List):
        ce_scores = self.cs_model.predict([[query, context] for context in contexts])        
        result = []
        for idx in range(len(contexts)):
            result.append({
                "cross_encoder_score": float(ce_scores[idx]),
                "context": contexts[idx]
            })
        sorted_result = sorted(result, key=lambda x: x["cross_encoder_score"], reverse=True)
        final_result = [res['context'] for res in sorted_result]
        return final_result

if __name__ == "__main__":
    # demo use
    reranker = Reranker()
    def rank_products(query,products):
        ce_scores = reranker.rerank([[query, product["final_summary"]] for product in products])
        result = []
        for idx in range(len(products)):
                result.append({
                    "cross_encoder_score": float(ce_scores[idx]),
                    "final_summary": products[idx]["final_summary"],  # i.e., the answer,
                    "index":idx,
                    "ndc":products[idx]["ndc"],
                    "score":products[idx]["score"]
                })
        # Sorting the list of dictionaries by 'cross-encoder_score' in descending order
        sorted_result = sorted(result, key=lambda x: x["cross_encoder_score"], reverse=True)
        return sorted_result
    

