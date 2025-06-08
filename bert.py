from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

class FinBERTSimilarity:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModel.from_pretrained("ProsusAI/finbert")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def embed(self, texts):
        """Convert texts to FinBERT embeddings."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Average pooling
    
    def compute_similarity(self, texts_1, texts_2):
        """Compute cosine similarity between two text lists."""
        embeddings_1 = self.embed(texts_1)
        embeddings_2 = self.embed(texts_2)
        return cosine_similarity(embeddings_1.cpu(), embeddings_2.cpu())
        

# # Example usage
# finbert = FinBERTSimilarity()
# texts_a = ["Revenue increased by 5%", "Net profit fell 10%"]
# texts_b = ["Sales grew 5%", "Profit dropped 10%"]
# similarity_matrix = finbert.compute_similarity(texts_a, texts_b)
# print(similarity_matrix)