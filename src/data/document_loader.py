from datasets import load_dataset
from haystack import Document

class DocumentLoader:
    """Handles loading and processing of documents from datasets."""
    
    def __init__(self, dataset_name, split="train"):
        self.dataset_name = dataset_name
        self.split = split
        
    def load_documents(self):
        """Load dataset and convert to Haystack Document objects."""
        dataset = load_dataset(self.dataset_name, split=self.split)
        if len(dataset) == 0:
            raise ValueError(f"Dataset {self.dataset_name} returned empty results")
        documents = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]
        print(f"Loaded {len(documents)} documents from dataset '{self.dataset_name}'")
        return documents