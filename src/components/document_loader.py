from datasets import load_dataset
from haystack import Document
from typing import List, Dict, Any

class DocumentLoader:
    """Handles loading and processing of documents from datasets."""
    
    def __init__(self, datasets_config: Dict[str, Any]):
        self.datasets_config = datasets_config
        
    def load_documents(self) -> List[Document]:
        """Main method to load all configured documents."""
        documents = []
        
        # Load Hugging Face datasets
        hf_documents = self.load_hf_documents()
        documents.extend(hf_documents)
        
        print(f"Loaded {len(documents)} documents in total")
        return documents
    
    def load_hf_documents(self) -> List[Document]:
        """Load documents from Hugging Face datasets specified in config."""
        all_documents = []
        
        # Get the list of Hugging Face datasets from config
        hf_datasets = self.datasets_config.get("huggingface", [])
        
        for dataset_name in hf_datasets:
            try:
                print(f"Loading Hugging Face dataset: {dataset_name}")
                dataset = load_dataset(dataset_name, split="train")
                
                if len(dataset) == 0:
                    print(f"Warning: Dataset {dataset_name} returned empty results")
                    continue
                    
                # Convert dataset items to Haystack Documents
                documents = [Document(content=doc["content"], meta=doc.get("meta", {})) 
                             for doc in dataset]
                
                all_documents.extend(documents)
                print(f"Loaded {len(documents)} documents from dataset '{dataset_name}'")
                
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {str(e)}")
        
        return all_documents