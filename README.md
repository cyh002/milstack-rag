# MilStack RAG

A sophisticated Retrieval-Augmented Generation (RAG) system built with Haystack, vLLM, and Milvus, featuring hybrid search capabilities and conversational memory.

## 🚀 Features

- **Hybrid Search**: Combines dense and sparse embeddings for superior retrieval accuracy
- **Multiple LLM Providers**: Support for local vLLM and OpenAI models
- **Conversational Memory**: Maintains context across queries for natural conversations
- **Multi-format Document Support**: Process TXT, PDF, CSV, JSON files and HuggingFace datasets
- **Flexible Configuration**: YAML-based configuration with environment variable support
- **Interactive UI**: Clean Gradio interface for easy interaction
- **Modular Architecture**: Component-based design for easy extension and customization

## 📋 Requirements

- Python 3.12+
- uv package manager
- GPU (recommended for vLLM)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd milstack-rag
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Set up environment variables** (create `.env` file):
   ```bash
   # LLM Configuration
   VLLM_MODEL_NAME="SeaLLMs/SeaLLMs-v3-1.5B-Chat"
   VLLM_BASE_URL="http://localhost:8000/v1"
   VLLM_API_KEY="token-abc123"

   # Dense Embedding Model
   DENSE_EMBEDDING_MODEL_NAME="Snowflake/snowflake-arctic-embed-l"
   DENSE_EMBEDDING_DIMENSION=1024

   # Sparse Embedding Model  
   SPARSE_EMBEDDING_MODEL_NAME="prithivida/Splade_PP_en_v1"

   # Milvus Configuration
   MILVUS_LOAD_TYPE="milvus-lite"
   MILVUS_COLLECTION_NAME="hybrid_docs"
   MILVUS_LITE_DB_PATH="./milvus_hybrid.db"
   MILVUS_SPARSE_VECTOR_FIELD_NAME="sparse_embedding"

   # Optional: OpenAI Configuration
   OPENAI_MODEL_NAME="gpt-4"
   OPENAI_API_KEY="your-openai-key"
   OPENAI_EMBEDDINGS_MODEL_NAME="text-embedding-3-large"
   ```

## 🚀 Quick Start

### 1. Start vLLM Server
```bash
uv run vllm serve "SeaLLMs/SeaLLMs-v3-1.5B-Chat" --api-key token-abc123 --port 8000
```

### 2. Run the RAG Application
```bash
uv run src/rag_app.py
```

This will:
- Load and index documents from the `data/` directory
- Start the hybrid RAG pipeline
- Launch the Gradio web interface
- Run a test query to verify everything works

## 📁 Project Structure

```
milstack-rag/
├── src/
│   ├── rag_app.py              # Main application entry point
│   ├── components/             # Core components
│   │   ├── config.py           # Configuration management
│   │   ├── document_manager.py # Document loading and management
│   │   ├── document_store.py   # Milvus document store factory
│   │   ├── embedder.py         # Dense and sparse embedders
│   │   ├── llm.py             # LLM providers (vLLM, OpenAI)
│   │   ├── memory_store.py     # Conversation memory
│   │   └── document_utils/     # Document processing utilities
│   ├── pipeline/               # RAG pipeline components
│   │   ├── rag_pipeline.py     # Main RAG pipeline
│   │   ├── pipeline_builder.py # Pipeline construction
│   │   ├── component_factory.py # Component factory
│   │   └── template_provider.py # Prompt templates
│   └── ui/
│       └── gradio_interface.py # Web interface
├── conf/
│   └── config.yaml             # Main configuration file
├── data/                       # Document storage
│   ├── txt_files/
│   ├── pdf_files/
│   ├── csv_files/
│   └── __init__.py
└── pyproject.toml             # Dependencies and project info
```

## 📚 Data Sources

The system supports multiple data sources:

### File-based Sources
- **Text files** (`.txt`): Plain text documents
- **PDF files** (`.pdf`): Extracted using PyPDF
- **CSV files** (`.csv`): Structured data
- **JSON files** (`.json`): Structured JSON data

### HuggingFace Datasets
Configure in `conf/config.yaml`:
```yaml
datasets:
  huggingface:
    - "bilgeyucel/seven-wonders"
    - "your-dataset-name"
```

## ⚙️ Configuration

### LLM Providers
Switch between local vLLM and OpenAI:
```yaml
llm:
  provider: "local"  # or "openai"
```

### Embedding Models
Configure dense and sparse embeddings:
```yaml
embedding:
  dense_provider: "sentence_transformers"
  sentence_transformers:
    document_embedder: "Snowflake/snowflake-arctic-embed-l"
  sparse_embedder:
    model: "prithivida/Splade_PP_en_v1"
```

### Vector Database
Choose between Milvus Lite or Server:
```yaml
vector_db:
  load_type: "milvus-lite"  # or "milvus-server"
  collection_name: "hybrid_docs"
```

## 🔧 Usage Examples

### Programmatic Usage
```python
from src.rag_app import MilstackRAG
from src.components.config import ConfigLoader

# Load configuration
config_loader = ConfigLoader("conf/config.yaml")
config = config_loader.get_config()

# Create and setup RAG application
app = MilstackRAG(config=config).setup()

# Run queries
answer = app.run_query("What are the Seven Wonders of the World?")
print(answer)
```

### Web Interface
Access the Gradio interface at `http://localhost:7860` after running the application.

## 🏗️ Architecture

### Hybrid Search Pipeline
1. **Query Processing**: User query is processed and potentially rephrased
2. **Dual Embedding**: Query is embedded using both dense and sparse models
3. **Hybrid Retrieval**: Milvus performs hybrid search combining both embedding types
4. **Context Assembly**: Retrieved documents are assembled with conversation history
5. **Generation**: LLM generates response using retrieved context

### Key Components
- **Document Store**: Milvus vector database with hybrid search support
- **Embedders**: Dense (Sentence Transformers) + Sparse (SPLADE) embeddings
- **Memory**: Conversation history management
- **Pipeline**: Haystack-based processing pipeline
- **UI**: Gradio web interface

## 🔍 Advanced Features

### Query Rephrasing
The system automatically rephrases queries based on conversation context for better retrieval.

### Memory Management
Maintains conversation history to provide contextual responses across multiple turns.

### Document Preprocessing
Automatic document splitting and preprocessing for optimal retrieval performance.

## 🐛 Troubleshooting

### Common Issues

1. **vLLM Connection Error**:
   - Ensure vLLM server is running on the correct port
   - Check `VLLM_BASE_URL` in environment variables

2. **Memory Issues**:
   - Reduce batch size for document processing
   - Use smaller embedding models if GPU memory is limited

3. **Milvus Errors**:
   - Check Milvus database path permissions
   - Ensure collection schema matches embedding dimensions

### Logging
Check `debug.log` for detailed operation logs and error traces.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- [Haystack](https://haystack.deepset.ai/) - NLP framework
- [Milvus](https://milvus.io/) - Vector database
- [vLLM](https://github.com/vllm-project/vllm) - Fast LLM inference
- [Gradio](https://gradio.app/) - Web interface framework