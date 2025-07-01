# LLM FastAPI Server

A production-ready FastAPI server for deploying Large Language Models (LLM) with Retrieval-Augmented Generation (RAG) capabilities. This project provides a comprehensive solution for serving custom LLMs with advanced features like document retrieval, embeddings, and chat functionality.

## 🚀 Features

- **Custom LLM Deployment**: Support for Hugging Face models with 4-bit quantization
- **RAG (Retrieval-Augmented Generation)**: Intelligent document retrieval and context-aware responses
- **Embedding Generation**: Text embedding creation using BGE-M3 model
- **Chat Interface**: Interactive chat functionality with conversation context
- **RESTful API**: Well-documented API endpoints with automatic OpenAPI documentation
- **Docker Support**: Containerized deployment for easy scaling
- **Production Ready**: Comprehensive logging, error handling, and health checks

## 🏗️ Project Structure

```
llm-fastapi-server/
├── src/
│   ├── main.py                    # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # API endpoint definitions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── llm_handler.py         # LLM model management
│   │   └── embedding_handler.py   # Embedding model management
│   ├── services/
│   │   ├── __init__.py
│   │   └── rag_service.py         # RAG functionality
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration management
│   │   └── logger.py              # Logging setup
│   └── utils/
│       ├── __init__.py
│       └── helpers.py             # Utility functions
├── notebooks/
│   ├── llm_setup.ipynb            # Model setup and testing
│   └── rag_development.ipynb      # RAG development notebook
├── data/
│   └── vector_db/                 # Vector database storage
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose setup
└── README.md                      # Project documentation
```

## 🔧 Installation

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended for larger models)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd llm-fastapi-server
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Hugging Face token**
   ```bash
   # Set your Hugging Face token as environment variable
   export HUGGINGFACE_TOKEN="your_token_here"
   ```

## 🚀 Usage

### Running the Server

#### Method 1: Direct Python execution
```bash
cd src
python main.py
```

#### Method 2: Using uvicorn
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Method 3: Using Docker
```bash
# Build and run with Docker Compose
docker-compose up -d
```

### API Endpoints

Once the server is running, access the interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

#### Available Endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message and endpoint overview |
| `/api/v1/health` | GET | Health check and service status |
| `/api/v1/generate` | POST | Generate text using LLM |
| `/api/v1/chat` | POST | Chat with the LLM |
| `/api/v1/embed` | POST | Create text embeddings |
| `/api/v1/rag` | POST | RAG-powered question answering |

### Example API Calls

#### Text Generation
```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explain artificial intelligence", "max_length": 512}'
```

#### Chat
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What is machine learning?"}'
```

#### RAG Query
```bash
curl -X POST "http://localhost:8000/api/v1/rag" \
     -H "Content-Type: application/json" \
     -d '{"question": "트랜스포머 모델이 무엇인가요?"}'
```

#### Create Embedding
```bash
curl -X POST "http://localhost:8000/api/v1/embed" \
     -H "Content-Type: application/json" \
     -d '{"text": "This is a sample text for embedding"}'
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HUGGINGFACE_TOKEN` | Hugging Face API token | Required |
| `MODEL_ID` | LLM model identifier | `Qwen/Qwen2.5-1.5B-Instruct` |
| `EMBEDDING_MODEL` | Embedding model name | `BAAI/bge-m3` |
| `MAX_TOKENS` | Maximum generation tokens | `512` |
| `TEMPERATURE` | Generation temperature | `0.7` |

### Model Configuration

The server supports various Hugging Face models:

- **Small Models (1B-3B)**: `Qwen/Qwen2.5-1.5B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`
- **Medium Models (7B-13B)**: `meta-llama/Llama-2-7b-chat-hf`, `mistralai/Mistral-7B-Instruct-v0.1`
- **Large Models (30B+)**: Requires multi-GPU setup

## 📊 Performance

### System Requirements

| Model Size | RAM | GPU Memory | Recommended GPU |
|------------|-----|------------|-----------------|
| 1B-3B | 8GB | 4GB | GTX 1660, RTX 3060 |
| 7B-13B | 16GB | 8GB | RTX 3080, RTX 4070 |
| 30B+ | 32GB | 24GB+ | RTX 4090, A100 |

### Optimization Features

- **4-bit Quantization**: Reduces memory usage by ~75%
- **Lazy Loading**: Models load only when first accessed
- **Efficient Caching**: Vector database persistence
- **Async Processing**: Non-blocking API responses

## 🧪 Development

### Running Tests
```bash
# Run the development notebooks
jupyter lab notebooks/
```

### Adding New Models

1. Update `llm_handler.py` with new model configuration
2. Modify `requirements.txt` if new dependencies are needed
3. Test with the provided notebooks

## 📦 Docker Deployment

### Build Image
```bash
docker build -t llm-fastapi-server .
```

### Run Container
```bash
docker run -p 8000:8000 \
  -e HUGGINGFACE_TOKEN="your_token" \
  -v $(pwd)/data:/app/data \
  llm-fastapi-server
```

### Production Deployment
```bash
docker-compose up -d
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing excellent model hosting
- [FastAPI](https://fastapi.tiangolo.com/) for the amazing web framework
- [LangChain](https://langchain.com/) for RAG capabilities
- [Sentence Transformers](https://www.sbert.net/) for embedding models

## 📞 Support

For questions and support:

- Create an issue in this repository
- Check the [documentation](http://localhost:8000/docs) when server is running
- Review the example notebooks in the `notebooks/` directory

---

**Note**: This server is designed for educational and development purposes. For production deployment, ensure proper security measures, authentication, and scaling configurations.
│   │   └── retrieval_service.py # Handles document retrieval
│   ├── core                   # Core application components
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration settings
│   │   └── logger.py          # Logging setup
│   └── utils                  # Utility functions
│       ├── __init__.py
│       └── helpers.py         # Helper functions
├── notebooks                  # Jupyter notebooks for setup and development
│   ├── llm_setup.ipynb
│   └── rag_development.ipynb
├── data                       # Directory for vector database files
│   └── vector_db
├── requirements.txt           # Project dependencies
├── Dockerfile                 # Docker image instructions
├── docker-compose.yml         # Docker application configuration
└── README.md                  # Project documentation
```

## Setup Instructions

1. **Clone the Repository**
   ```
   git clone <repository-url>
   cd llm-fastapi-server
   ```

2. **Install Dependencies**
   Use pip to install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. **Run the FastAPI Server**
   You can start the FastAPI server using:
   ```
   uvicorn src.main:app --reload
   ```

4. **Access the API**
   Once the server is running, you can access the API at `http://127.0.0.1:8000`. The interactive API documentation can be found at `http://127.0.0.1:8000/docs`.

## Usage Examples

- **Generate a Response**
  Send a POST request to the `/generate` endpoint with your query to receive a response from the LLM.

- **Retrieve Documents**
  Use the `/retrieve` endpoint to fetch relevant documents from the vector database.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.