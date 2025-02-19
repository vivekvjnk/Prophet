from langchain_ollama import OllamaEmbeddings
from pymilvus import model


# embeddings = OllamaEmbeddings(model="llama3.1:latest")
embeddings = model.dense.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L6-v2', # Specify the model name
    device='cpu' # Specify the device to use, e.g., 'cpu' or 'cuda:0'
)

from langchain_milvus import Milvus

# The easiest way is to use Milvus Lite where everything is stored in a local file.
# If you have a Milvus server you can use the server URI such as "http://localhost:19530".
URI = "http://localhost:19530"

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    # Set index_params if needed
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)

from langchain_core.documents import Document

vector_store_saved = Milvus.from_documents(
    [Document(page_content="foo!")],
    embeddings,
    collection_name="langchain_example",
    connection_args={"uri": URI},
)


vector_store_loaded = Milvus(
    embeddings,
    connection_args={"uri": URI},
    collection_name="langchain_example",
)

print(vector_store_loaded)