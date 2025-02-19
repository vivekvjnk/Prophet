from dataclasses import dataclass
from abc import ABC,abstractmethod
from langchain.docstore.document import Document
from typing import Any, Dict, List, Optional
import os
from abc import ABC, abstractmethod
import networkx as nx

#---------------plugin import-Begin-------------#
from .retrieval_stratergies.plugin import *
#---------------plugin import-End---------------#   

#---------------Graph-Begin-------------#
class GraphLoader:
    """
    A modular and extensible wrapper for loading NetworkX graphs from various sources.
    
    Core Design Principles:
    -----------------------
    1. **Modular Handlers**:
        Each source type (e.g., file, in-memory state) has a dedicated handler function
        to process and load graphs. This ensures clean separation of concerns.
    
    2. **Central Registry**:
        A registry is used to map source types to their respective handler functions.
        The registry enables dynamic routing of requests to the appropriate handler.
    
    3. **Extensibility**:
        New sources can be added seamlessly by implementing a handler function and 
        registering it with the central registry using the `register_handler` method.
        This allows for minimal to no modification of the existing code when expanding 
        functionality.
    
    Attributes:
    -----------
    handlers : dict
        A dictionary mapping source types (str) to their respective handler functions.

    Methods:
    --------
    load_graph(source_type, *args, **kwargs)
        General wrapper to load a NetworkX graph from a specified source type.
    
    register_handler(source_type, handler_function)
        Register a new handler function for a specific source type.
    
    Private Methods:
    ----------------
    _load_from_file(file_path, file_format="json")
        Load a graph from a file (supports JSON, YAML, and pickle formats).
    
    _load_from_memory(state, graph_key)
        Load a graph from an in-memory state (e.g., dictionary).
    """

    def __init__(self):
        """
        Initializes the GraphLoader with a registry of handlers for various graph sources.
        
        By default, the following source types are supported:
        - "file": For loading graphs from files (JSON, YAML, pickle).
        - "in_memory": For loading graphs from an in-memory dictionary.
        """
        self.handlers = {
            "file": self._load_from_file,
            "in_memory": self._load_from_memory,
        }

    def load_graph(self, source_type, *args, **kwargs):
        """
        General wrapper to load a NetworkX graph from a specified source type.
        
        Parameters:
        -----------
        source_type : str
            The type of the source to load the graph from (e.g., "file", "in_memory").
        *args : tuple
            Positional arguments to pass to the handler function.
        **kwargs : dict
            Keyword arguments to pass to the handler function.
        
        Returns:
        --------
        nx.Graph
            The loaded NetworkX graph.
        
        Raises:
        -------
        ValueError
            If an unknown source_type is provided.
        
        Future updates:
        --------------
        Add support for parquet format
        """
        if source_type not in self.handlers:
            raise ValueError(f"Unknown source type: {source_type}. Available types: {list(self.handlers.keys())}")
        return self.handlers[source_type](*args, **kwargs)

    def _load_from_file(self, file_path, file_format="json"):
        """
        Load a graph from a file.
        
        Supported formats: JSON, YAML, pickle.
        
        Parameters:
        -----------
        file_path : str
            Path to the file.
        file_format : str
            Format of the file (default: "json"). Supported formats: "json", "yaml", "pickle".
        
        Returns:
        --------
        nx.Graph
            The loaded NetworkX graph.
        
        Raises:
        -------
        ValueError
            If an unsupported file format is provided.
        TypeError
            If the file does not contain a valid NetworkX graph (for pickle format).
        """
        if file_format == "json":
            with open(file_path, "r") as f:
                data = json.load(f)
            return nx.node_link_graph(data)
        elif file_format == "yaml":
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
            return nx.node_link_graph(data)
        elif file_format == "pickle":
            with open(file_path, "rb") as f:
                graph = pickle.load(f)
            if not isinstance(graph, nx.Graph):
                raise TypeError("Pickle file does not contain a valid NetworkX graph.")
            return graph
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def _load_from_memory(self, state, graph_key):
        """
        Load a graph from an in-memory state.
        
        Parameters:
        -----------
        state : dict
            An in-memory dictionary storing graphs.
        graph_key : str
            Key to retrieve the graph from the dictionary.
        
        Returns:
        --------
        nx.Graph
            The loaded NetworkX graph.
        
        Raises:
        -------
        KeyError
            If the specified key is not found in the state.
        TypeError
            If the value at the specified key is not a valid NetworkX graph.
        """
        if graph_key not in state:
            raise KeyError(f"Graph key '{graph_key}' not found in the provided state.")
        graph = state[graph_key]
        if not isinstance(graph, nx.Graph):
            raise TypeError("The provided state does not contain a valid NetworkX graph at the specified key.")
        return graph

    def register_handler(self, source_type, handler_function):
        """
        Register a new handler function for a specific source type.
        
        Parameters:
        -----------
        source_type : str
            The type of the source (e.g., "api", "database").
        handler_function : callable
            A function to handle loading graphs from the specified source type.
        
        Raises:
        -------
        ValueError
            If a handler for the specified source type already exists.
        """
        if source_type in self.handlers:
            raise ValueError(f"Handler for source type '{source_type}' already exists.")
        self.handlers[source_type] = handler_function
'''
Graph loader usage example 

# Initialize the graph loader
loader = GraphLoader()

# Load from JSON file
graph_from_file = loader.load_graph("file", file_path="graph.json", file_format="json")

# Load from in-memory state
state = {"my_graph": nx.complete_graph(5)}
graph_from_memory = loader.load_graph("in_memory", state=state, graph_key="my_graph")

# Add a new handler (e.g., for Neo4j) in the future
def load_from_neo4j(*args, **kwargs):
    # Placeholder for Neo4j handler
    pass

loader.register_handler("neo4j", load_from_neo4j)

'''
#---------------Graph-End---------------#


class RetrievalPlugin(ABC):
    """Abstract base class for retrieval plugins."""
    
    def __init__(self,sources:list):
        self.loader = GraphLoader()
        self.source_list = sources
        self.artifact_path = os.path.join(os.path.dirname(__file__),"../artifacts")
        self.graph = {}

    @abstractmethod
    def retrieve(self, query: str, **kwds):
        """Retrieve relevant results based on the query."""
        pass


class RetrievalFactory:
    """Manages retrieval plugins."""
    
    def __init__(self,**kwargs):
        self.plugins = {}
        self.load_plugins(**kwargs)

    def register_plugin(self, name: str, plugin: RetrievalPlugin):
        """Register a new retrieval plugin."""
        self.plugins[name] = plugin

    def retrieve(self, query: str, **kwargs):
        """Use the specified retrieval plugin to fetch results."""
        if 'strategy' in kwargs:
            strategy = kwargs.get('strategy')
        else:
            raise(ValueError("Missing key 'strategy'"))
        
        if strategy not in self.plugins:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
        
        del kwargs['strategy']
        return self.plugins[strategy].retrieve(query, **kwargs)
    
    def load_plugins(self,**kwargs):
        plugins= load_plugins(**kwargs)
        self.plugins = plugins['plugins']
# Data class for retrieval output 
@dataclass
class RetrievalOutput:
    """
    A standardized output interface for retrieval plugins in the Odysseus module.

    The `RetrievalOutput` class provides a unified structure for representing the results
    of retrieval operations. It is designed to support multiple data modalities (e.g., text, 
    images, audio) while maintaining compatibility with graph-based processing workflows.

    Attributes:
        modality (str): 
            The modality of the retrieved data, such as "text", "image", or "audio". 
            This indicates the type of data contained in the `data` attribute.
        data (List[Dict[str, Any]]): 
            A list of dictionaries, each representing a retrieved item or result. 
            Each dictionary's structure depends on the modality but may include:
            - For text: `{"content": "retrieved content", "metadata": {"source": "source_name", "confidence": 0.95}}`
            - For other modalities: Similar entries customized for that data type.
        metadata (Dict[str, Any]): 
            Additional information about the retrieval process, such as:
            - Query parameters.
            - Provenance of the results.
            - Confidence scores or retrieval performance metrics.
        context_graph (Optional[Any]): 
            A NetworkX or similar graph structure representing relationships and context 
            associated with the retrieved data. Useful for graph-based pipelines and 
            downstream processing. This attribute is optional and may be omitted if 
            not applicable to the retrieval results.
        retrieval_summary (Optional[str]): 
            A summarized description of the retrieval results, providing an overview 
            of the key findings or insights. This attribute is optional and can be 
            generated dynamically or provided by the retrieval plugin.

    Example Usage:
        >>> output = RetrievalOutput(
        ...     modality="text",
        ...     data=[
        ...         {"content": "Paris is the capital of France.", 
        ...          "metadata": {"source": "encyclopedia", "confidence": 0.95}},
        ...         {"content": "The Eiffel Tower is located in Paris.", 
        ...          "metadata": {"source": "travel guide", "confidence": 0.92}}
        ...     ],
        ...     metadata={"retrieval_time": "2025-01-25", "query": "Paris"},
        ...     context_graph=None,
        ...     retrieval_summary="Key results about Paris include its status as the capital of France and iconic landmarks."
        ... )
        >>> print(output.modality)
        text
        >>> print(output.data[0]['content'])
        Paris is the capital of France.

    Design Principles:
        1. **Modality-Agnostic Structure**: 
           The `modality` and `data` attributes provide flexibility for representing 
           different types of retrieval results while maintaining a consistent interface.
        2. **Graph-Centric Flexibility**: 
           The optional `context_graph` attribute integrates seamlessly with graph-based 
           pipelines like SimGRAG and GraphRAG, allowing for context-aware processing and insights.
        3. **Rich Metadata and Summaries**: 
           The `metadata` and `retrieval_summary` attributes offer enhanced transparency, 
           traceability, and usability of retrieval results.
        4. **Extensibility**: 
           The design ensures minimal effort is required to add support for new data 
           modalities or extend attributes, adhering to the open-closed principle.

    Notes:
        - The `context_graph` attribute is optional and can be omitted when not relevant 
          to the retrieval results.
        - The `retrieval_summary` attribute is designed to provide a quick overview of 
          the retrieval results and can be generated dynamically if not explicitly provided.
        - The `data` attribute's structure should align with the `modality` specified 
          to ensure consistency across different retrieval plugins.

    Attributes:
        - `modality` (str): The type of data retrieved (e.g., "text", "image", "audio").
        - `data` (List[Dict[str, Any]]): The retrieval results in a list of dictionaries.
        - `metadata` (Dict[str, Any]): Additional metadata about the retrieval process.
        - `context_graph` (Optional[Any]): An optional graph structure providing context.
        - `retrieval_summary` (Optional[str]): An optional summary of the retrieval results.
    """
    modality: str  # e.g., "text", "image", "audio", etc.
    data: List[Dict[str, Any]]  # A list of data entries relevant to the query
    metadata: Dict[str, Any]  # Additional information about the retrieval (e.g., scores, provenance, etc.)
    context_graph: Optional[Any] = None  # NetworkX or similar graph structure (optional)
    retrieval_summary: Optional[str] = None  # Summarized description of the retrieval results



#---------------Information storage related-Begin-------------#

# Base class for vector database management. Source: MicrosoftGraphRAG
class BaseVectorStore(ABC):
    """The base class for vector storage data-access classes."""

    def __init__(
        self,
        collection_name: str,
        db_connection: Any | None = None,
        document_collection: Any | None = None,
        query_filter: Any | None = None,
        **kwargs: Any,
    ):
        self.collection_name = collection_name
        self.db_connection = db_connection
        self.document_collection = document_collection
        self.query_filter = query_filter
        self.kwargs = kwargs

    @abstractmethod
    def connect(self, **kwargs: Any) -> None:
        """Connect to vector storage."""

    @abstractmethod
    def load_documents(
        self, documents: list[Document], overwrite: bool = True
    ) -> None:
        """Load documents into the vector-store."""

    # @abstractmethod
    # def similarity_search_by_vector(
    #     self, query_embedding: list[float], k: int = 10, **kwargs: Any
    # ) -> list[VectorStoreSearchResult]:
    #     """Perform ANN search by vector."""

    # @abstractmethod
    # def similarity_search_by_text(
    #     self, text: str, text_embedder: TextEmbedder, k: int = 10, **kwargs: Any
    # ) -> list[VectorStoreSearchResult]:
    #     """Perform ANN search by text."""

    @abstractmethod
    def filter_by_id(self, include_ids: list[str] | list[int]) -> Any:
        """Build a query filter to filter documents by id."""

    @abstractmethod
    def search_by_id(self, id: str) -> Document:
        """Search for a document by id."""

from typing import Any
from abc import ABC, abstractmethod



class BaseDataFrame(ABC):
    """
    Abstract base class for dataframe operations.
    """

    @abstractmethod
    def head(self, n: int = 5):
        """Returns the first `n` rows of the dataframe."""
        pass

    @abstractmethod
    def query(self, query: str):
        """Filters the dataframe based on a condition."""
        pass


    @abstractmethod
    def get_backend(self) -> str:
        """Returns the backend name."""
        pass
    @abstractmethod
    def save(self, path: str, format: str = "csv", **kwargs):
        """Save the dataframe to the specified path."""
        pass

import pandas as pd


class PandasDataFrame(BaseDataFrame):
    def __init__(self, data, **kwargs):
        self.dataframe = pd.DataFrame(data, **kwargs)

    def head(self, n: int = 5):
        return self.dataframe.head(n)

    def query(self, condition: str):
        query_df = self.dataframe.query(condition)
        return PandasDataFrame(query_df)
    
    def filter(self,**kwargs):
        df = self.dataframe.filter(**kwargs)
        return PandasDataFrame(df)
    
    def get_backend(self) -> str:
        return "pandas"
    
    def save(self, path: str, format: str = "parquet", **kwargs):
        if format == "csv":
            self.dataframe.to_csv(path, **kwargs)
        elif format == "parquet":
            self.dataframe.to_parquet(path, **kwargs)
        elif format == "pickle":
            self.dataframe.to_pickle(path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

# TODO: integration with duck db

# import vaex

# class VaexDataFrame(BaseDataFrame):
#     def __init__(self, data, **kwargs):
#         if isinstance(data, str):
#             self.dataframe = vaex.open(data)
#         else:
#             self.dataframe = vaex.from_pandas(data)

#     def head(self, n: int = 5):
#         return self.dataframe.head(n).to_pandas_df()

#     def filter(self, condition: str):
#         filtered_df = self.dataframe.filter(condition)
#         return VaexDataFrame(filtered_df)

#     def to_csv(self, path: str, **kwargs):
#         self.dataframe.export_csv(path, **kwargs)

#     def get_backend(self) -> str:
#         return "vaex"

# import dask.dataframe as dd

# class DaskDataFrame(BaseDataFrame):
#     def __init__(self, data, npartitions: int = 1, **kwargs):
#         self.dataframe = dd.from_pandas(data, npartitions=npartitions)

#     def head(self, n: int = 5):
#         return self.dataframe.head(n, compute=True)

#     def filter(self, condition: str):
#         filtered_df = self.dataframe.query(condition)
#         return DaskDataFrame(filtered_df)

#     def to_csv(self, path: str, **kwargs):
#         self.dataframe.to_csv(path, single_file=True, **kwargs)

#     def get_backend(self) -> str:
#         return "dask"


def DataframeFactory(data, backend: str = "pandas", **kwargs):
    """
    Factory function to create an instance of a dataframe with the specified backend.
    """
    backend = backend.lower()
    if backend == "pandas":
        return PandasDataFrame(data, **kwargs)
    # elif backend == "vaex":
    #     return VaexDataFrame(data, **kwargs)
    # elif backend == "dask":
    #     return DaskDataFrame(data, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
#---------------Information storage related-End---------------#