from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import yaml

#---------------Retrieval stratergies-Begin-------------#
from .retrieval_stratergies.plugin import *
#---------------Retrieval stratergies-End---------------#

class RetrievalFactory:
    """Manages retrieval plugins."""
    
    def __init__(self,**kwargs):
        self.plugins = {}
        self.load_plugins(**kwargs)

    def register_plugin(self, name: str, plugin):
        """Register a new retrieval plugin."""
        if name not in self.plugins:
            self.plugins[name] = plugin
        else:
            raise(ValueError(f"Plugin name {name} already exists!!"))

    def invoke(self,strategy,**kwargs):
        """Use the specified retrieval plugin to fetch results."""

        if strategy not in self.plugins:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")   
        return self.plugins[strategy].invoke(**kwargs)
    
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
    data: Any  # A list of data entries relevant to the query
    # TODO: Enable following fields if necessary
    # metadata: Dict[str, Any]  # Additional information about the retrieval (e.g., scores, provenance, etc.)
    # context_graph: Optional[Any] = None  # NetworkX or similar graph structure (optional)
    # retrieval_summary: Optional[str] = None  # Summarized description of the retrieval results
