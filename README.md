# Prophet: AI-driven Knowledge Graph System

Prophet is a modular framework for building and experimenting with knowledge graph-based RAG (Retrieval-Augmented Generation) systems. It enables structured knowledge extraction, retrieval, and reasoning using graph-based techniques.

## Modules Overview

### 1. Bodhi - Knowledge Graph Extraction
Bodhi is a dedicated LLM based Knowledge Graph Extraction Pipeline for constructing knowledge graphs from unstructured documents. It supports various formats, including:
- PDFs
- Text files (.txt, .md, .csv, etc.)

Bodhi extracts entities, relationships, and constructs knowledge graphs in NetworkX format, serving as the foundation for downstream retrieval and inference.

### 2. Odysseus - Static Graph Retrieval Engine
Odysseus is a plugin-based graph retrieval engine designed for preprocessing and indexing knowledge graphs. It currently supports:
- **GraphRAG: Global Search** - Summarizes entire knowledge graphs for broad query coverage.
- **GraphRAG: Local Search** - Focuses on retrieving highly relevant graph substructures for detailed analysis.

Odysseus precomputes retrieval structures, optimizing runtime efficiency for downstream queries.

### 3. Alchemist - Runtime Graph Retrieval Engine
Alchemist serves as the dynamic counterpart to Odysseus, executing retrieval queries in real-time. Like Odysseus, it supports:
- **Global Search** - Retrieves broad contextual information.
- **Local Search** - Extracts specific, high-relevance subgraphs.

Alchemist ensures flexible and adaptive retrieval, integrating various retrieval strategies to enhance response accuracy.

### 4. Sanchayam - Storage Management Module
Sanchayam is a plugin-based storage manager enabling seamless integration of both object storage and file system-based storage solutions. It acts as the central data store for Prophet, ensuring efficient access and persistence of extracted knowledge graphs and retrieval artifacts.

## Environment setup 
This project is built with Anaconda. To replicate the conda environment run the following command (NB: Anaconda installation is required)

```bash
conda env create -f environment.yml
```

After activating the Conda environment, install the latest PyTorch with CUDA 12.4 manually: 

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124 
```

## Basic configurations
### config.yml file
As of now, config.yml file allows to following configurations:
1. LLM to be used in the pipeline
2. Maximum size limit for each text unit extracted from the source document
3. Storage backend : Prophet supports both object storage and file system storage. 
   - Defaults to local file system storage. Uses python os libraries in the backend. 
   - Object storage backend is under development
4. Storage directory : Custom storage directory path for saving the pipeline artifacts 

## Steps to run the system 
### Document processing 
- To prepare knowledge graph from new documents, store the document in `infra/data/` directory 
- Update the main section of the `Prophet.py` with the new file name 
example:
```python
    init_state = {"source_path":"<langgraph_application_structure.md>"} # Source name with extension
    ...
    odysseus = Odysseus(sources=["langgraph_application_structure"]) # Source name with without extension
```
- Then run the Prophet.py
```bash
python Prophet.py
```
- This will prepare knowledge graph and related vector databases for the given document. After all the static processing pipeline, zeromq based server will be initiated and start listening to port 5555. The Alchemist engine would connect with this server during runtime.

### Document querying 
- Make sure Odysseus server is up
- Update the `Alchemist/engine.py` file with the query to be answered
```python
...
state = {"query":"Explain concept of checkpointers?"} 
...
```
- Then run the Alchemist engine (Should run as python module)
```bash
python -m Alchemist.engine
```

## Roadmap
- Support for multiple retrieval strategies beyond GraphRAG
- MinIO storage backend integrations within Sanchayam
- Implement additional knowledge graph extraction techniques

## License
Prophet is open-source and licensed under [MIT License](LICENSE).
