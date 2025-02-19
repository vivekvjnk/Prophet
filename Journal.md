# The Prophet 

## Date: 2025-02-02 : 14:24 (UTC)

### 1. Summary of Progress
- Conceptualization of GraphRAGLocalSearch.
- Moved Prophet out of research assistant repo. Now it is a submodule in research assitant.
- Updated Odysseus with proper plugin design pattern.

- Fine tuning Global search
    1. Final answer prompt : Present prompt doesn't give enough context to the LLM. We have to specify information about the knowledge graph and community detection mechanism in the prompt to clarify what a community means in this context. 

### 2. Detailed Updates
- Added files in Odysseus and Alchemist related to LocalSearch.
- Added configuration fies for Odysseus to conform with plugin design pattern

### 3. Challenges & Open Questions
- None related to above updates right now

### 4. Next Steps
- Focus on GlobalSearch pipeline. Further optimization of prompts are required to generate good quality answer
- Updates are required in Alchemist implementation of GlobalSearch

## Date: 2025-02-03 : 23:36 (UTC)

### 1. Summary of Progress
- Tested bodhi pipeline with TMW DNP3 Outstation document
- System failed to parse code


### 2. Detailed Updates
- The document contain lot of code segments. 
    - Length of each code segment is much higher than the context limit
    - This resulted in poor code parsing 
- Started searching for methods to parse and build graphs for code segments effectively
- Found Joern, Graph4Code, Kythe
- Out of these Joern seems promising
- This library is used for vulnerability detection in code bases. (static code analyzer)
- Supports most of the programming languages
- Joern converts AST,Program Dependence graph and Flow graph into a comprehensive Code Property Graph (CPG)
- Also provides a prompt interface with specific query language

### 3. Challenges & Open Questions
- Code splitting from source document

### 4. Next Steps
- Update Bodhi 
    - First stage of Bodhi needs to be updated. Text chunking algorithm should identify code sections from source document.
    - Then these code sections are stored in a different file with meta data about the location of section in the source. Also a unique identifier would be attached to the code section. This will help to link the source document with the extracted code 
    - Text chunk which holds the code would be updated with a summary about the code and the unique identifier assigned to extracted code segment
- Extracted code sections will be attached to a CPG graph using Joern. 

## Date: 2025-02-04 : 20:08 (UTC)

### 1. Summary of Progress
- Decide to postpone code segmentation pipeline to a later timeline
    - This will be addressed during Joern integration with the system. Both concepts are closely related
    - Choice of code segmentation now would delay the GraphRAG pipeline implementation
    - Now we need to implement the LocalSearch and DriftSearch algorithms within 13 days
    - For now we choose to avoid big code segments altogether from graph generation pipeline

- Started work on LocalSearch 
    

### 2. Detailed Updates
- Local search
    - Vector DB : ChromaDB 
        - Simple, scalable and fast compared to milvus
    

### 3. Challenges & Open Questions
- None

### 4. Next Steps
- Setup chromadb docker 
- Build graph for Local search odysseus pipeline


## Date: 2025-02-05 : 07:36 (UTC)

### 1. Summary of Progress
- Bodhi pipeline update
- Odysseus GraphRAGGlobalSearch update
### 2. Detailed Updates
- Added functionality to continue entity/relationship extraction from intermediate&chunks files in Bodhi
- Proceeding with community summarization for the user manual document
- Repalced inconsitent missing node list calculation algorithm in Odysseus:GraphRAGGlobalSearch with more robust mechanisms

#### Design of vector db collections (Design) 
We need seperate collections for Local search vector retrieval pipeline. Following are the potential collections 
1. Entity collection
    - This contains all the entities present in the graph
    - Entities should be grouped based on the generated communities
        - That is each Document may contain multiple entities. Hence combining random entities to generate document based on context limit doesn't make sense. 
2. Relationship vector store 
    - Same rationale for entities applies here also
    - Group relationships based on communities.
    - Split relationships in the communities into documents of fixed token count 
3. Text unit/chunk Collection
    - Directly store each text chunk into this store
    - Each chunk/unit becomes a document
4. Community retport vector store
    - Each community report is a document 
- Started experiments with chromaDB

### 3. Challenges & Open Questions
- ChromaDB setup 
    - Familiarization with chromadb 


### 4. Next Steps
- Bodhi persistence updates
    - Add unique identifier to each text chunks in the document
    - Use this unique identifier during graph extraction to backtrack the source

- Odysseus GraphRAG persistence update
    - Need to add persistence for GraphRAG node summarization 


## Date: 2025-02-06 : 09:11 (UTC)

### 1. Summary of Progress
- Found 1 interesting bug related to node summarization in Odysseus:GraphRAG
- Added rough persistence for node summarization in Odysseus:GraphRAG
### 2. Detailed Updates
#### Odysseus: GraphRAG Updates
- Sometimes the generated node names may have potentially multiple nodes seperated by comma. LLM would identify them as independent nodes during summarization and prepare individual summaries for them. This results in inconsistent summary dictionary keys with respect to the graph keys. This lead to infinite loop in the node summarization. 
    - Updated single node summarization prompt
    - added logic to check if single node is present in the node dictionary or not. This logic would act as a last stand defence against these kind of unpredictable issues. Since we do iterative summarization of nodes with missing nodes in each iteration, this logic at the least would pin down to the node which causes issues. 
    - Also added punctuations to node names so that llm can identify individual node names as a whole. 
#### ChromaDB updates 
- Design of document storage interface
    - Given a collection name and text chunk, the interface should prepare a document and store the document in the specified collection in chromaDB
    - Token count related decisions should be taken from top level. Since ChromaDB doesn't explicitly impose any limitations on the size of individual documents, adding functionality to split documents based on token count in this api doesn't feel like a good choice. 
    - The interface should return some unique identifier to the prepared document 

#### Odysseus: Persistence on community summarization pipeline
- Implemented persistence for community summarization 
- Tested pipeline with 'low_level' document


### 3. Challenges & Open Questions
- Odysseus:Local search implementation - Still pending 
- For a document with roughly 46k characters community summarization takes around 10hrs to complete

### 4. Next Steps
- Neo4j graph database integration with Bodhi 


## Date: 2025-02-07 : 19:05 (UTC)

### 1. Summary of Progress
- Started work on chromaDB implementation for LocalSearch
- Testing one GlobalSearch in progress

### 2. Detailed Updates
#### GraphRAG Local Search
- Tried to configure chromaDB server in docker and establish connection through python script
    - Faced issues related to authentication, ports etc.
    - Hence parked the activity for now and decided to proceed with local storage of db.
- It's observed that loading stored db and creating new db takes around the same time : 9 seconds 
- Implemented and tested collection preparation and storage interfaces for GraphRAG global search

#### Odysseus GraphRAG Global Search Community summarization 
- Successfully summarized document with 325 communities (with persistence) (took around 10 hrs for the complete Odysseus pipeline)

### 3. Challenges & Open Questions


### 4. Next Steps
- Improvements on logging. Right now logging is not implemented properly

## Date: 2025-02-08 : 14:18 (UTC)

### 1. Summary of Progress
- Progressing with Local search
- Global search test is ongoing...
### 2. Detailed Updates
#### GraphRAG Local Search
- ChromaDB doesn't support vector search on document id 
    - Document ids are treated as simple unique identifiers with no semantic meaning
- Collection creation for entities, relationships, community summaries and text units completed
- Basic retrieval for entities completed
    - Whenever we get a user query, we search entity name collection and entity description collection to find matches 
    - If there are repeated matches from both collections, we use harmonic mean on their distances to normalize the distance, then retain the match from entity name collection with updated distance. 


#### GraphRAG Global Search : tests
TMW SDG User Manual
- Took 5hrs for Graph Extraction : Bodhi pipeline 
- Almost 12.5hrs for community summarization : Odysseus pipeline
- Failed 2 times during Odysseus. Found few bugs and solved.

### 3. Challenges & Open Questions


### 4. Next Steps


## Date: 2025-02-09 : 08:29 (UTC)

### 1. Summary of Progress
#### Global search
- Testing with linux from scratch book

#### local search 
- Started conceptualization&implementation of further parts of the pipeline 

### 2. Detailed Updates
1. Graph for local search implemented 
2. Completed entity retrieval with grading
3. Conceptualized relationship retrieval based on retrieved entities 

### 3. Challenges & Open Questions
#### GPU Bottlenecks 
- We've started seeing real GPU bottlenecks for the first time 
    - We're using one instance of ollama for Global search testing
    - Meantime, we develop local search : in this we have few llm calls
    - We can't do both using one instance of ollama. The model we use consumes almost all the GPU
- For now we solved the issue. ollama allows to set host ip address
    - We created a langchain runnable wrapper around ollama client interface
    - Now we are capable of using any number of ollama instances available in the local network

### 4. Next Steps
- Use retrieved entities to retrieve relevant communities, relationships and text units
- Decision on grading for the retrieved information


## Date: 2025-02-10 : 11:54 (UTC)

### 1. Summary of Progress
- LocalSearch pipeline : implemented most of the retrieval algorithms

### 2. Detailed Updates
#### Design of retrieve relationships <-> entities loop (GraphRAGLocalSearch) 
- Once we have primary set of entities, we  can extract set of relationships with respect to this primary set of entities 
- Then using these secondary set of entities, we can do relationship extraction again. This way we can collect all the connected nodes to the primary set of entities 
- The loop should be able to collect n level of entities and relationships, where n is a configurable parameter
Hence the state variable for the entities and relationships should be a dictionary of varying size 

entities: {<level_#>:[<list of entities at level_#>],}
relationships: {<level_#>:[<dictionary of relationships at level_#>],}
- Loop is working now. We have provision to set the depth(level) to which relationship/entity extraction should go.

- Implemented community summary retrieval and text unit retrieval also

### 3. Challenges & Open Questions


### 4. Next Steps
- Integrate entities, relationships, community summaries and text unit to generate a cohesive answer (Alchemist part of local search)
- Start working on articles
- Clean up, fine tuning and integration of all the pipelines (Bodhi, Odysseus and Alchemist)

### 5. Perls
#### Conceptualization of intuition maker (Source: SKA)
- System which is capable of extracting intuitions from HindSight 
1. Intuition making from chat history 
    - Take example of AlphaCodium code generation
- Bodhi pipeline
    - Log all the llm prompts, results and errors for Alpha Codium pipeline
    - Build graph using this information
- Workflow of the intuition maker
    - Start from the working solution
    - Iterate backward to the original problem through all the branches which lead to the solution 
    - Iterate through all the failed branches and identify reasons for failure
    - Distill failure reasons and lessons
    - Identify any intuitions applied in the successful branch which are derrived from any of the failed branches

## Date: 2025-02-11 : 07:53 (UTC)

### 1. Summary of Progress
- Implemented map-reduce pattern for retrieval in localsearch


### 2. Detailed Updates
- We had to add a dummy node after the get_relationships node 
    - Langgraph design constraint
    - Since we have a loop between get_relationships and get_entities, this will require multiple supersteps to complete
    - get_relationships, retrieve_text_units, get_communities all work in parallel. Hence to enable synchronism, after get_relationships we added a dummy node and synchronized dummy node with other parallel nodes

### 3. Challenges & Open Questions


### 4. Next Steps
- Integration of server and schema with odysseus:localsearch

## Date: 2025-02-11 : 19:18 (UTC)

### 1. Summary of Progress
#### Bodhi
- Found context length related potential issue with Bodhi pipeline
#### LocalSearch
- Odysseus and Alchemist basic level implementation : working

### 2. Detailed Updates
#### Bodhi
- If a text unit contain too many possible relationships and entities, the extraction loop would fail with small LLM models
- Surfaced during relationship extraction of "advanced_linux_programming". This book contain lot of code with explanations. While the model is working on some section related to race condition, the relationship extraction loop went into infinite loop even though limit conditions were clearly specified.
#### LocalSearch 
- Successfully retrieved information from vector databases
- Integrated basic plugins for LocalSearch in Odysseus as well as Alchemist

### 3. Challenges & Open Questions
#### Bodhi
- Issue found on Bodhi pipeline is still open. For now we solved it with reducing the token count for entity extraction

### 4. Next Steps

## Date: 2025-02-12 : 22:09 (UTC)

### 1. Summary of Progress
- Alchemist:LocalSearch basic implementation - Completed

### 2. Detailed Updates


### 3. Challenges & Open Questions


### 4. Next Steps




## Date: 2025-02-13 : 16:43 (UTC)

### 1. Summary of Progress
- Tested LocalSearch for few questions 
- Working as expected.. needs fine tuning

### 2. Detailed Updates


### 3. Challenges & Open Questions
- As of now odysseus:localsearch would run only once. After the graph breaks due to parallel execution. Need to debug. 
- odysseus:globalsearch doesn't have this problem 

### 4. Next Steps


## Date: 2025-02-14 : 13:59 (UTC)

### 1. Summary of Progress
- Solved Odysseus:LocalSearch failure after single user query

### 2. Detailed Updates
- Odysseus:LocalSearch was failing after an execution cycle. 
    - We were using same langgraph graph for the next cycle. This resulted in reusing the state from the last execution
    - Solution is to reset the graph for every user query

### 3. Challenges & Open Questions
- Thread management : Need further exploration on persistence, thread management and concept of checkpoints in langgraph 


### 4. Next Steps

## Date: 2025-02-16 : 21:41 (UTC)

### 1. Summary of Progress
- Conceptualization of Sanchayam : The storage module

### 2. Detailed Updates
#### Sanchayam
- Wrapper over data storage types
- User can specify Object data store or local file system for storage

### 3. Challenges & Open Questions


### 4. Next Steps


## Date: 2025-02-16 : 21:41 (UTC)

### 1. Summary of Progress
- Integrated Sanchayam with Bodhi

### 2. Detailed Updates
- Replaced all file operations in Bodhi with Sanchayam
- Verified the workflow few times

### 3. Challenges & Open Questions
- Some python libraries require file system paths. 
    - In Odysseus.LocalSearch we need to provide filesystem path for vector databases if we use simple PersistentClient. 
    - When the user chooses Object data stores like MinIO, we need to provide a local filesystem path. Then once all the updates/process are complete, it should be synced with the data store

### 4. Next Steps

## Date: 2025-02-17 : 08:15 (UTC)

### 1. Summary of Progress
- Completed integration of Sanchayam with Odysseus

### 2. Detailed Updates
- Channelled all file operations through Sanchayam
- Tested and verified artifact generation, vector database creation etc.

### 3. Challenges & Open Questions
- Preparing for the first release

### 4. Next Steps


## Date: 2025-02-19 : 22:53 (UTC)

### 1. Summary of Progress
- Releasing the first version open source under MIT license

### 2. Detailed Updates


### 3. Challenges & Open Questions


### 4. Next Steps

