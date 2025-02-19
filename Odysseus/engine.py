
import time

from infra.utils.utils import *

#---------------Pipeline specific imports-Begin-------------#
from .base import *
from .retrieval_stratergies.GraphRAGGlobalSearch import *
#---------------Pipeline specific imports-End---------------#
from Sanchayam.Sanchayam import Sanchayam

class Odysseus:
   def __init__(self,**kwargs):
      print(f"--Hello from Odysseus init\n")
   
      cwd = os.getcwd()
      config_path = f"{cwd}/config.yml"
      self.storage = Sanchayam(config_path=config_path)

      self.retriever:RetrievalFactory = RetrievalFactory(**kwargs,storage=self.storage) # This will load all the plugins


   def __call__(self, *args, **kwargs):
      self.start_server()
         
   def retrieve(self,query,**kwargs)->RetrievalOutput:
      results = self.retriever.retrieve(query,**kwargs)
      return results

   #---------------zeromq setup-Begin-------------#
   def handle_client(self,socket):
      """Handles client requests in a separate thread."""
      from .retrieval_stratergies.SchemaLoader import SchemaLoader
      schema_loder = SchemaLoader()
      while True:
         request_data = socket.recv_json()
         # Validate request against schema
         is_valid, error = schema_loder.validate(request_data,retrieval=request_data['retrieval_method'])

         if not is_valid:
            response = {"status": "error", "message": f"Invalid request: {error}"}
            socket.send_json(response)
            continue
         retrievals = self.retrieve(query=request_data['query'],strategy=request_data['retrieval_method'],source="low_level")
         response = {
            "query": request_data["query"],
            "retrieval_method": request_data["retrieval_method"],
            "retrievals": retrievals
            }
         socket.send_json(response)

   def start_server(self):
      import zmq
      import threading

      context = zmq.Context()
      socket = context.socket(zmq.REP)
      socket.bind("tcp://*:5555")
      
      print("✅ Odysseus is running on port 5555...")
      
      # Run server in a separate thread
      threading.Thread(target=self.handle_client, args=(socket,), daemon=True).start()

      while True:
         time.sleep(1)
         pass  # Keep main thread alive

   #---------------zeromq setup-End---------------#
   
if __name__ == "__main__":
   # odysseus = Odysseus(sources=["graphrag_paper"])
   odysseus = Odysseus(sources=["graphrag_paper","low_level","lttng_guide","tmw_dnp3_scl_um","tmw_sdg_um","lfs","advanced_linux_programming"])
   odysseus() # start odysseus server

if __name__ == "__main__1":
   odysseus = Odysseus()
   odysseus.load_stratergies(sources=["low_level"])
   results = odysseus.retrieve("Hello world",source="low_level")
   print("HEllo from main\n")
   print(results['data'][0]['summary'])

   
'''
# Major design decisions on Odysseus
## Objective of Odysseus 
- Odysseus serves as a modular retrieval engine within the Prophet pipeline, designed to operate on networkX graphs. Its primary objective is to enable flexible, efficient, and scalable information retrieval through a plugin-based architecture, supporting multiple retrieval mechanisms such as global search, local search, and drift search. By ensuring isolation, customizability, and seamless integration, Odysseus acts as the central orchestrator for retrieving meaningful insights from complex graph structures, while remaining adaptable to future retrieval strategies and evolving project needs.
## Plugin based design
1. **Independent Retrieval Plugins**:  
   Each retrieval mechanism (e.g., global search, local search, drift search) is implemented as a self-contained plugin, ensuring complete isolation. This allows each plugin to be developed, tested, and deployed independently of others.

2. **Standardized Interface**:  
   Each plugin adheres to a standardized interface with two primary methods:  
   - `__init__`: Handles initialization and preprocessing required for the retrieval.  
   - `__call__`: Executes the runtime retrieval logic and returns the relevant results.

3. **Flexible Input/Output Design**:  
   Plugins have the freedom to define their own input requirements and output formats while ensuring compatibility with Odysseus. This ensures that plugins can implement complex or specific retrieval logic without being constrained by a rigid structure.

4. **Seamless Integration**:  
   The architecture enables effortless integration of plugins into the Odysseus core. Adding a new retrieval mechanism is straightforward, requiring minimal changes to the core system.

5. **Modularity and Scalability**:  
   The plugin-based design makes Odysseus modular and highly scalable. Retrieval mechanisms can be extended or replaced without disrupting existing functionality.

6. **Dynamic Selection of Retrieval Strategies**:  
   Odysseus can dynamically select and execute specific plugins based on the query requirements, optimizing retrieval processes and ensuring tailored results.

7. **Future-Proof Design**:  
   The architecture is built to accommodate evolving needs, making it easy to introduce new retrieval methods or integrate with external systems while maintaining compatibility with the existing pipeline.
'''