import yaml
import importlib

class Sanchayam:
    """
    Sanchayam Storage Manager - Supports dynamic storage backends.
    
    This class provides a unified interface for interacting with different storage backends, 
    such as local file storage and MinIO object storage. The backend is dynamically loaded 
    based on the configuration provided in a YAML file.
    
    Attributes:
        config (dict): Loaded configuration settings.
        backend (StorageBackend): An instance of the selected storage backend.
    """

    def __init__(self, config_path: str):
        """
        Initialize the Sanchayam storage manager.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        
        Raises:
            ImportError: If the specified backend module or class cannot be loaded.
        """
        self._load_config(config_path)
        self._initialize_backend()

    def _load_config(self, config_path: str):
        """
        Load configuration settings from a YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        
        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def _initialize_backend(self):
        """
        Dynamically load and initialize the storage backend based on the configuration.
        
        Raises:
            ImportError: If the backend module or class cannot be loaded.
        """
        backend_name = self.config.get("storage_backend", "LocalStorage")
        storage_dir = self.config.get("storage_dir","storage")
        backend_module = f"Sanchayam.backends.{backend_name}"
        backend_class = backend_name  # Class name matches module name

        try:
            module = importlib.import_module(backend_module)
            self.backend = getattr(module, backend_class)(base_path=storage_dir)
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(f"Failed to load backend '{backend_name}': {e}")

    def file_exists(self, filename: str, storage_type: str) -> bool:
        """
        Check if a file exists in the storage backend.
        
        Args:
            filename (str): The name of the file.
            storage_type (str): The storage category (e.g., "artifacts", "data").
        
        Returns:
            bool: True if the file exists, False otherwise.
        """
        return self.backend.file_exists(filename, storage_type)

    def save_file(self, filename: str, data: bytes, storage_type: str):
        """
        Save a file to the storage backend.
        
        Args:
            filename (str): The name of the file.
            data (bytes): The file content as bytes.
            storage_type (str): The storage category (e.g., "artifacts", "data").
        """
        return self.backend.save_file(filename, data, storage_type)

    def read_file(self, filename: str, storage_type: str) -> bytes:
        """
        Read a file from the storage backend.
        
        Args:
            filename (str): The name of the file.
            storage_type (str): The storage category (e.g., "artifacts", "data").
        
        Returns:
            bytes: The content of the file.
        """
        return self.backend.read_file(filename, storage_type)

    def delete_file(self, filename: str, storage_type: str):
        """
        Delete a file from the storage backend.
        
        Args:
            filename (str): The name of the file.
            storage_type (str): The storage category (e.g., "artifacts", "data").
        """
        return self.backend.delete_file(filename, storage_type)
    
    def make_dir(self,dir:str):
        """
        Create a directory in the storage backend.
        
        Args:
            dir (str): The name of the directory to create.
        """
        return self.backend.make_dir(dir)
    
    def get_absolute_path(self, filename: str, storage_type: str) -> str:  
        """
        Get the absolute path of a file in the storage backend.
        
        Args:
            filename (str): The name of the file.
            storage_type (str): The storage category (e.g., "artifacts", "logs").
        
        Returns:
            str: The absolute path of the file.
        """     
        return self.backend.get_absolute_path(filename, storage_type)
       
     
#---------------Read yaml file example-Begin-------------#
if __name__ == "__main__1": 
    import os,yaml
    cwd = os.getcwd()
    config_path = f"{cwd}/config.yml"
    sanchayam = Sanchayam(config_path)
    file_out= sanchayam.read_file("low_level_graph.yml","artifacts/graph_extraction/low_level")
    dict_out = yaml.safe_load(file_out)
    print(dict_out)
#---------------Read yaml file example-End---------------#

#---------------Read pt file example-Begin-------------#
if __name__ == "__main__2":
    import os,yaml,torch,io
    cwd = os.getcwd()
    config_path = f"{cwd}/config.yml"
    sanchayam = Sanchayam(config_path)
    byte_buf_out = sanchayam.read_file("global_search_artifacts.pt","artifacts/graphRAG/community/low_level/")
    try:
        # Create a BytesIO object from the byte array
        byte_io = io.BytesIO(byte_buf_out)

        # Load the data using torch.load, passing the BytesIO object
        loaded_data = torch.load(byte_io)  # This will load your model or tensor

        # Now you can use the loaded data (e.g., your model)
        print(f"Loaded data:\n{loaded_data}")

    except Exception as e:
        print(f"Error loading PyTorch data: {e}")
#---------------Read pt file example-End---------------#