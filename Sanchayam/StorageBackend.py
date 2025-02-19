from abc import ABC, abstractmethod

class StorageBackend(ABC):
    """
    Abstract base class for all Sanchayam storage backends.

    This class defines a standard interface for different storage backends, 
    ensuring that all implementations provide methods for file operations such as 
    checking existence, saving, reading, and deleting files.

    Subclasses must implement all abstract methods to support specific storage systems, 
    such as local filesystem, AWS S3, MinIO, or other object storage services.
    """

    @abstractmethod
    def file_exists(self, filename: str, storage_type: str) -> bool:
        """
        Check if a file exists in the storage backend.

        Args:
            filename (str): The name of the file to check.
            storage_type (str): The storage category (e.g., "artifacts", "logs").

        Returns:
            bool: True if the file exists, False otherwise.
        """
        pass

    @abstractmethod
    def save_file(self, filename: str, data: bytes, storage_type: str):
        """
        Save a file to the storage backend.

        Args:
            filename (str): The name of the file to save.
            data (bytes): The file content as a byte stream.
            storage_type (str): The storage category (e.g., "artifacts", "logs").

        Raises:
            Exception: If the file cannot be saved due to storage issues.
        """
        pass

    @abstractmethod
    def read_file(self, filename: str, storage_type: str) -> bytes:
        """
        Read a file from the storage backend.

        Args:
            filename (str): The name of the file to read.
            storage_type (str): The storage category (e.g., "artifacts", "logs").

        Returns:
            bytes: The file content as a byte stream.

        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: If an error occurs while reading the file.
        """
        pass

    @abstractmethod
    def delete_file(self, filename: str, storage_type: str):
        """
        Delete a file from the storage backend.

        Args:
            filename (str): The name of the file to delete.
            storage_type (str): The storage category (e.g., "artifacts", "logs").

        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: If the file cannot be deleted due to storage issues.
        """
        pass
    @abstractmethod
    def make_dir(self,directory_path)->str:
        pass
    
    @abstractmethod
    def get_absolute_path(self, filename: str, storage_type: str) -> str:
        """
        Get the absolute path of a file in the storage backend.

        Args:
            filename (str): The name of the file.
            storage_type (str): The storage category (e.g., "artifacts", "logs").

        Returns:
            str: The absolute path where the file is stored.
        
        Raises:
            NotImplementedError: If the backend does not support direct file access.
        """
        pass
