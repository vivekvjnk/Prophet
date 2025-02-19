import os
from Sanchayam.StorageBackend import StorageBackend

class LocalStorage(StorageBackend):
    """
    Local filesystem-based storage backend for Sanchayam.

    This backend saves files to local directories and provides methods
    to check existence, read, write, and delete files.
    """

    def __init__(self, base_path: str = "storage"):
        """
        Initialize the LocalStorage backend.

        Args:
            base_path (str): The root directory for storing files.
        """
        self.base_path = os.path.abspath(base_path)
        os.makedirs(self.base_path, exist_ok=True)

    def _get_full_path(self,filename: str=None, storage_type: str=None) -> str:
        """
        Get the full path of a file based on its storage type.

        Args:
            filename (str): Name of the file.
            storage_type (str): Subdirectory for storing the file.

        Returns:
            str: Full absolute file path.
        """
        dir_path = os.path.join(self.base_path, storage_type)
        os.makedirs(dir_path, exist_ok=True)
        if filename:
            return os.path.join(dir_path, filename)
        else:
            return dir_path
        
    def file_exists(self, filename: str, storage_type: str) -> bool:
        """
        Check if a file exists in the local storage.

        Args:
            filename (str): Name of the file.
            storage_type (str): Storage category (e.g., "artifacts", "data").

        Returns:
            bool: True if the file exists, False otherwise.
        """
        return os.path.isfile(self._get_full_path(filename, storage_type))

    def save_file(self, filename: str, data: bytes, storage_type: str):
        """
        Save a file to the local storage.

        Args:
            filename (str): Name of the file.
            data (bytes): File data in bytes.
            storage_type (str): Storage category (e.g., "artifacts", "data").
        """
        with open(self._get_full_path(filename, storage_type), "wb") as file:
            file.write(data)

    def read_file(self, filename: str, storage_type: str) -> bytes:
        """
        Read a file from local storage.

        Args:
            filename (str): Name of the file.
            storage_type (str): Storage category (e.g., "artifacts", "data").

        Returns:
            bytes: File contents as bytes.
        """
        with open(self._get_full_path(filename, storage_type), "rb") as file:
            return file.read()

    def delete_file(self, filename: str, storage_type: str):
        """
        Delete a file from local storage.

        Args:
            filename (str): Name of the file.
            storage_type (str): Storage category (e.g., "artifacts", "data").
        """
        file_path = self._get_full_path(filename, storage_type)
        if os.path.isfile(file_path):
            os.remove(file_path)

    
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
        return self._get_full_path(filename=filename,storage_type=storage_type)
    
    def make_dir(self,dir):
        os.makedirs(dir, exist_ok=True)
        return dir