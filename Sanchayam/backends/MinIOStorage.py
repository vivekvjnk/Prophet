import os
import io
import tempfile
from Sanchayam.StorageBackend import StorageBackend
from minio import Minio
from minio.error import S3Error


class MinIOStorage(StorageBackend):
    """
    MinIO-based object storage backend for Sanchayam.

    This backend interacts with a MinIO server to store and retrieve files.
    """

    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket_name: str):
        """
        Initialize the MinIO storage backend.

        Args:
            endpoint (str): MinIO server URL.
            access_key (str): Access key for authentication.
            secret_key (str): Secret key for authentication.
            bucket_name (str): Bucket name to store files.
        """
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)
        self.bucket_name = bucket_name

        # Ensure the bucket exists
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)

        # Dictionary to track temp files created for MinIO objects
        self.temp_files = {}

    def file_exists(self, filename: str, storage_type: str) -> bool:
        """
        Check if a file exists in MinIO storage.

        Args:
            filename (str): Name of the file.
            storage_type (str): Storage category (e.g., "artifacts", "data").

        Returns:
            bool: True if the file exists, False otherwise.
        """
        object_name = f"{storage_type}/{filename}"
        try:
            self.client.stat_object(self.bucket_name, object_name)
            return True
        except S3Error:
            return False

    def save_file(self, filename: str, data: bytes, storage_type: str):
        """
        Save a file to MinIO storage.

        Args:
            filename (str): Name of the file.
            data (bytes): File data in bytes.
            storage_type (str): Storage category (e.g., "artifacts", "data").
        """
        object_name = f"{storage_type}/{filename}"
        self.client.put_object(
            self.bucket_name,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type="application/octet-stream",
        )

    def read_file(self, filename: str, storage_type: str) -> bytes:
        """
        Read a file from MinIO storage.

        Args:
            filename (str): Name of the file.
            storage_type (str): Storage category (e.g., "artifacts", "data").

        Returns:
            bytes: File contents as bytes.
        """
        object_name = f"{storage_type}/{filename}"
        response = self.client.get_object(self.bucket_name, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def delete_file(self, filename: str, storage_type: str):
        """
        Delete a file from MinIO storage.

        Args:
            filename (str): Name of the file.
            storage_type (str): Storage category (e.g., "artifacts", "data").
        """
        object_name = f"{storage_type}/{filename}"
        self.client.remove_object(self.bucket_name, object_name)

    def get_absolute_path(self, filename: str, storage_type: str) -> str:
        """
        Get the absolute path of a file in MinIO storage (creates a local temporary copy).

        Args:
            filename (str): The name of the file.
            storage_type (str): The storage category (e.g., "artifacts", "logs").

        Returns:
            str: The absolute path of the local temporary file.

        Workflow:
        - Download the specified file from MinIO.
        - Create a temporary local file.
        - Return the absolute path to that file.
        - Track the file in `self.temp_files` for later cleanup.
        """
        object_name = f"{storage_type}/{filename}"

        # Create a temporary directory to hold the file
        temp_dir = tempfile.mkdtemp()
        local_file_path = os.path.join(temp_dir, filename)

        # Download file from MinIO and save it locally
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            with open(local_file_path, "wb") as f:
                for data in response.stream(32 * 1024):
                    f.write(data)
            response.close()
            response.release_conn()
        except S3Error as e:
            raise FileNotFoundError(f"File '{filename}' not found in MinIO: {e}")

        # Track the temporary file path to manage cleanup and updates
        self.temp_files[local_file_path] = (filename, storage_type)

        return local_file_path

    def complete_file_update(self, local_file_path: str):
        """
        Complete the file update process by uploading the modified file back to MinIO.

        Args:
            local_file_path (str): The absolute path to the temporary local file.

        Workflow:
        - Upload the file back to MinIO (with the original filename and storage type).
        - Remove the local temporary file.
        """
        if local_file_path not in self.temp_files:
            raise ValueError(f"File '{local_file_path}' is not tracked for MinIO updates.")

        filename, storage_type = self.temp_files[local_file_path]

        # Read the local file and upload it back to MinIO
        with open(local_file_path, "rb") as f:
            file_data = f.read()
            self.save_file(filename, file_data, storage_type)

        # Cleanup: Remove the local temporary file
        os.remove(local_file_path)
        del self.temp_files[local_file_path]

    def make_dir(self,dir):
        # Stub function 
        pass