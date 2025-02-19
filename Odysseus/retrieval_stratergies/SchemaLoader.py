import yaml
import jsonschema
from jsonschema import validate
from typing import Tuple, Any
import os

class SchemaLoader:
    """Singleton-style schema loader that caches the JSON schema for validation."""
    _instance = None
    _schema = None

    def __new__(cls, schema_path: str = "config.yml"):
        """Ensures a single instance of SchemaLoader is created."""
        if cls._instance is None:
            cls._instance = super(SchemaLoader, cls).__new__(cls)

            plugin_dir = os.path.dirname(__file__)  # Directory of the current module
            filepath = os.path.join(plugin_dir, "config.yml")
            cls._instance._load_schema(filepath)
        return cls._instance

    def _load_schema(self, schema_path: str):
        """Loads the JSON schema from the YAML file (only once)."""
        try:
            with open(schema_path, "r") as f:
                schema_data = yaml.safe_load(f)
            self._schema = schema_data.get("plugins", {})
            print("✅ Schema loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading schema: {e}")
            self._schema = {}

    def validate(self, data: Any,retrieval="GraphRAGGlobalSearch") -> Tuple[bool, str]:
        """Validates a given JSON object against the schema."""
        try:
            validate(instance=data, schema=self._schema[retrieval])
            return True, "Valid message"
        except jsonschema.exceptions.ValidationError as e:
            return False, str(e)

# # Global instance to avoid multiple schema loads
# schema_loader = SchemaLoader()
