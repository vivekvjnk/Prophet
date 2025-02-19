import importlib.util
import yaml
import os

def load_plugins(**kwargs):
    """Loads plugins from a directory based on a YAML configuration file.

    Returns:
        A dictionary where keys are plugin names and values are the loaded 
        plugin class objects. Returns an empty dictionary if any error occurs.
    """
    plugin_dir = os.path.dirname(__file__)  # Directory of the current module
    filepath = os.path.join(plugin_dir, "config.yml")

    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)  # Use safe_load to avoid arbitrary code execution
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error reading config file: {e}")
        return {}
    
    loaded_plugins = {}
    missing_plugins=[]
    loaded_modules = []

    plugins = config.get('plugins', {})  # 'plugins' key in YAML
    for plugin_name in plugins.keys():
        file_path = os.path.join(plugin_dir, f"{plugin_name}.py")  # Full file path
        if not os.path.exists(file_path):
            missing_plugins.append(plugin_name)
            print(f"Warning: Plugin file not found: {plugin_name}")
            continue
        try:
            module_name = f"Alchemist.retrieval_stratergies.{plugin_name}"  # Full module path
            # Dynamically load module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find the class within the module
                for name, obj in module.__dict__.items():
                    if isinstance(obj, type) and obj.__module__ == module.__name__:  # Check if it's a class from the module
                        loaded_plugins[plugin_name] = obj(**kwargs)  # Store the class object
                        break  # Stop after finding the first class
                else:  # if no class found
                    print(f"Warning: No class found in plugin file: {file_path}")
            else:
                print(f"Error: Failed to load module {plugin_name} (spec={spec})")

        except (ImportError, AttributeError, Exception) as e:
            print(f"Error loading plugin {plugin_name}: {e}")
            continue  # Skip
        loaded_modules.append(f"Alchemist.retrieval_stratergies.{plugin_name}")  
    output = {"plugins":loaded_plugins,"missing_plugins":missing_plugins}
    return output



#---------------Test code-Begin-------------#
if __name__ == "__main__":    
    loaded_plugins = load_plugins()

    if loaded_plugins:
        for name, plugin_class in loaded_plugins['plugins'].items():
            print(f"Loaded plugin: {name} ({plugin_class.__name__})")
            plugin_instance = plugin_class()  # Instantiate the plugin
    else:
        print("No plugins loaded.")
#---------------Test code-End---------------#
