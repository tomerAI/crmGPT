import json
import yaml
from langchain_core.tools import tool


@tool
def fetch_metadata_as_yaml() -> str:
    """Reads the catalog.json file and returns its contents as a YAML string."""
    try:
        # Define the path to the JSON file
        json_file_path = 'src/utilities/catalog.json'

        # Load the JSON data
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        # Convert JSON data to YAML
        yaml_data = yaml.dump(data, default_flow_style=False)

        return yaml_data
    except Exception as e:
        return str(e)
