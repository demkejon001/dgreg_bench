import os
import importlib

# Get the directory of the current module (__init__.py)
current_dir = os.path.dirname(__file__)

# List all files in the directory
for filename in os.listdir(current_dir):
    # Only consider Python files (ending in .py) and exclude __init__.py
    if filename.endswith('.py') and filename != '__init__.py':
        module_name = filename[:-3]  # Remove the '.py' extension
        # Dynamically import the module
        importlib.import_module(f'.{module_name}', package=__name__)

from reg_datasets.dataset_registry import DATASETS