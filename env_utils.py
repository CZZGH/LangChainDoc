import os
from dotenv import load_dotenv
load_dotenv()

def get_env_variable(var_name: str, default: str = None) -> str:
    """Retrieve an environment variable or return a default value."""
    return os.getenv(var_name, default)