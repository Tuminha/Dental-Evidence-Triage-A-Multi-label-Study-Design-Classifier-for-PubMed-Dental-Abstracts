"""
Small helpers. Keep implementation minimal; you'll expand in notebooks.
"""

import os
import time
import requests


def need_env(name, default=None):
    """
    Get an environment variable or raise an error if not found.
    
    Args:
        name: Environment variable name
        default: Default value if not found (if None, raises error)
        
    Returns:
        str: Environment variable value
        
    Raises:
        RuntimeError: If variable not found and no default provided
    """
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def eutils_get(endpoint, params, pause=0.34):
    """
    Thin wrapper for NCBI E-utilities GET with polite rate limit.
    
    Args:
        endpoint: E-utilities endpoint (e.g., 'esearch.fcgi', 'efetch.fcgi')
        params: Dictionary of query parameters
        pause: Time to wait after request (seconds) to respect rate limits
        
    Returns:
        requests.Response: HTTP response object
        
    Raises:
        requests.HTTPError: If request fails
    """
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{endpoint}"
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    time.sleep(pause)
    return r

