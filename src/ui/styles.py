"""
Helper module for custom CSS styling injections.
"""

from src.settings import load_custom_css

def inject_global_styles() -> None:
    """
    Applies custom styles to the running Streamlit instance.
    """
    load_custom_css()
