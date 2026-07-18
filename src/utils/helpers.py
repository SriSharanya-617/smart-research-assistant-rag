"""
General utility functions for calculations, string formatting, and estimates.
"""

def format_file_size(size_in_bytes: int) -> str:
    """
    Converts a size in bytes to a human-readable string (e.g. KB, MB).
    
    Args:
        size_in_bytes: Size in bytes.
        
    Returns:
        str: Human-readable string.
    """
    for unit in ['Bytes', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} TB"


def estimate_token_count(text: str) -> int:
    """
    Estimates the number of tokens in a string of text.
    Standard heuristic: ~4 characters per token for English text.
    
    Args:
        text: Input string.
        
    Returns:
        int: Estimated token count.
    """
    if not text:
        return 0
    # standard simple estimation
    return max(1, len(text) // 4)
