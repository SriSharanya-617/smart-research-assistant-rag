"""
Text pre-processing and cleaning utilities.
"""

import re
import unicodedata
from src.logger import setup_logger

logger = setup_logger("text_preprocessing")

class TextPreprocessor:
    """
    Cleans and normalizes extracted text before splitting it into chunks.
    Ensures optimal text quality for generation models.
    """
    
    def __init__(self):
        # Match control characters except tabs (\t) and newlines (\n)
        self.control_chars_re = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
        
        # Match multiple consecutive spaces
        self.spaces_re = re.compile(r'[ \t\r\f\v]+')
        
        # Match 3 or more newlines to reduce to 2 newlines (preserves paragraph splits)
        self.newlines_re = re.compile(r'\n{3,}')

    def clean_text(self, text: str) -> str:
        """
        Runs the full text pre-processing and cleaning pipeline.
        
        Args:
            text: Raw input text.
            
        Returns:
            str: Standardized, clean text.
        """
        if not text:
            return ""
            
        logger.debug("Applying text cleaning and normalization.")
        
        # 1. Normalize Unicode (NFKC handles compatible character conversions)
        cleaned = unicodedata.normalize("NFKC", text)
        
        # 2. Remove invisible control characters (preserve standard tab/newline)
        cleaned = self.control_chars_re.sub('', cleaned)
        
        # 3. Normalize whitespace (merge repeated spaces/tabs, strip margins)
        cleaned = self.spaces_re.sub(' ', cleaned)
        
        # 4. Remove repeated blank lines (keep at most 2 newlines to preserve paragraphs)
        cleaned = self.newlines_re.sub('\n\n', cleaned)
        
        # 5. Clean leading and trailing whitespace per line
        cleaned = "\n".join(line.strip() for line in cleaned.splitlines())
        
        # 6. Final trim of the overall string
        cleaned = cleaned.strip()
        
        logger.debug(f"Preprocessing completed. Length changed from {len(text)} to {len(cleaned)} characters.")
        return cleaned
