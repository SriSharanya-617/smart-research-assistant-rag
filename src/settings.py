"""
Streamlit page structure and style manager.
Handles the initialization of Streamlit settings, theme controls, and UI constants.
"""

import os
import streamlit as st
from src.constants import APP_ICON, APP_SUBTITLE, APP_TITLE, CSS_PATH, LOGO_PATH
from src.logger import setup_logger

logger = setup_logger("ui_settings")

def init_page_config() -> None:
    """
    Sets up the core Streamlit page parameters.
    Should be called as the very first Streamlit command in app.py.
    """
    try:
        st.set_page_config(
            page_title=f"{APP_TITLE} - {APP_SUBTITLE}",
            page_icon=APP_ICON,
            layout="wide",
            initial_sidebar_state="expanded"
        )
        logger.info("Streamlit page configuration set successfully.")
    except st.errors.StreamlitAPIException as e:
        logger.warning(f"Could not set page config (likely already set): {e}")

def load_custom_css() -> None:
    """
    Loads custom css styling rules from assets/styles.css and injects it into Streamlit page.
    """
    if os.path.exists(CSS_PATH):
        try:
            with open(CSS_PATH, "r", encoding="utf-8") as f:
                css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
            logger.info("Custom stylesheet assets/styles.css injected successfully.")
        except Exception as e:
            logger.error(f"Failed to inject custom stylesheet: {e}")
    else:
        logger.warning(f"Custom stylesheet not found at: {CSS_PATH}")

def display_app_header() -> None:
    """
    Renders the app header including logo and title.
    """
    col1, col2 = st.columns([0.1, 0.9])
    
    with col1:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=80)
        else:
            st.write(APP_ICON)
            
    with col2:
        st.title(APP_TITLE)
        st.markdown(f"*{APP_SUBTITLE}*")
