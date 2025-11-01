"""Main entry point for LLM Evaluation Benchmark Viewer."""

import streamlit as st
import subprocess
import sys

from viewer.components.main_page import MainPage

# Configure Streamlit page
st.set_page_config(
    page_title="LLM Evaluation Benchmark Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def streamlit_app():
    """Streamlit app content."""
    main_page = MainPage()
    main_page.render()


def main():
    """Entry point for pip-installed command."""
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__] + sys.argv[1:])


if __name__ == "__main__":
    # When run directly via streamlit run, execute the app
    streamlit_app()
