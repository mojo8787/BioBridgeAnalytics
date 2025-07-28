import app  # noqa: F401

# Streamlit Cloud looks for `streamlit_app.py` by default. This thin wrapper simply
# imports the main `app.py`, which contains the full BioData Explorer application.
# The import executes all Streamlit commands defined in `app.py`, launching the
# application automatically when deployed. 