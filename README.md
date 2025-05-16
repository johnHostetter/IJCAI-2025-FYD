# IJCAI-2025-FYD
A simplification method to create human-readable fuzzy logic rules independent of an intended learning paradigm.

# Requirements
Python 3.9.18
Tested on Ubuntu 22.04.3 LTS
Library dependencies available in `requirements.txt`

# Installation
1. Create a virtual environment:
   ```bash
   python -m venv env
   ```
2. Activate the virtual environment:

   _Windows_
   ```bash
   .\env\Scripts\activate
   ```
   _macOS/Linux_
   ```bash
   source env/bin/activate
   ```
3. Install the required libraries:
   ```bash
    pip install -r requirements.txt
    ```
4. Install the package:
   ```bash
   pip install -e .
   ```
5. Run the tests (which double as examples):
   ```bash
    pytest
    ```