import os

# ============================================================
# Configuration — single source of truth for all run scripts
# ============================================================

# --- Grover search parameters ---
N_QUBITS = 3                 # Number of qubits
TARGET_STATE = '101'         # Bitstring to search for
SHOTS = 4096                 # Number of measurement shots

# --- IBM Quantum credentials ---
# Set these as environment variables (or in a .env file) to keep them
# out of version control. Get your token from:
#   https://quantum.ibm.com/ -> Account -> API token
IBM_CHANNEL = os.getenv('IBM_CHANNEL', 'ibm_quantum')
IBM_TOKEN  = os.getenv('IBM_TOKEN', '')
IBM_INSTANCE = os.getenv('IBM_INSTANCE', 'ibm-q/open/main')
IBM_BACKEND = os.getenv('IBM_BACKEND', 'ibm_brisbane')
