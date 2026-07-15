import logging
import os
import sys

# Create the logger globally so it can be imported anywhere
logger = logging.getLogger("mlforgex")
logger.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# Console Handler (always active)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(stream_handler)

def add_file_handler(log_dir):
    """Adds a FileHandler to the logger, saving logs to the specified directory."""
    log_file = os.path.join(log_dir, "training_logs.log")
    
    # Check if a FileHandler for this exact file already exists
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(log_file):
            return # Already added

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
