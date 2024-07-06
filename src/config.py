import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Data')
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Result')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Model')

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)