import os

# Get the absolute path to the root of the project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Construct full paths
datasets_path = os.path.join(PROJECT_ROOT, 'datasets')
features_path = os.path.join(PROJECT_ROOT, 'datasets', 'features')


# This is where the sentence encoder is stored
embedding_model_path = os.path.join(PROJECT_ROOT, 'source', 'models', 'paraphrase-mpnet-base-v2', 'models--sentence-transformers--paraphrase-mpnet-base-v2', 'snapshots', 'bef3689366be4ad4b62c8e1cec013639bea3c86a')

