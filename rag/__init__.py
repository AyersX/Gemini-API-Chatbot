import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import json


_model = None
def getmodel():
  global _model
  if _model is None:
    from sentence_transformers import SentenceTransformer
    EMBED_MODEL:str = "paraphrase-multilingual-MiniLM-L12-v2"
    _model = SentenceTransformer(EMBED_MODEL)
  return _model


#FILEPATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#FOLDER PATH FOR READ
PARENT_DIR = os.path.dirname(BASE_DIR)
FOLDER_TARGET:str = "data"

#PROMPT/INSTRUCTIONS/LASTCHAT
PROMPTS:str = os.path.join(PARENT_DIR, FOLDER_TARGET, "prompts.txt")
LAST_CHAT:str= os.path.join(PARENT_DIR, FOLDER_TARGET, "last_chat.txt")

#DATA KNOWLEDGE & HISTORY
DATA_KNOWLEDGE:str = os.path.join(PARENT_DIR, FOLDER_TARGET, "knowledge.txt")
DATA_HISTORY:str = os.path.join(PARENT_DIR, FOLDER_TARGET, "history.txt")

#INDEX FOR KNOWLEDGE & HISTORY
KNOWLEDGE_IDX = os.path.join(PARENT_DIR, FOLDER_TARGET, "knowledge.bin")
HISTORY_IDX = os.path.join(PARENT_DIR, FOLDER_TARGET, "history.bin")