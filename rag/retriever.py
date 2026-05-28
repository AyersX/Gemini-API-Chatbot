from rag import DATA_KNOWLEDGE, DATA_HISTORY, KNOWLEDGE_IDX, HISTORY_IDX
from rag.index import make_index_knowledge, make_index_history
import faiss
import os


def make_chunk_knowledge() -> str:
    SEPARATOR = "<<<--->>>"
    if os.path.getsize(DATA_KNOWLEDGE) == 0:
      raise ValueError("KNOWLEDGE FILE IS EMPTY")
    else:
      with open(DATA_KNOWLEDGE, "r", encoding="utf-8") as file:
        data = file.read().strip()
    chunks:str = data.split(f"{SEPARATOR}")
    return chunks


def make_chunk_history() -> str:
    SEPARATOR = "<<<--->>>"
    with open(DATA_HISTORY, "r", encoding="utf-8") as file:
      data = file.read().strip()

    chunks:str = data.split(f"{SEPARATOR}")
    return chunks


def load_knowledge():
  if os.path.exists(KNOWLEDGE_IDX):
    chunks = make_chunk_knowledge()
    index = make_index_knowledge(chunks)
  else:
    raise FileNotFoundError("knowledge file is empty")
  return index, chunks



def load_history():
  if os.path.exists(HISTORY_IDX):
    chunks = make_chunk_history()
    index = make_index_history(chunks)
  else:
    raise FileNotFoundError("history file is empty")
  return index, chunks