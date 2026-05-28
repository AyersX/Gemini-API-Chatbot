import faiss
import numpy as np
import json
from rag import getmodel, KNOWLEDGE_IDX, HISTORY_IDX
import os


def make_index_history(chunks:str):
  if os.path.getsize(HISTORY_IDX) > 0:
    index = faiss.read_index(HISTORY_IDX)

  else:
    print(f"\nEmbedding {len(chunks)} history chunks..\n")

    MODEL = getmodel()
    c_embedding = MODEL.encode(chunks, show_progress_bar=True)
    c_embedding = np.array(c_embedding).astype("float32")

    dimension:int = c_embedding.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(c_embedding)
    faiss.write_index(index, HISTORY_IDX)
  return index

def make_index_knowledge(chunks:str):
  if os.path.getsize(KNOWLEDGE_IDX) > 0:
    index = faiss.read_index(KNOWLEDGE_IDX)
  else:
    print(f"\nEmbedding {len(chunks)} knowledge chunks..\n")

    MODEL = getmodel()
    c_embedding = MODEL.encode(chunks, show_progress_bar=True)
    c_embedding = np.array(c_embedding).astype("float32")

    dimensions:int = c_embedding.shape[1]
    index = faiss.IndexFlatL2(dimensions)
    index.add(c_embedding)
    faiss.write_index(index, KNOWLEDGE_IDX)
  return index