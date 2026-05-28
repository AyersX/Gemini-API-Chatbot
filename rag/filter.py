import os
import json
import numpy as np
from rag import getmodel
from rag.index import make_index_history
from rag.retriever import make_chunk_history

def knowledge_trend(query, index, chunks, top_k=3) -> str:

  print("\nEmbedding for data..")
  MODEL = getmodel()
  query_vector = MODEL.encode([query])
  query_vector = np.array(query_vector).astype("float32")

  distances, indices = index.search(query_vector, top_k)

  results = []
  for idx in (indices[0]):
    if idx != -1:
      results.append(chunks[idx])
      topics = "\n".join(results)
  print("\nLooking for relevant topic in database knowledge..")
  # for i, data in enumerate(results, start=1):
  #   # print(f"Chat {i} {data[:50]}..")
  print(f"\nDistance {distances}\n")
  return topics


def history_trend(query, index, chunks, top_k=10) -> str:
  print("\nEmbedding for history..")
  MODEL = getmodel()
  query_vector = MODEL.encode([query])
  query_vector = np.array(query_vector).astype("float32")

  distances, indices = index.search(query_vector, top_k)

  results = []
  for idx in indices[0]:
    if idx != -1:
      results.append(chunks[idx])
      topics = "\n".join(results)
  print("\nLooking for relevant topic in history..")
  # for i, data in enumerate(results, start=1):
  #   # print(f"Chat {i} {data[:50]}..")
  print(f"\nDistance {distances}\n")
  return topics
#for debugging
