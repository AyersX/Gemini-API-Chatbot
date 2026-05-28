from rag import DATA_HISTORY, LAST_CHAT
from rag.index import make_index_history
from rag.retriever import make_chunk_history
from rag import PROMPTS
from datetime import datetime
from zoneinfo import ZoneInfo
import re
import os


def getprompt() -> str:
  with open(PROMPTS, "r", encoding="utf-8") as file:
    prompts = file.read()

    current_time = datetime.now(
      ZoneInfo("Asia/Jakarta")
    ).strftime("%A, %d-%m-%Y %H:%M WIB")

    time_template = prompts.format(
    CURRENT_TIME=current_time)
    return time_template


def save(query, ai_response) -> None:
  SEPARATOR = "<<<--->>>"

  with open(DATA_HISTORY, "a", encoding="utf-8") as file:
    chunks:int = file.write(f'\n\n\n\n{SEPARATOR}\n\n\n\nuser: "{query}"\n\nai: "{ai_response}"')

  with open(LAST_CHAT, "w", encoding="utf-8") as file:
    chat = file.write(f'user: "{query}"\n\nai: "{ai_response}"\n\n\n\n{SEPARATOR}')

  chunks = make_chunk_history()
  make_index_history(chunks)
  print(f"\n{'saved successfully':^55}")


def lastchat() -> str:
  if not os.path.exists(LAST_CHAT) and os.path.getsize(LAST_CHAT) == 0:
    raise FileNotFoundError("lastchat file is empty/not found")
  else:
    with open(LAST_CHAT, "r", encoding="utf-8") as file:
      chat:str= file.read().strip()
  return chat


def clean(text):
  text = re.sub(r'\*+', '', text)
  text = re.sub(r'#+', '', text)
  text = re.sub(r'`+', '', text)
  text = re.sub(r' +', ' ', text)
  return text.strip()