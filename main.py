from rag.retriever import load_knowledge, load_history
from rag.filter import knowledge_trend, history_trend
from rag.util import clean, save, lastchat, getprompt
from google import genai
from dotenv import load_dotenv
from google.genai import types
import os
import time



def main():
  #move it to the loop if u update the knowledge
  index_knowledge, chunks_knowledge = load_knowledge()

  load_dotenv()
  API_KEY = os.getenv("API_KEY")
  if not API_KEY:
    raise ValueError("API_KEY not found in .env")


  while True:

    print(f"{'='*98:^80}")
    print(f"{'Ai Assistant By AyersX':^80}")

    index_history, chunks_history = load_history()
    LATEST:str = lastchat()

    query = input("Query: ").strip()
    if query.lower() == "out":
      break

    knowledge:str = history_trend(query, index_knowledge, chunks_knowledge)

    history:str = history_trend(query, index_history, chunks_history)
    if LATEST and LATEST in history:
      history:str = history.replace(LATEST, "").strip()

    ALL_PROMPTS =f"""\n\n\n
[USER INFORMATION]\n
{knowledge}\n\n\n
[LATEST CONVERSATION]\n
{LATEST}\n\n\n
[RELEVANT CONVERSATIONS]\n
{history}\n\n\n
[USER]\n
{query}\n\n\n""".strip()

    INSTRUCTIONS = getprompt()
    client = genai.Client(api_key=API_KEY)
    ai_config = types.GenerateContentConfig(
      system_instruction=INSTRUCTIONS,
      max_output_tokens=800,
      temperature=0
  )

    response = client.models.generate_content(
      model="gemini-3.1-flash-lite",
      contents=ALL_PROMPTS,
      config=ai_config
  )


    #optional
    print("-"*98)
    print(f"Token input: {response.usage_metadata.prompt_token_count}")
    print(f"Token output: {response.usage_metadata.candidates_token_count}")
    print(f"Total token: {response.usage_metadata.total_token_count}\n")

    #see the prompts for debugging (optional)
    #Print(ALLL_PROMPTS)

    print("thinking..")
    time.sleep(2)
    print(f"{'='*98:^80}")
    print(f"\n{'THE ANSWER':^55}")
    print("AI: ",response.text, end="", flush=True)

    clean_resp = clean(response.text)
    save(query, clean_resp)
if __name__ == "__main__":
  main()