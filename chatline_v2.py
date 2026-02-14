import os
import json
import re
import asyncio
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

# LangChain Community / Core
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS 
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter

# We replace langchain_google_genai with our own implementation using google-genai SDK
from langchain_core.embeddings import Embeddings

# Costanti
EXIT_COMMANDS = ["esci", "quit", "exit"]
OPEN_FILE_COMMAND_PREFIX = "apri "
DB_PATH = "./data/faiss_index"

class GeminiEmbeddings(Embeddings):
    """Custom Embeddings class using the official google-genai SDK."""
    
    def __init__(self, client: genai.Client, model: str = "models/gemini-embedding-001"):
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        # google-genai supports batch embedding? 
        # The SDK documentation suggests using models.embed_content. 
        # For simple implementations we loop or check batch support.
        # As of early versions, simple loop is safest if batch API details are vague, 
        # but modern Gemini APIs usually support batch. 
        # checks: client.models.embed_content(..., contents=texts)
        
        # We will assume batch support for contents list, or loop if it fails. 
        # Actually most robust way for now is simple loop to avoid size limits issues blindly.
        embeddings = []
        for text in texts:
             response = self.client.models.embed_content(
                model=self.model,
                contents=text
            )
             # response.embeddings might be a list if input was list, or single if input was single.
             # If input is single string, it returns one embedding.
             if response.embeddings:
                 embeddings.append(response.embeddings[0].values)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        response = self.client.models.embed_content(
            model=self.model,
            contents=text
        )
        if response.embeddings:
            return response.embeddings[0].values
        return []

class GeminiGenAIWrapper:
    """Wrapper to make google-genai behave like a Runnable for LangChain if needed,
       or just a helper for our async calls."""
    def __init__(self, client: genai.Client, model: str, temperature: float = 0.7):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.config = types.GenerateContentConfig(temperature=temperature)

    async def ainvoke(self, input_data: Any) -> str:
        """Mimics LangChain ainvoke but uses official SDK."""
        # input_data from PromptTemplate is usually a StringValue or Dict.
        # If it came from a PromptTemplate | Wrapper chain, input_data is the prompt string.
        
        prompt_text = ""
        if isinstance(input_data, str):
            prompt_text = input_data
        elif hasattr(input_data, 'to_string'):
             prompt_text = input_data.to_string()
        elif isinstance(input_data, dict):
            # If the prompt template returned a dict (rare for strict string chain), handle it.
            # Usually PromptTemplate.format(**kwargs) returns a string (StringPromptValue).
            # But LangChain chain passing might pass PromptValue.
            pass 
        else:
             prompt_text = str(input_data)
             
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt_text,
            config=self.config
        )
        return response.text

class Codes: 
      def __init__(self, api_key: str, proxy: Optional[str] = None): 
           http_options = {'proxy': proxy} if proxy else None
           # Initialize official SDK client
           if http_options:
               # Note: 'http_options' in v1.0 might differ, checking standard usage.
               # For now assuming standard Client init.
               # If proxy support is strictly needed, we might need requests/httpx transport config.
               # google-genai uses httpx. 
               self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(proxy=proxy))
           else:
               self.client = genai.Client(api_key=api_key)
           
           # Use our custom GeminiEmbeddings with the same client
           self.embeddings = GeminiEmbeddings(self.client, model="models/gemini-embedding-001")

      async def call_model(self, modelone: str, prompt, temperature):
          return await self.client.aio.chats.create(model=modelone, config=types.GenerateContentConfig(system_instruction=prompt, temperature=temperature))
    
def read_key_files(keywordprocessing1: str, keywordprocessing2: str):
    return json.dumps({"keyword1": keywordprocessing1, "keyword2": keywordprocessing2})

def format_md(docs: List[Any]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)
    
def execute_tool_call(model_response: str, available_tools: Dict[str, callable]):
    match = re.search(r"<call:(\w+)>(.*?)</call>", model_response, re.DOTALL)
    if not match:
       return None
    if match:
        func_name = match.group(1)
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
             return None

        if func_name in available_tools:
            return available_tools[func_name](**args)
        else:
            return None

# --- LOGICA DI ESECUZIONE GENERAZIONE FINALE RISPOSTA ---

async def run_assistant(llm_response: str, tool_result: Any, client_dress: Codes) -> str:
        if not tool_result:
            return "Mi dispiace, non ho trovato informazioni utili."
        final_query_prompt = """The 'system' response: \n"{response}"
        The tool returned this data: {tool_result}\n
        If you cannot respond using a tool, extract the main keywords in the text into a comma-separated list, then if possible
        summarize the text in a very short form.
        If is irrilevant give an answer of keywords or summary, respond, ** I am glad to assist. ** Bye. *
        """
        
        chat = client_dress.client.chats.create(
            model="gemini-3.0-flash-preview", 
            config=types.GenerateContentConfig(
                system_instruction=final_query_prompt.format(response=llm_response, tool_result=tool_result), 
                temperature=0.1
            )
        )
        
        response = await chat.send_message("Please proceed.")
        return response.text

async def generate_response(input_text, from_document: Optional[str], chain: Any, llm1: Any, available_functions: Dict[str, Any], client_dress: Codes):
    doc : Optional[str] = None
    allowed_extensions = {
    "text2": ".md",
    "text4": ".pdf"
    }
    
    summary_res = ""
    
    if from_document is not None:
        name, file_extension = os.path.splitext(from_document)
        file_extension_lower = file_extension.lower()
        if file_extension_lower in allowed_extensions.values():
            doc = name

    try:
        response = await chain.ainvoke({"input": input_text})
        print("AI: \n", response, "\n")
        
        try:
            if doc is None:
                TOOLS_DESCRIPTION = """
                read_key_file(keyword: str): Restituisce il topic del testo
                """
                summary_res = await llm1.ainvoke({"tools": TOOLS_DESCRIPTION, "query": input_text })
            elif doc is not None and from_document:
                index_path = f"{from_document}_faiss"
                if os.path.exists(index_path):
                    # FAISS load_local is synchronous
                    vector_store = FAISS.load_local(index_path, client_dress.embeddings, allow_dangerous_deserialization=True)
                    docs = vector_store.similarity_search(doc, k=3)
                    summary_res = format_md(docs)
                        
                    
            risultato_tool = execute_tool_call(summary_res, available_functions)             
                         
            if risultato_tool:
               r_t = await run_assistant(response, risultato_tool, client_dress)
               print('--------------------------------\n') 
               print(r_t)
               print('--------------------------------\n')
            else:
               print(f"Risposta testuale: {summary_res}")

        except Exception as e:
            print(f"Errore nella generazione del riassunto/tool: {e}")
    
    except Exception as e:
        print(f"Errore nell'elaborazione del messaggio: {e}")
        response = "Mi dispiace, si è verificato un errore nell'elaborazione della tua richiesta."
    return "input_text"

def get_text_splitter(extension: str):
    params = {"chunk_size": 800, "chunk_overlap": 100}
    if extension == ".md":
        return MarkdownTextSplitter(**params)
    return RecursiveCharacterTextSplitter(**params)
    
async def load_file(path: str, client_dress: Codes) -> str | None:
    # Made async to be called properly
    allowed_extensions = {
        "text1": ".txt",
        "text2": ".md",
        "text3": ".json",
        "text4": ".pdf"
    }
    
    name, file_extension = os.path.splitext(path)
    file_extension_lower = file_extension.lower()
 
    if file_extension_lower in allowed_extensions.values():
        try:   
            def format_txt(docs):
                return "".join(doc for doc in docs)
            
            if file_extension_lower == allowed_extensions['text1'] or file_extension_lower == allowed_extensions['text3']:
                # Run file IO in thread
                def read_sync():
                    with open(path, mode="r", encoding="utf-8") as f:
                        lines = f.readlines()
                        print(f"DEBUG: Uso readlines(), n. righe {len(lines)}:")
                        return format_txt(lines)
                return await asyncio.to_thread(read_sync)
            elif file_extension_lower == allowed_extensions['text2'] or file_extension_lower == allowed_extensions['text4']:
                def vector_sync():
                    loader = PyPDFLoader(path) if file_extension_lower == ".pdf" else TextLoader(path, encoding="utf-8")
                    documents = loader.load()
                    
                    splitter = get_text_splitter(file_extension_lower)
                    splits = splitter.split_documents(documents)

                    # Crea/Carica VectorStore
                    vector_store = FAISS.from_documents(splits, client_dress.embeddings)
                    index_name = f"{path}_faiss"
                    vector_store.save_local(index_name)
                    return format_md(splits)
                return await asyncio.to_thread(vector_sync)
            
        except FileNotFoundError:
            print(f"Errore: Il file '{path}' non è stato trovato.")
            return None
        except Exception as e:
            print(f"Errore durante il caricamento del file '{path}': {e}")
            return None
    else:
        print(f"Errore: Estensione del file '{file_extension}' non supportata. Estensioni supportate: {list(allowed_extensions.values())}")
        return None

def parse_command(command_input: str) -> tuple[str, str | None]:
    command_lower = command_input.lower().strip()
    if command_lower in EXIT_COMMANDS:
        return "exit", None
    elif command_lower.startswith(OPEN_FILE_COMMAND_PREFIX):
        file_name = command_input[len(OPEN_FILE_COMMAND_PREFIX):].strip()
        return "open_file", file_name
    else:
        return command_input, None

async def main():
    load_dotenv() 
    api_key = os.getenv('GEMINI_API_KEY')
    proxy = os.getenv('PROXY_URL', None) # Better to use env var
    models1 = "gemini-2.5-flash"
    models2 = "gemini-3.0-flash-preview" 
    
    print("🤖 Assistente Gemma pronto! (v2 Google GenAI SDK)")
    available_functions = {"read_key_files": read_key_files}

    system_template =  """ 'system',
                           Sei un assistente utile a tutti. Hai la documentazione in stile Google per il codice fornito mentre tutte le espressioni matematiche valutale in questo modo:   
                           <ragionamento>
                           **
                           </ragionamento>
                           **
                           <risposta>
                           Per tutte le altre tipologie che non richiedono documentazione, rispondi per rounds item di modo che si possano estrarre parole-chiave dal testo.
                           </risposta>
    """
    system_templateA = f"{system_template}\nuser:" + "{input}"
    prompt = PromptTemplate.from_template(system_templateA)

    s2_prompt = """ 'system',
                    You are an assistant specialized in calling external functions.
                    You have the following tools (tools) available:
                    **  {tools}        
                    If the user asks something that requires a tool, respond in this manners:
                    **  <call:tool_name>{{"param": "value"}}</call>
                    if the external functions cannot found reply: **  I do not have the capability to assist with programming tasks such as creating or modifying Python code. 
                    My current tools are focused on calling external functions like  reading key files. 
                    I cannot provide assistance with file management or programming. **                           
    """
    full_prompt = f"{s2_prompt}\nuser:" + "{query}"
    prompt_template = PromptTemplate.from_template(full_prompt)

    client_dress = Codes(api_key, proxy)
    
    # Init Wrappers instead of LangChain legacy models
    modello = GeminiGenAIWrapper(client=client_dress.client, model=models1, temperature=0.7)
    llm_tool = GeminiGenAIWrapper(client=client_dress.client, model=models2, temperature=0.1)

    # chains
    chain = prompt | modello | StrOutputParser()
    llm1  = prompt_template | llm_tool | StrOutputParser()
    
    while True:
      gray_color = "\033[90m"
      reset_color = "\033[0m"
      print(f"\n{'-'*55}\n\n{reset_color}")
      print(f"{gray_color}\n\n{'-'*20} Chat {'-'*20}\n") 	
      print()
      
      msg = await asyncio.to_thread(input, "Tu:")
      
      action_type, data = parse_command(msg)
      if action_type == "exit":
         print("Arrivederci!")
         break
      elif action_type == "open_file":
           file_name = data
           file_content = await load_file(file_name, client_dress)
           if file_content is not None:
              await generate_response(file_content, data, chain, llm1, available_functions, client_dress)
      else:
           await generate_response(action_type, data, chain, llm1, available_functions, client_dress)
      
          
if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
