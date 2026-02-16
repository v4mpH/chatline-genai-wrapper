import os
import pyttsx3
import vosk
import sounddevice as sd
import json
import queue
import re
import sys
from typing import Optional, Dict, List, Tuple, Any
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts  import MessagesPlaceholder
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter


# Costanti
EXIT_COMMANDS = ["esci", "quit", "exit"]
OPEN_FILE_COMMAND_PREFIX = "apri "
doc : Optional[str] = None
DB_PATH = "./data/faiss_index"

embeddings_model = OllamaEmbeddings(model="nomic-embed-text-v2-moe")

def read_key_files(keywordprocessing1: str, keywordprocessing2: str):
    # Search keywords into output text
    # Args: Keyword
    return json.dumps({"keyword1": keywordprocessing1, "keyword2": keywordprocessing2})

def format_md(docs: List[Any]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)
    
def get_speech_engine() -> pyttsx3.Engine:
    """Inizializza e configura l'engine di sintesi vocale.

    Invece di usare una costante globale, questa funzione permette di 
    ottenere un'istanza pulita dell'engine pyttsx3 quando necessario.

    Returns:
        pyttsx3.Engine: L'istanza dell'engine di sintesi vocale configurata.
    """
    engine = pyttsx3.init()
    # È possibile aggiungere configurazioni qui (rate, volume, voices)
    return engine

def speak(engine, text):
    engine.say(text)
    engine.runAndWait()


def listen():
    model_path = "C:\\Users\\v4mp1\\Favorites\\Links\\voicesearch\\vosk-model-it-0.22"
    model = vosk.Model(model_path)
    vosk.SetLogLevel(-1)
    q = queue.Queue()
  
    def callback(indata, frames, time, status):
        if status:
            print(status, flush=True)
        q.put(bytes(indata))
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
        recognizer = vosk.KaldiRecognizer(model, 16000)
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                return result.get("text", "")    
# La funzione di Parsing (quella prima)
def execute_tool_call(model_response, available_tool):
    """
    Parses Gemma's output and executes the function.

    Args:
        model_response (str): La risposta generata dal modello linguistico.
        available_tool (dict): available functions
    Returns:
        Any: Il risultato dell'esecuzione della funzione, se una chiamata a una funzione è stata trovata.
             Restituisce None se non è stata trovata una chiamata a una funzione.
    """    
    match = re.search(r"<call:(\w+)>(.*?)</call>", model_response, re.DOTALL)
    if match:
        func_name = match.group(1)
        args = json.loads(match.group(2))
        if func_name in available_tool:
            logging.info(f"Esecuzione della funzione: {func_name} con argomenti: {args}")
            return available_tool[func_name](**args)
        else:
           logging.warning(f"Funzione non riconosciuta: , mostro i tool avviabili")
           return available_tool
    return None
    

# --- LOGICA DI ESECUZIONE GENERAZIONE FINALE RISPOSTA ---

def run_assistant(llm_response: str, tool_result: Any, pipe_model: OllamaLLM) -> str:
    """Genera la risposta finale basata sui dati ottenuti dai tool.

    Args:
        llm_response: Risposta originale dell'assistente.
        tool_result: Dati restituiti dal tool.
        pipe_model: Modello LLM da utilizzare per la formattazione.

    Returns:
        str: Risposta finale per l'utente.
    """
    if not tool_result:
        return "Mi dispiace, non ho trovato informazioni utili."
    final_query_prompt = """The 'system' response: /n"{response}"
    The tool returned this data: {tool_result}/n
    If you cannot respond using a tool, extract the main keywords in the text into a comma-separated list, then if possible
    summarize the text in a very short form.
    If is irrilevant give an answer of keywords or summary, respond, ** I am glad to assist. ** Bye. *
    """
    final_answer_prompt = PromptTemplate.from_template(final_query_prompt)
    
    chain = final_answer_prompt | pipe_model | StrOutputParser()
    return chain.invoke({"response": llm_response, "tool_result": tool_result})
    
    # Se non è un tool, restituisce la risposta diretta del modello
    return response

def generate_response(session_id, input_text, from_document: Optional[str], chain_with_history: Any, llm1: Any, available_functions: Dict[str, Any], pipe_model: OllamaLLM):
    """Gestisce il flusso di generazione risposta, inclusa l'eventuale analisi documenti.

    Args:
        session_id: ID della sessione chat.
        input_text: Messaggio dell'utente.
        from_document: Nome del documento caricato (opzionale).
        chain_with_history: Chain principale con memoria.
        llm1: Chain per l'estrazione dei tool.
        available_functions: Funzioni disponibili.
        pipe_model: Modello per la risposta finale del topic di un documento.

    Returns:
        Tuple[str, str]: ID sessione e messaggio di risposta.
    """
    history = []
    
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
    # Processo la risposta dell'LLM
    try:
        response = chain_with_history.invoke({"input": input_text}, config={"session_id": str(session_id)})
        print("AI: \n", response, "\n")
        
        history.append({"role": "user", "content": input_text})
        history.append({"role": "assistant", "content": response})

        # Generazione di un riassunto della conversazione ed estrazione dei principali tag riscontrati
        try:
            if doc is None:
                TOOLS_DESCRIPTION = """
                read_key_file(keyword: str): Restituisce il topic del testo
                """
                summary_res = llm1.invoke({"tools": TOOLS_DESCRIPTION, "query": input_text })
            elif doc is not None and from_document:
                index_path = f"{from_document}_faiss"
                if os.path.exists(index_path):
                    # Carichiamo la CARTELLA nativa di FAISS, non un file JSON
                    vector_store = FAISS.load_local(index_path, embeddings_model, allow_dangerous_deserialization=True)
                    docs = vector_store.similarity_search(doc, k=3)
                    summary_res = format_md(docs)
                        
                    
                                         
            risultato_tool = execute_tool_call(summary_res, available_functions)             
                         
            
            if risultato_tool:
               # Stampa keyword estratte dall'llm specializzato
               r_t = run_assistant(response, risultato_tool, pipe_model)
               print('--------------------------------\n') 
               print(r_t)
               print('--------------------------------\n')
            else:
               print(f"Risposta testuale: {summary_res}")

            # Gestione errore della invoke di estrazione keyword    
        except Exception as e:
            print(f"Errore nella generazione del riassunto: {e}")
    
    # Gestione errore della invoke di risposta del LLM all'input dell'utente
    except Exception as e:
        print(f"Errore nell'elaborazione del messaggio: {e}")
        response = "Mi dispiace, si è verificato un errore nell'elaborazione della tua richiesta."
        history.append({"role": "user", "content": input_text})
        history.append({"role": "assistant", "content": response})
    return session_id, "input_text"
    

def get_text_splitter(extension: str):
    # Restituisce lo splitter appropriato in base all'estensione.
    params = {"chunk_size": 800, "chunk_overlap": 100}
    if extension == ".md":
        return MarkdownTextSplitter(**params)
    return RecursiveCharacterTextSplitter(**params)
    
def load_file(path: str) -> str | None:
    # Carica un file, lo processa e lo indicizza se necessario.
    
    allowed_extensions = {
        "text1": ".txt",
        "text2": ".md",
        "text3": ".json",
        "text4": ".pdf",
        'text5': ".log"
    }
    
    name, file_extension = os.path.splitext(path)
    file_extension_lower = file_extension.lower()
 
    if file_extension_lower in allowed_extensions.values():
        try:   
            def format_txt(docs):
                return "".join(doc for doc in docs)
            
            if file_extension_lower == allowed_extensions['text1'] or file_extension_lower == allowed_extensions['text3'] or file_extension_lower == allowed_extensions['text5']:
                with open(path, mode="r", encoding="utf-8") as f:
                    lines = f.readlines()
                    print(f"DEBUG: Uso readlines(), n. righe {len(lines)}:")
                    file_content = format_txt(lines)
                return file_content
            elif file_extension_lower == allowed_extensions['text2'] or file_extension_lower == allowed_extensions['text4']:
                loader = PyPDFLoader(path) if file_extension_lower == ".pdf" else TextLoader(path, encoding="utf-8")
                documents = loader.load()
                
                splitter = get_text_splitter(file_extension_lower)
                splits = splitter.split_documents(documents)

                # Crea/Carica VectorStore
                vector_store = FAISS.from_documents(splits, embeddings_model)
                index_name = f"{path}_faiss"
                vector_store.save_local(index_name)
                
                # Restituiamo tutto il contenuto dei chunk per coerenza con la richiesta originale
                file_content = format_md(splits)
                return file_content
            
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
    """
    Analizza il comando utente e restituisce l'azione e i dati associati.

    Args:
        command_input: La stringa di comando grezza dall'utente.

    Returns:
        Una tupla (action_type, data).
        action_type può essere: "exit", "open_file", "unrecognized", "process_message".
        data può essere il nome del file, il messaggio originale o None.
    """
    command_lower = command_input.lower().strip()

    if command_lower in EXIT_COMMANDS:
        return "exit", None
    elif command_lower.startswith(OPEN_FILE_COMMAND_PREFIX):
        file_name = command_input[len(OPEN_FILE_COMMAND_PREFIX):].strip()
        return "open_file", file_name
    elif command_lower == "listen":
        return "listen", None    
    else:
        return command_input, None


def main():
    # Funzione principale per gestire l'interazione con l'utente.
    DEFAULT_CHAT_ID = "id_1234"
    print("🤖 Assistente Gemma pronto! Digita 'apri <nome_file.XXX>' per aprire un file, 'listen' per la digitazione vocale o 'esci' per uscire.")
    available_functions = {"read_key_files": read_key_files}
    try:
        model = OllamaLLM(model="gemma3:12b", temperature=1.0, max_tokens=4096)

        llm_tool = init_chat_model("ollama:functiongemma", temperature=.1, tools=available_functions)

        engine = get_speech_engine()
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")
        sys.exit(1)

    system_template =  """
                       Sei un assistente utile a tutti. Hai la documentazione in stile Google per il codice fornito mentre tutte le espressioni matematiche valutale in questo modo:   
                       <ragionamento>
                       **
                       </ragionamento>
                       **
                       <risposta>
                       Per tutte le altre tipologie che non richiedono documentazione, rispondi per rounds item di modo che si possano estrarre parole-chiave dal testo.
                       </risposta>
    """
                                            
    prompt = ChatPromptTemplate([
        ("system", system_template),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
    ])

    s2_prompt = """ You are an assistant specialized in calling external functions.
                    You have the following tools (tools) available:
                    **  {tools}        
                    If the user asks something that requires a tool, respond in this manners:
                    **  <call:tool_name>{{"param": "value"}}</call>
                    if the external functions cannot found reply: **  I do not have the capability to assist with programming tasks such as creating or modifying Python code. 
                    My current tools are focused on calling external functions like  reading key files. 
                    I cannot provide assistance with file management or programming. **                           
    """

    full_prompt = f"{s2_prompt}\nUser:" + "{query}"

    prompt_template = PromptTemplate.from_template(full_prompt)

    # chain del chatbot e del ricercatore
    chain = prompt | model
    llm1  = prompt_template | llm_tool | StrOutputParser() 
      
    # creazione memoria conversazionale
    chat_map = {}
    def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in chat_map:
            chat_map[session_id] = InMemoryChatMessageHistory()
        return chat_map[session_id]

    # chain del chatbot con memoria conversazionale
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_chat_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    

    while True:
      try:  
          # sequenze di escape ANSI per modificare il colore del testo nei terminali
          gray_color = "\033[90m"
          reset_color = "\033[0m"
          print(f"\n{'-'*55}\n\n{reset_color}")
          print(f"{gray_color}\n\n{'-'*20} Chat {'-'*20}\n") 	
          print()
          msg = input("Tu:")
          action_type, data = parse_command(msg)
          if action_type == "listen":
            speak(engine,"ascolto")
            result = listen()
            if result:
                msg = result
                if not msg: continue
          if action_type == "exit":
             print("Arrivederci!")
             sys.exit(0)
          elif action_type == "open_file":
               file_name = data # data è il nome del file
               file_content = load_file(file_name)
               if file_content is not None:
                  chat_id, mess = generate_response(DEFAULT_CHAT_ID, file_content, data, chain_with_history, llm1, available_functions, llm_tool)
          else:
               # action_type è il messaggio originale da processare
               chat_id, response_message = generate_response(DEFAULT_CHAT_ID, action_type, data, chain_with_history, llm1, available_functions, llm_tool) 
      except KeyboardInterrupt:
            break
          
if __name__ == "__main__":
    main()