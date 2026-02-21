import os
from random import randrange
import json
import re
import sys
import asyncio
import time
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv
from google import genai
from mistralai import Mistral
import gradio as gr
from pydantic import BaseModel, ValidationError, Field
import logging

# Configura il logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# LangChain Community / Core
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS 
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter

# We replace langchain_google_genai with our own implementation using mistralAI and google-genai embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Costanti
EXIT_COMMANDS = ["esci", "quit", "exit"]
OPEN_FILE_COMMAND_PREFIX = "apri "
OPEN_AUDIO_COMMAND_PREFIX = "voxtralaudio "
DB_PATH = "./data/faiss_index"
ALLOWED_UPLOAD_DIR = "./uploads"

# Crea la cartella di upload se non esiste
os.makedirs(ALLOWED_UPLOAD_DIR, exist_ok=True)

class Config(BaseModel):
    mistral_api_key: str = Field(..., min_length=1)
    gemini_api_key: str = Field(..., min_length=1)
    proxy_url: Optional[str] = None

class GeminiEmbeddings(Embeddings):
    # Custom Embeddings class using the official google-genai SDK.
    
    def __init__(self, eclient: genai.Client, model: str = "models/gemini-embedding-001"):
        self.eclient = eclient
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Embed search docs.
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
             response = self.eclient.models.embed_content(
                model=self.model,
                contents=text
            )
             # response.embeddings might be a list if input was list, or single if input was single.
             # If input is single string, it returns one embedding.
             if response.embeddings:
                 embeddings.append(response.embeddings[0].values)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        # Embed query text.
        response = self.eclient.models.embed_content(
            model=self.model,
            contents=text
        )
        if response.embeddings:
            return response.embeddings[0].values
        return []


class mistralAIWrapper(Runnable):
    """Wrapper to make mistralai behave like a Runnable for LangChain if needed,
       or just a helper for our async calls.

    Questa classe permette di integrare i modelli Mistral all'interno delle chain (LCEL).
    Risolve l'errore 'Expected a Runnable' ereditando dalla classe base Runnable.

    Attributes:
        client (Mistral): Istanza del client Mistral AI.
        model (str): Nome del modello da interrogare.
        temperature (float): Parametro di creatività della risposta.
    """

    def __init__(self, client: Mistral, model: str, temperature: float = 0.7):
        """Inizializza il wrapper Mistral.

        Args:
            client: Istanza del client Mistral.
            model: Nome del modello (es. 'mistral-small-latest').
            temperature: Valore per il campionamento della risposta.
        """
        self.client = client
        self.model = model
        self.temperature = temperature

    def invoke(self, input_data: Any, config: Optional[Dict] = None) -> str:
        """Metodo sincrono obbligatorio per l'interfaccia Runnable.

        Args:
            input_data: Messaggi o testo in input.
            config: Configurazioni opzionali di LangChain.

        Returns:
            Testo della risposta generata.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(input_data, config))
        return asyncio.run(self.ainvoke(input_data, config))

    async def ainvoke(self, input_data: Any, config: Optional[Dict] = None) -> str:
        """Versione asincrona del metodo invoke.

        Args:
            input_data: Input per il modello.
            config: Configurazione opzionali.

        Returns:
            Risposta del modello come stringa.
        """
        # Trasformazione input in formato messaggi Mistral
        messages = []
        if hasattr(input_data, 'to_messages'):
            for msg in input_data.to_messages():
                role = "user"
                if msg.type == "system": role = "system"
                elif msg.type == "ai": role = "assistant"
                messages.append({"role": role, "content": msg.content})
        elif isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        else:
            messages = [{"role": "user", "content": str(input_data)}]

        response = await self.client.chat.complete_async(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        return response.choices[0].message.content

class Codes: 
      def __init__(self, api_key: str, apiembed: str): 
           # Initialize official SDK client
           self.client = Mistral(api_key=api_key)
           self.eclient = genai.Client(api_key=apiembed)
           # Use our custom embeddings model with client
           
           self.embeddings = GeminiEmbeddings(self.eclient, model="models/gemini-embedding-001")

      async def call_model(self, modelone: str, prompt, temperature, tools):
          return await self.client.chat.complete_async(model=modelone, messages=[{"role": "user", "content": prompt},], temperature=temperature, tools=tools)

      async def transcribe_audio(self, audio_file_path: str) -> str:
          """Transcribe audio file using Mistral AI transcription service"""
          try:
              # Converti il file in .mp3 se non lo è già
              if not audio_file_path.lower().endswith('.mp3'):
                  from pydub import AudioSegment
                  import os
                  
                  # Carica il file audio
                  audio = AudioSegment.from_file(audio_file_path)
                  
                  # Crea un percorso temporaneo per il file .mp3
                  temp_mp3_path = os.path.splitext(audio_file_path)[0] + '.mp3'
                  
                  # Esporta il file in formato .mp3
                  audio.export(temp_mp3_path, format='mp3')
                  
                  # Usa il file .mp3 per la trascrizione
                  audio_file_path = temp_mp3_path
              
              # Leggi il file in formato binario
              with open(audio_file_path, "rb") as audio_file:
                  audio_data = audio_file.read()

              # Carica il file audio in formato binario
              uploaded_file = await self.client.files.upload_async(
                  file=audio_data,
                  purpose="batch"  # Specificare lo scopo del caricamento
              )

              # Esegui la trascrizione usando l'ID del file caricato
              transcription_result = await self.client.audio.transcriptions.complete(
                  model="voxtral-mini-latest",
                  file_id=uploaded_file.id
              )

              # Rimuovi il file temporaneo .mp3 se è stato creato
              if audio_file_path.endswith('.mp3') and 'temp' in audio_file_path:
                  os.remove(audio_file_path)

              return transcription_result.text
          except Exception as e:
              logger.error(f"Error in audio transcription: {e}")
              import traceback
              traceback.print_exc()
              return ""

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

async def run_assistant(llm_response: str, tool_result: Any, client_dress: Codes, original_file_path: Optional[str] = None) -> str:
        if not tool_result:
            # Check if the file is a PDF and fallback to OCR
            if original_file_path and original_file_path.lower().endswith('.pdf'):
                logger.info("Fallback OCR attivato")
                try:
                    # Upload the file to Mistral
                    with open(original_file_path, "rb") as f:
                        uploaded_file = await client_dress.client.files.upload_async(
                            file={"file_name": os.path.basename(original_file_path), "content": f}
                        )
                    
                    # Process the file with OCR
                    ocr_result = await client_dress.client.ocr.process(
                        model="mistral-ocr-latest",
                        file=uploaded_file
                    )
                    
                    # Use OCR result as new context
                    tool_result = ocr_result.text
                except Exception as e:
                    logger.error(f"Errore durante l'OCR: {e}")
                    return "Mi dispiace, non ho trovato informazioni utili."
            else:
                return "Mi dispiace, non ho trovato informazioni utili."
        
        final_query_prompt = """The 'system' response: \n"{response}"
        If you cannot respond using a tool, extract the main keywords in the text into a comma-separated list, then if possible
        summarize the text in a very short form.
        If is irrilevant give an answer of keywords or summary, respond, ** I am glad to assist. ** Bye. *
        """
        
        prompt_question_tool = ChatPromptTemplate.from_messages([
        ("system", final_query_prompt),
        ("human", tool_result)
        ])
        llm_tool = mistralAIWrapper(client=client_dress.client, model="mistral-small-latest", temperature=0.1)
        response = llm_tool.invoke({"response": llm_response})

        chat = client_dress.call_model(
            model="mistral-small-latest", 
            messages=response,
            temperature=0.7,
            tools=tools_result
            )
    
        return chat.choices[0].message.content

async def generate_response(session_id, input_text, from_document: Optional[str], chain: Any, llm1: Any, available_functions: Dict[str, Any], client_dress: Codes):
    doc : Optional[str] = None
    allowed_extensions = {
    "text2": ".md",
    "text4": ".pdf"
    }
    
    summary_res = ""
    r_t : Optional[str] = None
    
    if from_document is not None:
        # Valida il percorso del file
        try:
            file_path = os.path.abspath(os.path.join(ALLOWED_UPLOAD_DIR, from_document))
            if not file_path.startswith(os.path.abspath(ALLOWED_UPLOAD_DIR)):
                logger.warning(f"Tentativo di accesso a percorso non consentito: {from_document}")
                return "Mi dispiace, non puoi accedere a questo file.", None
            
            name, file_extension = os.path.splitext(from_document)
            file_extension_lower = file_extension.lower()
            if file_extension_lower in allowed_extensions.values():
                doc = name
        except Exception as e:
            logger.error(f"Errore nella validazione del percorso del file: {e}")
            return "Mi dispiace, si è verificato un errore nella validazione del file.", None

    try:
        response = await chain_with_history.ainvoke({"input": input_text}, config={"session_id": str(session_id)})
        
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
               r_t = await run_assistant(response, risultato_tool, client_dress, original_file_path=from_document)
            else:
               r_t = None

        except Exception as e:
            logger.error(f"Errore nella generazione del riassunto/tool: {e}")
            import traceback
            traceback.print_exc()
            r_t = None
    
    except Exception as e:
        logger.error(f"Errore nell'elaborazione del messaggio: {e}")
        import traceback
        traceback.print_exc()
        response = "Mi dispiace, si è verificato un errore nell'elaborazione della tua richiesta."
        r_t = None
    return response, r_t

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
        "text4": ".pdf",
        "audio1": ".mp3",
        "audio2": ".wav",
        "audio3": ".ogg",
        "audio4": ".m4a",
        "audio5": ".oga"
    }
    
    # Valida il percorso del file
    try:
        file_path = os.path.abspath(os.path.join(ALLOWED_UPLOAD_DIR, path))
        if not file_path.startswith(os.path.abspath(ALLOWED_UPLOAD_DIR)):
            logger.warning(f"Tentativo di accesso a percorso non consentito: {path}")
            return None
        
        name, file_extension = os.path.splitext(path)
        file_extension_lower = file_extension.lower()
        
        if file_extension_lower in allowed_extensions.values():
            try:   
                def format_txt(docs):
                    return "".join(doc for doc in docs)
                
                if file_extension_lower in [".mp3", ".wav", ".ogg", ".m4a", ".oga"]:
                    # Handle audio file transcription
                    transcription = await client_dress.transcribe_audio(file_path)
                    if transcription:
                        # Save transcription to text file
                        txt_filename = f"{name}_transcription.txt"
                        txt_filepath = os.path.join(ALLOWED_UPLOAD_DIR, txt_filename)
                        with open(txt_filepath, "w", encoding="utf-8") as txt_file:
                            txt_file.write(transcription)
                        return f"Audio transcribed successfully. Transcription saved to: {txt_filename}"
                    else:
                        return "Failed to transcribe audio file"
                
                elif file_extension_lower == allowed_extensions['text1'] or file_extension_lower == allowed_extensions['text3']:
                    # Run file IO in thread
                    def read_sync():
                        with open(file_path, mode="r", encoding="utf-8") as f:
                            lines = f.readlines()
                            logger.debug(f"DEBUG: Uso readlines(), n. righe {len(lines)}:")
                            return format_txt(lines)
                    return await asyncio.to_thread(read_sync)
                elif file_extension_lower == allowed_extensions['text2'] or file_extension_lower == allowed_extensions['text4']:
                    def vector_sync():
                        loader = PyPDFLoader(file_path) if file_extension_lower == ".pdf" else TextLoader(file_path, encoding="utf-8")
                        documents = loader.load()
                        
                        splitter = get_text_splitter(file_extension_lower)
                        splits = splitter.split_documents(documents)

                        # Crea/Carica VectorStore
                        vector_store = FAISS.from_documents(splits, client_dress.embeddings)
                        index_name = f"{file_path}_faiss"
                        vector_store.save_local(index_name)
                        return format_md(splits)
                    return await asyncio.to_thread(vector_sync)
                
            except FileNotFoundError:
                logger.error(f"Errore: Il file '{path}' non è stato trovato.")
                return None
            except Exception as e:
                logger.error(f"Errore durante il caricamento del file '{path}': {e}")
                return None
        else:
            logger.error(f"Errore: Estensione del file '{file_extension}' non supportata. Estensioni supportate: {list(allowed_extensions.values())}")
            return None
    except Exception as e:
        logger.error(f"Errore nella validazione del percorso del file: {e}")
        return None

def parse_command(command_input: str) -> tuple[str, str | None]:
    # Assicurati che command_input sia una stringa
    if isinstance(command_input, list):
        # Se è una lista di dizionari, estrai il contenuto
        if isinstance(command_input[0], dict):
            command_input = command_input[0].get("content", "")
        else:
            command_input = ' '.join(command_input)
    elif isinstance(command_input, dict):
        command_input = command_input.get("content", "")
    
    command_lower = command_input.lower().strip()
    if command_lower in EXIT_COMMANDS:
        return "exit", None
    elif command_lower.startswith(OPEN_FILE_COMMAND_PREFIX):
        file_name = command_input[len(OPEN_FILE_COMMAND_PREFIX):].strip()
        return "open_file", file_name
    elif command_lower.startswith("voxtralaudio "):
        audio_file = command_input[len(OPEN_AUDIO_COMMAND_PREFIX):].strip()
        return "transcribe_audio", audio_file
    else:
        return command_input, None

async def chat_response(message: str, history: List[List[str]], session_id: str, client_dress: Codes, chain: Any, llm1: Any, available_functions: Dict[str, Any], uploaded_file: Optional[str] = None) -> str:
    """Handle chat responses and commands."""
    action_type, data = parse_command(message)
    
    if action_type == "exit":
        return "Arrivederci!"
    elif action_type == "open_file":
        file_name = data
        file_content = await load_file(file_name, client_dress)
        if file_content is not None:
            response, r_t = await generate_response(session_id, file_content, data, chain, llm1, available_functions, client_dress)
            return r_t if r_t else response
        else:
            return f"Errore nel caricamento del file: {file_name}"
    elif action_type == "transcribe_audio":
        audio_file = data
        transcription_result = await load_file(audio_file, client_dress)
        if transcription_result:
            return transcription_result
        else:
            return f"Errore nella trascrizione del file audio: {audio_file}"
    elif uploaded_file is not None:
        # Gestisci il file caricato
        file_content = await load_file(uploaded_file.name, client_dress)
        if file_content is not None:
            response, r_t = await generate_response(session_id, file_content, uploaded_file.name, chain, llm1, available_functions, client_dress)
            return r_t if r_t else response
        else:
            return f"Errore nel caricamento del file: {uploaded_file.name}"
    else:
        response, r_t = await generate_response(session_id, action_type, data, chain, llm1, available_functions, client_dress)
        return r_t if r_t else response

async def main_gradio():
    load_dotenv() 
    api_key = os.getenv('MISTRAL_API_KEY')
    apiembed = os.getenv('GEMINI_API_KEY')
    proxy = os.getenv('PROXY_URL', None) # Better to use env var
    
    try:
        config = Config(mistral_api_key=api_key, gemini_api_key=apiembed, proxy_url=proxy)
    except ValidationError as e:
        logger.error(f"Errore di validazione della configurazione: {e}")
        raise
    
    s1 = randrange(10000, 99999)
    s2 = int(time.time() * 1000)
    s3 = randrange(10000, 99999)
    DEFAULT_CHAT_ID = f"{s1}_{s2}_{s3}"
    models1 = "mistral-small-latest" 
    
    available_functions = {"read_key_files": read_key_files}

    system_msg = """ 
                           Sei un assistente utile a tutti. Hai la documentazione in stile Google per il codice fornito mentre tutte le espressioni matematiche valutale in questo modo:   
                           Ragionamento:
                           **
                           **
                           Risposta:
                           Per tutte le altre tipologie che non richiedono documentazione, rispondi per rounds item di modo che si possano estrarre parole-chiave dal testo.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    tool_system_prompt = """ 
                    You are an assistant specialized in calling external functions.
                    You have the following tools (tools) available:
                    **  {tools}        
                    If the user asks something that requires a tool, respond in this manners:
                    **  <call:tool_name>{{"param": "value"}}</call>
                    if the external functions cannot found reply: **  I do not have the capability to assist with programming tasks such as creating or modifying Python code. 
                    My current tools are focused on calling external functions like  reading key files. 
                    I cannot provide assistance with file management or programming. **                           
    """
    prompt_template_tool = ChatPromptTemplate.from_messages([
        ("system", tool_system_prompt),
        ("human", "{query}")
    ])

    client_dress = Codes(config.mistral_api_key, config.gemini_api_key)
    
    # Init Wrappers instead of LangChain legacy models
    modello = mistralAIWrapper(client=client_dress.client, model=models1, temperature=0.7)
    llm_tool = mistralAIWrapper(client=client_dress.client, model=models1, temperature=0.1)

    # chains
    chain = prompt | modello | StrOutputParser()
    llm1  = prompt_template_tool | llm_tool | StrOutputParser()
    
    # creazione memoria conversazionale
    chat_map = {}
    def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in chat_map:
            chat_map[session_id] = InMemoryChatMessageHistory()
        return chat_map[session_id]

    # chain del chatbot con memoria conversazionale
    global chain_with_history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_chat_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    # Define handle_file_upload function before creating the Gradio interface
    def handle_file_upload(file_obj):
        if file_obj is None:
            return None
        # Salva il file nella cartella di upload
        file_path = os.path.join(ALLOWED_UPLOAD_DIR, file_obj.name)
        with open(file_path, "wb") as f:
            f.write(file_obj.read())
        logger.info(f"File caricato: {file_path}")
        return file_obj

    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# 🤖 Assistente PHIL (v2 con trascrizione audio)")
        gr.Markdown("### Comandi disponibili:")
        gr.Markdown("- `apri nomefile.ext` per caricare file (txt, md, json, pdf, mp3, wav, m4a, ogg, oga)")
        gr.Markdown("- `voxtralaudio nomefile.XXX` per trascrivere file audio")
        gr.Markdown("- `esci` per terminare")
        
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Il tuo messaggio")
        file = gr.File(label="Carica un file")
        clear = gr.Button("Pulisci chat")
        
        # Store session ID and chat history in Gradio state
        state = gr.State(DEFAULT_CHAT_ID)
        uploaded_file = gr.State(None)
        
        def user(user_message, history, uploaded_file):
            return "", history + [{"role": "user", "content": user_message}], uploaded_file

        async def bot(history, state, uploaded_file):
            user_message = history[-1]["content"]
            bot_message = await chat_response(user_message, history, state, client_dress, chain, llm1, available_functions, uploaded_file)
            history.append({"role": "assistant", "content": bot_message})
            return history, state, None

        def clear_chat(state):
            # Generate a new session ID
            s1 = randrange(10000, 99999)
            s2 = int(time.time() * 1000)
            s3 = randrange(10000, 99999)
            new_session_id = f"{s1}_{s2}_{s3}"
            return [], new_session_id

        file.upload(handle_file_upload, [file], [uploaded_file])
        msg.submit(user, [msg, chatbot, uploaded_file], [msg, chatbot, uploaded_file], queue=False).then(
            bot, [chatbot, state, uploaded_file], [chatbot, state, uploaded_file]
        )
        clear.click(clear_chat, [state], [chatbot, state])
    
    demo.launch(debug=False, share=False)

if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(main_gradio())
        finally:
            loop.close()
    except KeyboardInterrupt:
        logger.info("\nInterrotto dall'utente.")