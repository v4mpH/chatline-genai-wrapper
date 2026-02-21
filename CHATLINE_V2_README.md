# Chatline v2 - Documentazione Completa

## 🎯 Core Features

1. **RAG (Retrieval-Augmented Generation)** 
   - Supporto per file txt, md, json, pdf
   - Ricerca semantica con embeddings Google GenAI
   - Vector store FAISS per efficiente retrieval

2. **OCR Fallback Automatico**
   - Attivazione automatica per file PDF quando i tool standard falliscono
   - Utilizzo di Mistral OCR API (mistral-ocr-latest)
   - Integrazione trasparente nel flusso di risposta

3. **Trascrizione Audio Asincrona**
   - Supporto per formati: mp3, wav, m4a, ogg, oga
   - Utilizzo di Voxtral Large V3 via Mistral AI
   - Salvataggio automatico in file di testo

4. **Gestione Documenti Avanzata**
   - Creazione automatica di indici FAISS
   - Ricerca semantica con k=3 risultati più rilevanti
   - Supporto per multiple estensioni di file

5. **Tool Integration Professionale**
   - Integrazione completa con Mistral AI SDK
   - Supporto per tool calling asincrono
   - Gestione degli errori robusta

## 🔧 Comandi Disponibili

### Comando `apri`
```
apri nomefile.ext
```
Carica e processa file con estensioni supportate:
- `.txt`, `.md`, `.json` - Testo semplice
- `.pdf` - Documenti PDF (con OCR fallback)
- `.mp3`, `.wav`, `.ogg`, `.m4a` - File audio

### Comando `voxtralaudio`
```
voxtralaudio nomefile.ext
```
Trascrive file audio e salva automaticamente in:
- `nomefile_transcription.txt`
- Utilizza Whisper Large V3
- Operazione completamente asincrona

### Comando `esci`
```
esci
```
Termina l'applicazione in modo pulito

## 📁 File Management

### Salvataggio Automatico
- **Trascrizioni audio**: Salvate come `nomefile_transcription.txt`
- **Indici FAISS**: Salvati come `percorsofile_faiss/`
- **Log**: Messaggi di errore e stato chiaramente visibili

### Estensioni Supportate
```
Testo:    .txt, .md, .json
Documenti: .pdf
Audio:    .mp3, .wav, .ogg, .m4a
```

## 🤖 Architettura Tecnica

### Design Pattern
- **Async/Await**: Tutte le operazioni I/O sono non bloccanti
- **Modularità**: Classi ben separate con responsabilità chiare
- **Dependency Injection**: Client API iniettati nei componenti

### Classi Principali

#### `Codes`
- Gestione dei client Mistral AI e Google GenAI
- Metodi asincroni per OCR e trascrizione audio
- Upload file e processing

#### `mistralAIWrapper`
- Implementazione Runnable per LangChain
- Metodi sincroni e asincroni
- Gestione dei messaggi in formato Mistral

#### `GeminiEmbeddings`
- Embeddings personalizzati con Google GenAI
- Supporto per batch embedding
- Metodi per documenti e query

### Flusso di Esecuzione
```
Utente → parse_command() → load_file() → generate_response() → run_assistant()
                          ↓
                   (OCR/Audio Transcription if needed)
                          ↓
                   (FAISS Vector Store if applicable)
```

## 🚀 Vantaggi Competitivi

### Estensibilità
- Facile aggiunta di nuovi formati file
- Architettura modulare per nuove funzionalità
- Integrazione semplice con nuovi tool

### Robustezza
- Gestione completa degli errori
- Fallback automatici (OCR per PDF)
- Validazione degli input

### User Experience
- Interfaccia chiara e intuitiva
- Feedback in tempo reale
- Messaggi di errore informativi

### Performance
- Operazioni asincrone non bloccanti
- Vector store efficienti con FAISS
- Processing parallelo dove possibile

## 📋 Requisiti

### Dipendenze Principali
```
mistralai>=1.12.3
google-genai==1.60.0
langchain-core>=1.2.5
langchain-community>=0.4.1
langchain-text-splitters>=1.1.0
faiss-cpu>=1.13.2
pypdf>=4.3.0
python-dotenv>=1.2.1
typing_extensions==4.15.0
httpx==0.28.1
gradio>=6.5.1
```

### Variabili d'Ambiente
```
MISTRAL_API_KEY=your_mistral_api_key
GEMINI_API_KEY=your_google_api_key
PROXY_URL=optional_proxy_url
```

## 🎯 Use Cases

### Personal Assistance
- Ricerca semantica in documenti
- Estrazione di informazioni da PDF
- Trascrizione di note vocali

### Developer Automation
- Analisi di log e documentazione
- Ricerca in codebase
- Gestione di file di progetto

### Content Processing
- Trascrizione di interviste e podcast
- Analisi di documenti PDF
- Estrazione di informazioni da file JSON

### Educational Tools
- Trascrizione di lezioni registrate
- Analisi di materiali didattici
- Ricerca in appunti e documenti

## 🔮 Roadmap Futura ChatBot

### Funzionalità Pianificate
- Supporto per file video
- Traduzione automatica delle trascrizioni
- Analisi del sentiment nei testi
- Integrazione con database esterni

### Miglioramenti
- Ottimizzazione delle performance
- Supporto per batch processing
- API REST per integrazione

## 📝 Note di Implementazione

### Best Practices Seguite
- Codice ben documentato e commentato
- Nomi di variabili e funzioni chiari
- Gestione degli errori completa
- Separazione delle responsabilità

### Compatibilità
- Python 3.10+
- Windows, Linux, macOS
- Architetture x86_64 e ARM

### Testing
- Test manuali completati
- Gestione degli errori verificata
- Fallback OCR testato
- Trascrizione audio non validata completamente

---

**Versione**: 2.0
**Data**: 21-02-2026