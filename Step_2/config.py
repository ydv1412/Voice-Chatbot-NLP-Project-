import os
from dotenv import load_dotenv
load_dotenv(override=True)

# DEBUG
DEBUG = os.getenv("DEBUG_PRINT", "0") == "1"

# ASR
ASR_MODEL = os.getenv("ASR_MODEL", "small")

# Microphone
MIC_SAMPLE_RATE = int(os.getenv("MIC_SAMPLE_RATE", "16000"))
MIC_CHANNELS = int(os.getenv("MIC_CHANNELS", "1"))
MIC_BLOCK_MS = int(os.getenv("MIC_BLOCK_MS", "30"))        ## can change to 10/20
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "2"))
MAX_UTTERANCE_SECONDS = int(os.getenv("MAX_UTTERANCE_SECONDS", "25"))

# LLM 
LLM_GGUF = os.getenv("LLM_GGUF") 

# Neo4j 
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "******")     ## for security
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
NEO4J_FT_INDEX = os.getenv("NEO4J_FT_INDEX", "quoteTextFT")

# Speaker ID 
USE_SPK_ID = os.getenv("USE_SPK_ID", "1") == "1"
SPEAKER_DB_PATH = os.getenv("SPEAKER_DB_PATH", ".cache/speakers.json")
SPEAKER_ID_THRESHOLD = float(os.getenv("SPEAKER_ID_THRESHOLD", "0.65"))

# TTS(Default)
TTS_VOICE = os.getenv("TTS_VOICE", "")     
TTS_RATE = int(os.getenv("TTS_RATE", "185"))
TTS_VOLUME = float(os.getenv("TTS_VOLUME", "1.0"))
