NLP Chatbot Project:
Step 1:
1. I downloaded the xml wikiquote dump file.
2. I extracted 30000 quotes, source, heading context, author, with relation like said by, misattributed to , disputed with or about.
3. I preprocessed the dataset and removed the quotes with length grater than 1000 characters.
4. I was left with 27799 records at the end with 6 columns.


Neo4j Graph Database:
1. I created Neo4j graph database, below are the details.
  Neo4j Graph Database:
  instance name = quotes_db
  Neo4j Version = 5.26.9  
  Database User = neo4j
  Password  = shri@1412
2. I created two node types and Quotes and Person
3. Quote node has properties id, text, source, status, target,context_heading.
4. Person has properties id and name.
5. I created 4 relationship types said by, about, misattributed to and disputed with.


Now I am going to create an API which fetches the Neo4j databases for the autocomplition of the quotes as a conclusion of the step one.
I created a basic GET API and saved with API.py
How to Run API.py:-
  Activate the Environment:-
    . .\.venv\Scripts\Activate.ps1
  set the variables:-
    $env:NEO4J_URI = "neo4j://127.0.0.1:7687"
    $env:NEO4J_USER = "neo4j"
    $env:NEO4J_PASSWORD = "shri@1412"
    $env:NEO4J_DATABASE = "neo4j"

Open CMD, move to the parent folder and run the below command:-
  uvicorn API:app --reload --port 8000

If any modification needed for API.py after the Step 2 then we will get back to API.py 
I am concluding the step 1.

Step 2:
created another environment for step2 using the anaconda ptompt using python 3.11
Activate the second environment by moving to step2 folder in anaconda:
  conda activate step2

Installed the dependencies mentioned in requirementsuirements.txt



now set up these environment variables in the venv step 2 throught prompt:

  STEP1_API_BASE=http://127.0.0.1:8000
  LLM_GGUF=C:\Users\shri\Data_Science\Text Mining\mistral-7b-instruct-v0.1.Q4_K_M.gguf
  ASR_MODEL=small
  SIM_THRESHOLD=0.75

using command:-

conda activate step2
conda env config vars set STEP1_API_BASE="http://127.0.0.1:8000" LLM_GGUF="C:\Users\shri\Data_Science\Text Mining\mistral-7b-instruct-v0.1.Q4_K_M.gguf" ASR_MODEL=small SIM_THRESHOLD=0.75
conda deactivate
conda activate step2


## requirements.txt

gradio==4.44.0
faster-whisper==1.0.3
numpy==1.26.4
soundfile==0.12.1
python-dotenv==1.0.1
requests==2.32.3
scikit-learn==1.5.1
webrtcvad==2.0.10s
edge-tts==6.1.11
pyttsx3==2.90
pydub==0.25.1
fastapi
uvicorn


Run the Project by the Command:
  python main.py

You are good to go. Enjoy...