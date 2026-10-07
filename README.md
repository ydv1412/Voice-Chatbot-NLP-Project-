#  Voice-Driven Quote Assistant

An end-to-end **voice-based conversational retrieval system** for identifying incomplete or partially remembered quotes and answering questions about their authors, sources, attribution, and context.

The system combines:

**Whisper ASR • Speaker Recognition • Neo4j Knowledge Graph • Lucene Full-Text Search • Hybrid Reranking • Mistral-7B • Text-to-Speech**

Unlike a purely generative chatbot, the assistant retrieves quote information from a structured knowledge base before answering, helping keep responses grounded in the underlying data.

---

##  Problem

People often remember only fragments of famous quotes:

> "I remember part of the quote, but who said it?"

or:

> "Finish the quote: Selfishness is not living..."

A generative LLM may complete the quote incorrectly or attribute it to the wrong person.

The goal of this project was therefore to build a **voice-driven quote assistant** that can:

- understand spoken requests;
- recognize the speaker;
- extract incomplete quote fragments;
- search a structured quote knowledge base;
- retrieve the most likely quote;
- answer questions about authors and sources;
- maintain conversational context;
- personalize the spoken response.

---

#  System Architecture

```text
                    🎙️ Microphone
                         │
                         ▼
                    WebRTC VAD
                         │
                ┌────────┴─────────┐
                ▼                  ▼
           Whisper ASR       SpeechBrain
                │            Speaker Encoder
                │                  │
                │             User Identity
                │                  │
                └────────┬─────────┘
                         ▼
               Language Understanding
                         │
                ┌────────┴────────┐
                │                 │
          Intent Detection   Quote Fragment
             (Regex)        Extraction (LLM)
                │                 │
                └────────┬────────┘
                         ▼
                    Intent Router
                         │
                         ▼
                Neo4j Quote Retriever
                         │
                ┌────────┴─────────┐
                │                  │
          Lucene Search       Graph Traversal
                │                  │
                └────────┬─────────┘
                         ▼
                     Reranking
                         │
                         ▼
                  Best Quote Match
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Deterministic Answer     Mistral-7B
              │                Answer Generation
              └──────────┬──────────┘
                         ▼
                 Personalized TTS
                         │
                         ▼
                    🔊 Response
```

---

#  Quote Knowledge Base

The quote collection was extracted from a **Wikiquote dump**.

The initial extraction produced approximately:

**30,000 quotes and associated metadata**

Pattern-based parsing was used to identify:

- Quote text
- Source
- Context
- Author/entity
- Attribution status

The data contained several issues including:

- missing quotes;
- duplicate records;
- fragmented author information;
- meaningless or very short sources;
- extremely long quotes;
- inconsistent Wikiquote templates.

After preprocessing and filtering, the final dataset contained:

> **27,799 cleaned quote records**

---

# 🕸️ Neo4j Knowledge Graph

Rather than storing the quote collection as flat documents, the system represents quotes and people using a **Neo4j graph database**.

## Nodes

### Quote

Stores information such as:

```text
text
source
heading_context
status
```

### Person

Represents the author or entity associated with the quote.

---

## Relationships

Quotes and people are connected through semantic relationships:

```text
(Quote)-[:SAID_BY]->(Person)

(Quote)-[:ABOUT]->(Person)

(Quote)-[:MISATTRIBUTED_TO]->(Person)

(Quote)-[:DISPUTED_WITH]->(Person)
```

This allows the system to distinguish between:

> "Who actually said this?"

and

> "Who is this quote about?"

or whether a quote is disputed or commonly misattributed.

---

#  Hybrid Quote Retrieval

The retrieval system combines **full-text search with graph traversal and custom reranking**.

## Step 1 — Lucene Search

A Lucene-based Neo4j full-text index:

```text
quoteTextFT
```

is used to search quote text.

Multiple matching strategies are used to handle incomplete or imperfect fragments:

- exact matching;
- token-based matching;
- fuzzy matching.

This makes retrieval more robust to incomplete phrases and ASR errors.

---

## Step 2 — Graph Enrichment

Candidate quotes are enriched by traversing their graph relationships.

For each candidate, the system retrieves:

- quote text;
- author;
- source;
- attribution status;
- related people;
- semantic relationships.

---

## Step 3 — Reranking

Lucene score alone was not reliable enough for short quote fragments.

A custom reranking score was therefore introduced:

```text
final_score =
    0.55 × token_coverage
  + 0.35 × normalized_fulltext_score
  + 0.10 × phrase_bonus
```

The candidate with the highest final score is selected.

This combines:

**lexical coverage + search relevance + phrase-level matching**

instead of relying exclusively on the Lucene ranking.

---

#  Language Understanding

The assistant separates **what the user is asking** from **which quote they are referring to**.

> **Fragment decides the search. Intent decides the action.**

---

## Quote Fragment Extraction

Mistral-7B is used as a span extractor to identify only the relevant quote fragment from the spoken request.

For example:

```text
User:
"Can you tell me who said the quote
selfishness is not living as one wishes?"

                ↓

Extracted fragment:

"selfishness is not living as one wishes"
```

Command words and author names are removed before retrieval.

At least three meaningful quote words are required before performing the search.

---

#  Intent Detection & Routing

Structured intents are detected using deterministic rules rather than sending every request through an LLM.

Examples include:

```text
"who said..."
"who wrote..."
        ↓
     said_by


"finish it..."
"complete the quote..."
        ↓
   finish_quote


"where is it from?"
"what is the source?"
        ↓
      source


"who is it about?"
        ↓
       about
```

This design keeps common operations **fast and deterministic**.

The router then decides whether to:

- execute a database search;
- answer a control command;
- use previous conversation context;
- request additional quote words.

---

#  Multi-Turn Conversation

The assistant maintains quote context between turns.

For example:

```text
User:
"Find the quote 'I resolved from the beginning...'"

Assistant:
[retrieves quote]

User:
"Who said it?"

Assistant:
[uses previous quote context]

User:
"What is the source?"

Assistant:
[answers from the same quote]
```

Each recognized speaker maintains their own context.

This prevents follow-up questions from different users from interfering with each other's conversations.

---

#  Speaker Recognition

Speaker recognition was implemented using **SpeechBrain**.

## Registration

When a user says:

```text
"register"
```

the enrollment procedure begins.

The user records **five short readings**.

SpeechBrain generates a:

**192-dimensional speaker embedding**

for each recording.

A centroid embedding is calculated and stored for the user.

---

## Identification

For subsequent requests, the incoming speaker embedding is compared against stored user centroids using **cosine similarity**.

The system accepts a speaker match when:

```text
similarity ≥ 0.65
duration ≥ 0.8 seconds
```

When a speaker is recognized, the assistant automatically loads that user's:

- conversation session;
- last quote context;
- TTS preferences.

---

#  Voice Pipeline

The complete voice interaction pipeline consists of:

```text
Microphone
    ↓
WebRTC Voice Activity Detection
    ↓
Whisper Speech Recognition
    ↓
Speaker Identification
    ↓
Intent + Quote Understanding
    ↓
Quote Retrieval
    ↓
Answer Generation
    ↓
Text-to-Speech
```

### Voice Activity Detection

WebRTC VAD controls microphone recording.

Recording stops after approximately:

**500 ms of silence**

with a maximum recording duration of:

**25 seconds**

---

#  Personalized Text-to-Speech

The system supports per-user voice preferences.

Users can issue commands such as:

```text
"Set my voice to David"

"Set my voice to Zira"

"Set my rate to slow"

"Make it louder"

"Make it quieter"
```

Preferences are stored in JSON files and automatically restored when the speaker is recognized.

The final implementation uses **pyttsx3** for lightweight local speech synthesis.

---

#  Two Answer Generation Modes

The project explores two different approaches to answer generation.

##  Deterministic Mode

For structured questions, the assistant can generate answers directly from graph fields.

Examples:

```text
"Who said it?"
→ return SAID_BY

"What's the source?"
→ return source

"Finish the quote"
→ return complete quote

"Find the quote"
→ quote + author + source
```

This approach is extremely fast and does not require generative inference for the final response.

---

##  LLM Answer Mode

An optional LLM answering mode can be enabled using:

```python
USE_LLM_ANSWERS = 1
```

The reranked database results are passed to **Mistral-7B**.

The system prompt instructs the model to use only the supplied database results.

This allows raw structured results such as:

```text
Albert Einstein
Source: Unknown
```

to be transformed into a more conversational response while keeping the answer tied to retrieved graph information.

---

#  Evaluation

The evaluation focused primarily on the custom **retrieval and answering pipeline**.

## Retrieval Performance

| Metric | Result |
|---|---:|
| **Top-1 Accuracy** | **93.7%** |
| **Top-5 Recall** | **96.8%** |
| **MRR@5** | **0.95** |
| **Median Retrieval Latency** | **22 ms** |

The results show that the hybrid retrieval and reranking pipeline can identify the correct quote from incomplete fragments with high accuracy.

---

#  Speaker Recognition Testing

Speaker recognition was tested with:

**5 different speakers**

The system performed reliably when:

- speech samples were sufficiently long;
- enrollment and testing environments were similar.

Performance degraded for:

- very short speech;
- recordings made in substantially different environments.

---

#  Key Engineering Decisions

Several design decisions were made during development.

### 1. Retrieval instead of relying on LLM memory

Exact quotes and attribution information are retrieved from the knowledge graph rather than generated from model memory.

### 2. Hybrid retrieval instead of Lucene score alone

Short quote fragments made pure full-text ranking unreliable.

Combining token coverage, full-text score and phrase matching improved candidate selection.

### 3. Deterministic routing for structured commands

Using an LLM for every routing decision introduced unnecessary latency on local hardware.

Regex-based routing was therefore used for common structured intents.

### 4. LLMs where they add value

Mistral-7B is used selectively for:

- quote-fragment extraction;
- optional natural-language answer generation.

This creates a hybrid system combining deterministic components with generative AI.

---

#  Tech Stack

### Speech

- Whisper
- WebRTC VAD
- SpeechBrain
- pyttsx3

### NLP / LLM

- Mistral-7B
- Regex-based intent routing

### Retrieval

- Lucene Full-Text Search
- Token Coverage
- Fuzzy Matching
- Custom Reranking

### Knowledge Graph

- Neo4j
- Cypher

### Data

- Wikiquote
- Python
- Pandas
- JSON

---

#  End-to-End Example

```text
🎙️ User:
"Who said the quote selfishness is not living as one wishes?"

                     ↓

Whisper ASR
                     ↓

Intent:
said_by

Quote Fragment:
"selfishness is not living as one wishes"

                     ↓

Lucene + Neo4j Retrieval
                     ↓

Candidate Reranking
                     ↓

Best Matching Quote
                     ↓

Graph Relationship:
SAID_BY

                     ↓

🔊 Assistant:
[Author retrieved from the knowledge graph]
```

The same retrieved quote can then support follow-up questions such as:

```text
"Finish the quote."

"What is the source?"

"Repeat it."

"Who is it about?"
```

without requiring the user to repeat the quote fragment.

---

#  Limitations

The current implementation has several limitations:

- very short quote fragments may not contain enough information for reliable retrieval;
- Wikiquote contains incomplete or noisy source information;
- regex-based intent routing requires manually defined patterns;
- speaker recognition can degrade across different recording environments;
- local Mistral-7B inference introduces additional latency;
- lightweight local TTS provides limited voice personalization.

---

#  Future Work

Potential improvements include:

- LLM-based intent routing as inference becomes faster;
- more flexible natural-language query understanding;
- improved fuzzy retrieval for very short fragments;
- learned reranking models;
- better speaker verification across recording environments;
- neural personalized TTS;
- larger and cleaner quote knowledge bases;
- support for semantic queries such as:

```text
"Tell me a quote about Einstein."
```

---

#  Project Presentation

A detailed presentation covering the complete architecture, quote extraction, Neo4j graph design, voice interface, retrieval pipeline, speaker recognition and evaluation is available in this repository.

👉 **[View Project Presentation](./Voice_Quote_Assistant_Presentation.pdf)**

---

#  Demo


>  **Project demo: Coming soon**

---

#  Author

**Shri Prakash Yadav**

M.Sc. Data Science  
University of Naples Federico II
