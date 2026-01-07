# Aurora LLM Integration with Soundstar Tri-Hybrid Engine

## Technical Specification Document

**Author**: Manus AI  
**Version**: 2.0  
**Date**: January 7, 2026  
**Status**: Final

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Aurora LLM Architecture](#3-aurora-llm-architecture)
4. [Codex as Implementation Tool](#4-codex-as-implementation-tool)
5. [API Endpoints Design](#5-api-endpoints-design)
6. [Natural Language to Musical Parameters Workflow](#6-natural-language-to-musical-parameters-workflow)
7. [Data Models and Schemas](#7-data-models-and-schemas)
8. [Real-Time Prompt Refinement Implementation](#8-real-time-prompt-refinement-implementation)
9. [Error Handling and Fallback Strategies](#9-error-handling-and-fallback-strategies)
10. [Performance Optimization Strategies](#10-performance-optimization-strategies)
11. [Appendices](#11-appendices)

---

## 1. Executive Summary

This technical specification document outlines the architecture of **Aurora LLM** as the primary semantic intelligence layer for the **Soundstar Tri-Hybrid Engine**, with **GPT Codex** serving as the underlying implementation tool. The primary objective is to enable **natural language-driven music generation** with **intelligent prompt enhancement**, allowing users to describe their musical vision in plain English and receive high-fidelity, semantically coherent audio output.

### Key Architectural Distinction

| Component | Role | Description |
|-----------|------|-------------|
| **Aurora LLM** | **Primary Architecture** | The semantic intelligence layer that defines the music understanding pipeline, orchestrates all NLP operations, and provides the unified interface for the Soundstar system |
| **GPT Codex** | **Implementation Tool** | The underlying engine that Aurora LLM uses internally to perform specific NLP tasks such as prompt parsing, enhancement, and structured output generation |

This separation of concerns provides several advantages:

1. **Abstraction**: The Soundstar system interacts only with Aurora LLM, not directly with Codex, allowing for future backend swaps.
2. **Customization**: Aurora LLM can apply music-specific fine-tuning, prompt engineering, and post-processing on top of Codex outputs.
3. **Resilience**: Aurora LLM can implement fallback strategies when Codex is unavailable.
4. **Consistency**: Aurora LLM ensures consistent behavior and output formats regardless of the underlying implementation.

---

## 2. Architecture Overview

### 2.1 High-Level System Architecture

Aurora LLM serves as the **Semantic Intelligence Layer** that sits between the user interface and the Soundstar Tri-Hybrid Engine's Control Encoder. Internally, Aurora LLM leverages GPT Codex as its primary implementation tool for natural language processing tasks.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│  (Web UI / API Client)                                                       │
│  - Natural Language Input                                                    │
│  - Real-time Prompt Suggestions                                              │
│  - Parameter Refinement Controls                                             │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AURORA LLM                                      │
│                    (Primary Semantic Intelligence Layer)                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    AURORA ORCHESTRATION LAYER                        │    │
│  │  - Request Routing & Load Balancing                                  │    │
│  │  - Caching & Optimization                                            │    │
│  │  - Fallback Management                                               │    │
│  │  - Output Normalization                                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    AURORA PROCESSING MODULES                         │    │
│  │                                                                       │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │   Prompt    │  │  Semantic   │  │  Parameter  │  │  Structural │ │    │
│  │  │  Enhancer   │  │   Parser    │  │   Mapper    │  │   Planner   │ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │    │
│  │         │                │                │                │        │    │
│  │         └────────────────┴────────────────┴────────────────┘        │    │
│  │                                    │                                 │    │
│  │                                    ▼                                 │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │              GPT CODEX (Implementation Tool)                 │    │    │
│  │  │  - Natural Language Understanding                            │    │    │
│  │  │  - Structured Output Generation                              │    │    │
│  │  │  - Context-Aware Completion                                  │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    UNIFIED CONTROL EMBEDDING                         │    │
│  │  - Genre, Tempo, Mood, Instrumentation                               │    │
│  │  - Structural Form (AABA, Verse/Chorus, etc.)                        │    │
│  │  - f0, Loudness, Harmonic Mix Parameters                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MUSICAI TRI-HYBRID ENGINE                              │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐                │
│  │  Structural   │───▶│  RAVE Codec   │───▶│    DDSP       │                │
│  │  Transformer  │    │  (Latent)     │    │  Refinement   │                │
│  └───────────────┘    └───────────────┘    └───────────────┘                │
│                                    │                                         │
│                                    ▼                                         │
│                          ┌───────────────┐                                   │
│                          │   Ultimate    │                                   │
│                          │    Mixer      │                                   │
│                          └───────────────┘                                   │
│                                    │                                         │
│                                    ▼                                         │
│                          ┌───────────────┐                                   │
│                          │  Final Audio  │                                   │
│                          │   Output      │                                   │
│                          └───────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | Layer | Responsibility | Input | Output |
|-----------|-------|----------------|-------|--------|
| **Aurora LLM** | Primary | Orchestrates all semantic processing, manages Codex calls, applies music-specific logic | Raw user text prompt | Unified Control Embedding |
| **GPT Codex** | Implementation | Performs NLP tasks as directed by Aurora LLM | Aurora-formatted requests | Raw NLP outputs |
| **Control Encoder** | Processing | Converts Aurora outputs to numerical embeddings | Structured parameters | `control_embedding` tensor |
| **Structural Transformer** | Generation | Generates RAVE latent vector from control embedding | `control_embedding` | `z_rave` (latent vector) |
| **RAVE Codec** | Synthesis | Decodes latent vector to high-fidelity audio waveform | `z_rave` | `w_rave` (audio waveform) |
| **DDSP Refinement Net** | Refinement | Provides interpretable timbre control | `z_rave` | `w_ddsp` (refined audio) |
| **Ultimate Mixer** | Output | Blends RAVE and DDSP outputs | `w_rave`, `w_ddsp` | `w_final` (final audio) |

---

## 3. Aurora LLM Architecture

### 3.1 Aurora LLM Core Design

Aurora LLM is designed as a **modular, music-specialized semantic intelligence system**. It provides a unified interface for all natural language processing needs within Soundstar, abstracting away the complexity of the underlying implementation tools.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AURORA LLM                                      │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         AURORA CORE                                  │    │
│  │                                                                       │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │    │
│  │  │   Request   │  │   Music     │  │   Output    │                   │    │
│  │  │   Router    │  │  Knowledge  │  │  Validator  │                   │    │
│  │  │             │  │    Base     │  │             │                   │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │    │
│  │                                                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    AURORA PROCESSING MODULES                         │    │
│  │                                                                       │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  PROMPT ENHANCER MODULE                                      │    │    │
│  │  │  - Adds musical context to raw prompts                       │    │    │
│  │  │  - Suggests instrumentation, tempo, mood                     │    │    │
│  │  │  - Resolves ambiguities                                      │    │    │
│  │  │  [Implementation: Codex with music-specific system prompt]   │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                                                                       │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  SEMANTIC PARSER MODULE                                      │    │    │
│  │  │  - Extracts musical entities (genre, mood, tempo, etc.)      │    │    │
│  │  │  - Identifies reference artists/songs                        │    │    │
│  │  │  - Detects structural requirements                           │    │    │
│  │  │  [Implementation: Codex with entity extraction prompt]       │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                                                                       │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  PARAMETER MAPPER MODULE                                     │    │    │
│  │  │  - Maps extracted entities to Soundstar schema                 │    │    │
│  │  │  - Applies music theory rules                                │    │    │
│  │  │  - Validates parameter coherence                             │    │    │
│  │  │  [Implementation: Rule-based + Codex validation]             │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                                                                       │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  STRUCTURAL PLANNER MODULE                                   │    │    │
│  │  │  - Generates section-by-section structure                    │    │    │
│  │  │  - Plans energy curves and transitions                       │    │    │
│  │  │  - Suggests chord progressions                               │    │    │
│  │  │  [Implementation: Codex with music structure prompt]         │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                                                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    IMPLEMENTATION LAYER                              │    │
│  │                                                                       │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │    │
│  │  │ GPT Codex   │  │  Local LLM  │  │ Rule-Based  │                   │    │
│  │  │ (Primary)   │  │  (Backup)   │  │  (Fallback) │                   │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │    │
│  │                                                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Aurora LLM Modules

#### 3.2.1 Prompt Enhancer Module

The Prompt Enhancer Module takes raw user input and enriches it with musical context, making it more suitable for downstream processing.

**Responsibilities:**
- Add missing musical context (tempo, key, time signature)
- Suggest appropriate instrumentation based on genre
- Resolve ambiguous descriptions
- Expand abbreviated or colloquial terms

**Implementation Strategy:**
Aurora LLM uses Codex with a music-specific system prompt to perform enhancement. The system prompt is maintained by Aurora and includes:
- Music production terminology
- Genre-specific conventions
- Common user intent patterns

#### 3.2.2 Semantic Parser Module

The Semantic Parser Module extracts structured musical entities from natural language descriptions.

**Responsibilities:**
- Identify genre and sub-genre
- Extract tempo (explicit or inferred)
- Detect mood and emotional qualities
- Recognize instrumentation mentions
- Identify reference artists or songs

**Implementation Strategy:**
Aurora LLM uses Codex with an entity extraction prompt that outputs structured JSON. Aurora then validates and normalizes the output.

#### 3.2.3 Parameter Mapper Module

The Parameter Mapper Module converts extracted entities into Soundstar's internal parameter schema.

**Responsibilities:**
- Map genre names to genre embeddings
- Normalize tempo to [0, 1] range
- Convert mood descriptors to mood embeddings
- Encode instrumentation as multi-hot vectors
- Apply music theory rules for coherence

**Implementation Strategy:**
This module uses a combination of rule-based mapping (for well-defined conversions) and Codex validation (for edge cases and coherence checking).

#### 3.2.4 Structural Planner Module

The Structural Planner Module generates detailed structural plans for the music to be generated.

**Responsibilities:**
- Define section structure (intro, verse, chorus, etc.)
- Plan energy curves across sections
- Suggest transitions between sections
- Recommend chord progressions

**Implementation Strategy:**
Aurora LLM uses Codex with a music structure prompt that incorporates music theory knowledge. The output is validated against structural templates.

### 3.3 Aurora LLM Music Knowledge Base

Aurora LLM maintains an internal **Music Knowledge Base** that provides context for all processing modules. This knowledge base includes:

| Category | Contents |
|----------|----------|
| **Genre Taxonomy** | Hierarchical genre classification with parent-child relationships |
| **Tempo Ranges** | Typical BPM ranges for each genre |
| **Instrumentation Profiles** | Common instruments for each genre |
| **Mood Mappings** | Associations between mood descriptors and musical characteristics |
| **Structural Templates** | Common song structures for each genre |
| **Artist Style Profiles** | Characteristic features of popular artists for style transfer |

---

## 4. Codex as Implementation Tool

### 4.1 Codex Integration Architecture

GPT Codex serves as the primary implementation tool for Aurora LLM's natural language processing tasks. Aurora LLM abstracts Codex behind a unified interface, allowing for:

1. **Prompt Engineering**: Aurora applies music-specific prompt templates before sending to Codex.
2. **Output Post-Processing**: Aurora validates and normalizes Codex outputs.
3. **Error Handling**: Aurora manages Codex errors and implements fallbacks.
4. **Caching**: Aurora caches Codex responses to reduce latency and costs.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AURORA → CODEX INTEGRATION                                │
│                                                                              │
│  Aurora LLM                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  1. Receive user request                                             │    │
│  │  2. Select appropriate module (Enhancer, Parser, etc.)               │    │
│  │  3. Prepare Codex request with music-specific prompt                 │    │
│  │  4. Check cache for existing response                                │    │
│  │  5. If cache miss, send request to Codex                             │    │
│  │  6. Receive Codex response                                           │    │
│  │  7. Validate and normalize response                                  │    │
│  │  8. Apply music-specific post-processing                             │    │
│  │  9. Cache response for future use                                    │    │
│  │  10. Return processed result                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    CODEX INTERFACE LAYER                             │    │
│  │                                                                       │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │    │
│  │  │   Prompt    │  │   Request   │  │  Response   │                   │    │
│  │  │  Templates  │  │   Builder   │  │   Parser    │                   │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │    │
│  │                                                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    GPT CODEX API                                     │    │
│  │                    (OpenAI Service)                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Codex Request Templates

Aurora LLM uses specialized prompt templates for each processing module. These templates are designed to elicit music-specific responses from Codex.

#### 4.2.1 Prompt Enhancement Template

```
SYSTEM: You are Aurora, a music production AI assistant. Your task is to enhance user prompts by adding musical context while preserving the user's original intent.

INSTRUCTIONS:
1. Analyze the user's prompt for musical intent
2. Add specific details that are implied but not stated (tempo, key, instrumentation)
3. Expand colloquial or abbreviated terms
4. Ensure the enhanced prompt is clear and actionable for music generation
5. Do not change the fundamental character of the requested music

USER PROMPT: {{user_prompt}}

OUTPUT FORMAT:
{
  "enhanced_prompt": "string",
  "additions": [
    {"type": "string", "value": "string", "reasoning": "string"}
  ],
  "confidence": number
}
```

#### 4.2.2 Semantic Parsing Template

```
SYSTEM: You are Aurora, a music semantic parser. Extract structured musical parameters from the given description.

INSTRUCTIONS:
1. Identify all musical entities in the prompt
2. For each entity, provide the extracted value and confidence
3. If an entity is not explicitly mentioned, infer it from context or mark as "unspecified"
4. Use standard music terminology

USER PROMPT: {{user_prompt}}

OUTPUT FORMAT:
{
  "genre": {"value": "string", "confidence": number},
  "sub_genre": {"value": "string | null", "confidence": number},
  "tempo": {"type": "explicit | inferred", "bpm": number, "confidence": number},
  "mood": [{"value": "string", "confidence": number}],
  "instrumentation": [{"name": "string", "role": "string", "confidence": number}],
  "structure": {"value": "string", "confidence": number},
  "key_signature": {"value": "string | null", "confidence": number},
  "time_signature": {"value": "string", "confidence": number},
  "energy_level": {"value": number, "confidence": number},
  "reference_artists": [{"name": "string", "relevance": number}]
}
```

#### 4.2.3 Structural Planning Template

```
SYSTEM: You are Aurora, a music structure planner. Generate a detailed structural plan for the requested music.

INSTRUCTIONS:
1. Based on the genre and style, determine an appropriate song structure
2. Define sections with their relative durations and energy levels
3. Plan transitions between sections
4. Suggest a chord progression that fits the mood and genre
5. Ensure the structure is musically coherent

PARAMETERS:
- Genre: {{genre}}
- Mood: {{mood}}
- Tempo: {{tempo}} BPM
- Duration: {{duration}} seconds

OUTPUT FORMAT:
{
  "structure_type": "string (e.g., 'Verse-Chorus', 'AABA', 'Through-Composed')",
  "sections": [
    {
      "name": "string",
      "duration_ratio": number,
      "energy_level": number,
      "description": "string"
    }
  ],
  "transitions": [
    {
      "from": "string",
      "to": "string",
      "type": "string (e.g., 'build', 'drop', 'fade', 'cut')"
    }
  ],
  "chord_progression": {
    "key": "string",
    "mode": "string",
    "chords": ["string"]
  }
}
```

### 4.3 Codex Response Processing

Aurora LLM applies the following post-processing steps to Codex responses:

1. **JSON Validation**: Ensure the response is valid JSON matching the expected schema.
2. **Value Normalization**: Normalize values to Soundstar's internal formats (e.g., tempo to [0, 1]).
3. **Confidence Thresholding**: Flag low-confidence values for user review.
4. **Music Theory Validation**: Check for musically coherent combinations (e.g., tempo-genre compatibility).
5. **Embedding Generation**: Convert categorical values to numerical embeddings.

---

## 5. API Endpoints Design

All API endpoints are exposed through Aurora LLM's unified interface. The endpoints are prefixed with `/api/v1/aurora`.

### 5.1 Endpoint Summary

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| `POST` | `/api/v1/aurora/enhance` | Enhance a raw user prompt with musical context | Yes |
| `POST` | `/api/v1/aurora/suggest` | Get real-time suggestions as user types | Yes |
| `POST` | `/api/v1/aurora/parse` | Parse prompt into structured musical parameters | Yes |
| `POST` | `/api/v1/aurora/plan` | Generate structural plan for music | Yes |
| `POST` | `/api/v1/aurora/generate` | Full pipeline: enhance, parse, plan, and generate audio | Yes |
| `GET`  | `/api/v1/aurora/history` | Retrieve user's generation history | Yes |
| `POST` | `/api/v1/aurora/feedback` | Submit feedback on Aurora suggestions | Yes |

### 5.2 Endpoint Specifications

#### 5.2.1 `POST /api/v1/aurora/enhance`

Enhances a raw user prompt by adding musical context.

**Request Body:**

```json
{
  "prompt": "string (required) - The raw natural language prompt from the user",
  "context": {
    "genre_hint": "string (optional) - A hint about the desired genre",
    "mood_hint": "string (optional) - A hint about the desired mood",
    "duration_hint": "number (optional) - Desired duration in seconds"
  },
  "enhancement_level": "string (optional) - 'minimal', 'moderate', 'aggressive'. Default: 'moderate'"
}
```

**Response Body (Success - 200 OK):**

```json
{
  "original_prompt": "string",
  "enhanced_prompt": "string",
  "additions": [
    {
      "type": "string - 'genre', 'instrumentation', 'mood', 'structure', 'tempo'",
      "value": "string",
      "reasoning": "string",
      "confidence": "number (0.0 - 1.0)"
    }
  ],
  "aurora_metadata": {
    "module": "prompt_enhancer",
    "implementation": "codex",
    "processing_time_ms": "number",
    "cache_hit": "boolean"
  }
}
```

#### 5.2.2 `POST /api/v1/aurora/parse`

Parses a prompt into structured musical parameters.

**Request Body:**

```json
{
  "prompt": "string (required) - The prompt to parse (can be raw or enhanced)"
}
```

**Response Body (Success - 200 OK):**

```json
{
  "parsed_parameters": {
    "genre": {"value": "string", "confidence": "number"},
    "sub_genre": {"value": "string | null", "confidence": "number"},
    "tempo": {"type": "string", "bpm": "number", "confidence": "number"},
    "mood": [{"value": "string", "confidence": "number"}],
    "instrumentation": [{"name": "string", "role": "string", "confidence": "number"}],
    "structure": {"value": "string", "confidence": "number"},
    "key_signature": {"value": "string | null", "confidence": "number"},
    "time_signature": {"value": "string", "confidence": "number"},
    "energy_level": {"value": "number", "confidence": "number"},
    "reference_artists": [{"name": "string", "relevance": "number"}]
  },
  "control_encoder_params": {
    "genre_embedding": "[number]",
    "tempo_normalized": "number",
    "mood_embedding": "[number]",
    "instrumentation_embedding": "[number]",
    "structure_embedding": "[number]",
    "energy_level": "number"
  },
  "aurora_metadata": {
    "module": "semantic_parser",
    "implementation": "codex",
    "processing_time_ms": "number"
  }
}
```

#### 5.2.3 `POST /api/v1/aurora/plan`

Generates a structural plan for the music.

**Request Body:**

```json
{
  "parameters": {
    "genre": "string (required)",
    "mood": "string (required)",
    "tempo_bpm": "number (required)",
    "duration_seconds": "number (required)"
  }
}
```

**Response Body (Success - 200 OK):**

```json
{
  "structural_plan": {
    "structure_type": "string",
    "sections": [
      {
        "name": "string",
        "duration_ratio": "number",
        "energy_level": "number",
        "description": "string"
      }
    ],
    "transitions": [
      {
        "from": "string",
        "to": "string",
        "type": "string"
      }
    ],
    "chord_progression": {
      "key": "string",
      "mode": "string",
      "chords": ["string"]
    }
  },
  "aurora_metadata": {
    "module": "structural_planner",
    "implementation": "codex",
    "processing_time_ms": "number"
  }
}
```

#### 5.2.4 `POST /api/v1/aurora/generate`

Full pipeline: enhance, parse, plan, and generate audio.

**Request Body:**

```json
{
  "prompt": "string (required) - The raw natural language prompt",
  "options": {
    "enhance_prompt": "boolean (optional) - Whether to enhance the prompt. Default: true",
    "duration_seconds": "number (optional) - Desired audio duration. Default: 30",
    "output_format": "string (optional) - 'wav', 'mp3', 'flac'. Default: 'wav'",
    "sample_rate": "number (optional) - Output sample rate. Default: 44100"
  }
}
```

**Response Body (Success - 200 OK):**

```json
{
  "generation_id": "string",
  "status": "string - 'completed', 'processing', 'queued'",
  "audio_url": "string",
  "enhanced_prompt": "string",
  "parsed_parameters": { ... },
  "structural_plan": { ... },
  "generation_metadata": {
    "total_time_ms": "number",
    "aurora_enhance_time_ms": "number",
    "aurora_parse_time_ms": "number",
    "aurora_plan_time_ms": "number",
    "engine_time_ms": "number",
    "implementation_used": "codex | local_llm | rule_based"
  }
}
```

---

## 6. Natural Language to Musical Parameters Workflow

### 6.1 Workflow Overview

Aurora LLM orchestrates a five-stage pipeline for translating natural language to musical parameters:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AURORA LLM PROCESSING PIPELINE                            │
│                                                                              │
│  Stage 1: Input Preprocessing                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  - Normalize text (lowercase, whitespace)                            │    │
│  │  - Spell correction for music terms                                  │    │
│  │  - Abbreviation expansion                                            │    │
│  │  - Content moderation                                                │    │
│  │  [Implementation: Rule-based]                                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  Stage 2: Prompt Enhancement                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  - Add musical context                                               │    │
│  │  - Suggest missing parameters                                        │    │
│  │  - Resolve ambiguities                                               │    │
│  │  [Implementation: Codex via Prompt Enhancer Module]                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  Stage 3: Semantic Parsing                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  - Extract musical entities                                          │    │
│  │  - Identify references and style cues                                │    │
│  │  - Assign confidence scores                                          │    │
│  │  [Implementation: Codex via Semantic Parser Module]                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  Stage 4: Parameter Mapping                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  - Map to Soundstar schema                                             │    │
│  │  - Apply music theory rules                                          │    │
│  │  - Validate coherence                                                │    │
│  │  [Implementation: Rule-based + Codex validation]                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  Stage 5: Embedding Generation                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  - Convert to numerical embeddings                                   │    │
│  │  - Generate Unified Control Embedding                                │    │
│  │  [Implementation: Pre-trained embedding models]                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│                    UNIFIED CONTROL EMBEDDING                                 │
│                    (Ready for Tri-Hybrid Engine)                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Stage Details

#### Stage 1: Input Preprocessing (Rule-Based)

| Step | Description | Example |
|------|-------------|---------|
| **Normalization** | Convert to lowercase, remove excessive whitespace | "  A FAST  rock song " → "a fast rock song" |
| **Spell Correction** | Correct common misspellings of musical terms | "bossa noav" → "bossa nova" |
| **Abbreviation Expansion** | Expand common abbreviations | "EDM" → "Electronic Dance Music" |
| **Content Moderation** | Remove or flag inappropriate content | (Content filtering) |

#### Stage 2: Prompt Enhancement (Codex)

Aurora LLM sends the preprocessed prompt to Codex using the Prompt Enhancement Template. The response includes:
- Enhanced prompt with added context
- List of additions with reasoning
- Confidence score

#### Stage 3: Semantic Parsing (Codex)

Aurora LLM sends the enhanced prompt to Codex using the Semantic Parsing Template. The response includes:
- Extracted musical entities
- Confidence scores for each entity
- Reference artists/songs

#### Stage 4: Parameter Mapping (Rule-Based + Codex)

Aurora LLM maps extracted entities to Soundstar's schema:

| Extracted Entity | Soundstar Parameter | Mapping Logic |
|------------------|-------------------|---------------|
| `genre` | `genre_embedding` | Lookup in genre taxonomy |
| `tempo.bpm` | `tempo_normalized` | `(bpm - 60) / (200 - 60)` |
| `mood` | `mood_embedding` | Average of mood descriptor embeddings |
| `instrumentation` | `instrumentation_embedding` | Multi-hot encoding |
| `structure` | `structure_embedding` | Lookup in structure taxonomy |
| `energy_level` | `energy_level` | Direct mapping to [0, 1] |

#### Stage 5: Embedding Generation (Pre-trained Models)

Aurora LLM generates the Unified Control Embedding:

```
unified_control_embedding = Concat(
    genre_embedding,          # dim: 128
    tempo_embedding,          # dim: 32
    mood_embedding,           # dim: 128
    instrumentation_embedding,# dim: 256
    structure_embedding,      # dim: 64
    energy_embedding          # dim: 32
)
# Total dimension: 640
```

---

## 7. Data Models and Schemas

### 7.1 Database Schema

```sql
-- Aurora Sessions
CREATE TABLE aurora_sessions (
    id VARCHAR(36) PRIMARY KEY,
    user_id INT NOT NULL,
    original_prompt TEXT NOT NULL,
    enhanced_prompt TEXT,
    parsed_parameters JSON,
    structural_plan JSON,
    control_embedding BLOB,
    status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
    implementation_used ENUM('codex', 'local_llm', 'rule_based') DEFAULT 'codex',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Aurora Suggestions
CREATE TABLE aurora_suggestions (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    module ENUM('prompt_enhancer', 'semantic_parser', 'parameter_mapper', 'structural_planner') NOT NULL,
    suggestion_type VARCHAR(64) NOT NULL,
    original_value TEXT,
    suggested_value TEXT NOT NULL,
    confidence DECIMAL(3, 2) NOT NULL,
    reasoning TEXT,
    user_accepted BOOLEAN DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES aurora_sessions(id)
);

-- Generation Results
CREATE TABLE generation_results (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    audio_url TEXT NOT NULL,
    audio_duration_seconds DECIMAL(10, 2) NOT NULL,
    sample_rate INT NOT NULL,
    output_format VARCHAR(16) NOT NULL,
    total_time_ms INT NOT NULL,
    aurora_time_ms INT,
    engine_time_ms INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES aurora_sessions(id)
);

-- User Feedback
CREATE TABLE user_feedback (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    result_id VARCHAR(36),
    feedback_type ENUM('positive', 'negative', 'neutral') NOT NULL,
    prompt_quality_rating TINYINT,
    suggestion_relevance_rating TINYINT,
    audio_quality_rating TINYINT,
    comments TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES aurora_sessions(id),
    FOREIGN KEY (result_id) REFERENCES generation_results(id)
);

-- Indexes
CREATE INDEX idx_aurora_sessions_user_id ON aurora_sessions(user_id);
CREATE INDEX idx_aurora_sessions_status ON aurora_sessions(status);
CREATE INDEX idx_aurora_suggestions_session_id ON aurora_suggestions(session_id);
```

### 7.2 TypeScript Type Definitions

```typescript
// aurora.types.ts

export interface AuroraSession {
  id: string;
  userId: number;
  originalPrompt: string;
  enhancedPrompt: string | null;
  parsedParameters: ParsedMusicalParameters | null;
  structuralPlan: StructuralPlan | null;
  controlEmbedding: number[] | null;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  implementationUsed: 'codex' | 'local_llm' | 'rule_based';
  createdAt: Date;
  updatedAt: Date;
}

export interface ParsedMusicalParameters {
  genre: { value: string; confidence: number };
  subGenre: { value: string | null; confidence: number };
  tempo: { type: 'explicit' | 'inferred'; bpm: number; confidence: number };
  mood: Array<{ value: string; confidence: number }>;
  instrumentation: Array<{ name: string; role: string; confidence: number }>;
  structure: { value: string; confidence: number };
  keySignature: { value: string | null; confidence: number };
  timeSignature: { value: string; confidence: number };
  energyLevel: { value: number; confidence: number };
  referenceArtists: Array<{ name: string; relevance: number }>;
}

export interface StructuralPlan {
  structureType: string;
  sections: Array<{
    name: string;
    durationRatio: number;
    energyLevel: number;
    description: string;
  }>;
  transitions: Array<{
    from: string;
    to: string;
    type: string;
  }>;
  chordProgression: {
    key: string;
    mode: string;
    chords: string[];
  };
}

export interface ControlEncoderParams {
  genreEmbedding: number[];
  tempoNormalized: number;
  moodEmbedding: number[];
  instrumentationEmbedding: number[];
  structureEmbedding: number[];
  energyLevel: number;
}

export interface AuroraSuggestion {
  id: string;
  sessionId: string;
  module: 'prompt_enhancer' | 'semantic_parser' | 'parameter_mapper' | 'structural_planner';
  suggestionType: string;
  originalValue: string | null;
  suggestedValue: string;
  confidence: number;
  reasoning: string;
  userAccepted: boolean | null;
  createdAt: Date;
}

export interface AuroraMetadata {
  module: string;
  implementation: 'codex' | 'local_llm' | 'rule_based';
  processingTimeMs: number;
  cacheHit?: boolean;
}
```

---

## 8. Real-Time Prompt Refinement Implementation

### 8.1 UI Component Architecture

The prompt refinement UI interacts with Aurora LLM's unified API:

```tsx
// components/AuroraPromptRefinement.tsx

import React, { useState, useCallback, useEffect } from 'react';
import { trpc } from '@/lib/trpc';
import { useDebounce } from '@/hooks/useDebounce';

export function AuroraPromptRefinement() {
  const [prompt, setPrompt] = useState('');
  const [enhancedPrompt, setEnhancedPrompt] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [parameters, setParameters] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  
  const debouncedPrompt = useDebounce(prompt, 300);

  // Real-time suggestions via Aurora LLM
  const suggestMutation = trpc.aurora.suggest.useMutation({
    onSuccess: (data) => {
      // Aurora handles all suggestion logic internally
      setSuggestions(data.suggestions);
    },
  });

  // Prompt enhancement via Aurora LLM
  const enhanceMutation = trpc.aurora.enhance.useMutation({
    onSuccess: (data) => {
      setEnhancedPrompt(data.enhanced_prompt);
      setSuggestions(data.additions);
      setIsProcessing(false);
    },
  });

  // Full parsing via Aurora LLM
  const parseMutation = trpc.aurora.parse.useMutation({
    onSuccess: (data) => {
      setParameters(data.parsed_parameters);
    },
  });

  useEffect(() => {
    if (debouncedPrompt.length > 10) {
      suggestMutation.mutate({ partial_prompt: debouncedPrompt });
    }
  }, [debouncedPrompt]);

  const handleEnhance = useCallback(() => {
    if (prompt.length < 10) return;
    setIsProcessing(true);
    enhanceMutation.mutate({ prompt, enhancement_level: 'moderate' });
  }, [prompt]);

  const handleParse = useCallback(() => {
    if (!enhancedPrompt) return;
    parseMutation.mutate({ prompt: enhancedPrompt });
  }, [enhancedPrompt]);

  // ... rest of component
}
```

### 8.2 Aurora LLM Client Service

```typescript
// services/auroraClient.ts

import { trpc } from '@/lib/trpc';

export class AuroraClient {
  /**
   * Enhance a user prompt with musical context.
   * Aurora LLM handles all implementation details internally.
   */
  async enhance(prompt: string, options?: EnhanceOptions): Promise<EnhanceResult> {
    return trpc.aurora.enhance.mutate({
      prompt,
      enhancement_level: options?.level || 'moderate',
      context: options?.context,
    });
  }

  /**
   * Get real-time suggestions as user types.
   */
  async suggest(partialPrompt: string): Promise<SuggestResult> {
    return trpc.aurora.suggest.mutate({
      partial_prompt: partialPrompt,
      suggestion_count: 3,
    });
  }

  /**
   * Parse a prompt into structured musical parameters.
   */
  async parse(prompt: string): Promise<ParseResult> {
    return trpc.aurora.parse.mutate({ prompt });
  }

  /**
   * Generate structural plan for music.
   */
  async plan(parameters: PlanParameters): Promise<PlanResult> {
    return trpc.aurora.plan.mutate({ parameters });
  }

  /**
   * Full generation pipeline.
   */
  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult> {
    return trpc.aurora.generate.mutate({
      prompt,
      options: {
        enhance_prompt: options?.enhance ?? true,
        duration_seconds: options?.duration ?? 30,
        output_format: options?.format ?? 'wav',
      },
    });
  }
}

export const auroraClient = new AuroraClient();
```

---

## 9. Error Handling and Fallback Strategies

### 9.1 Aurora LLM Fallback Hierarchy

Aurora LLM implements a multi-level fallback strategy when Codex is unavailable:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AURORA LLM FALLBACK HIERARCHY                             │
│                                                                              │
│  Level 1: GPT Codex (Primary Implementation)                                │
│     │  - Full NLP capabilities                                               │
│     │  - Highest quality suggestions                                         │
│     │                                                                        │
│     │ [Unavailable / Error / Timeout]                                        │
│     ▼                                                                        │
│  Level 2: Local LLM via LM Studio (Secondary Implementation)                │
│     │  - Runs locally, no external dependency                                │
│     │  - Good quality, may be slower                                         │
│     │  - Uses same Aurora prompt templates                                   │
│     │                                                                        │
│     │ [Unavailable / Error]                                                  │
│     ▼                                                                        │
│  Level 3: Rule-Based Parser (Tertiary Implementation)                       │
│     │  - Keyword extraction                                                  │
│     │  - Predefined mappings                                                 │
│     │  - No intelligent suggestions                                          │
│     │                                                                        │
│     │ [Error]                                                                │
│     ▼                                                                        │
│  Level 4: Default Parameters (Emergency Fallback)                           │
│       - Sensible defaults for all parameters                                 │
│       - User can manually adjust                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Implementation Selection Logic

```typescript
// services/auroraImplementationSelector.ts

export class AuroraImplementationSelector {
  private codexClient: CodexClient;
  private localLLMClient: LocalLLMClient;
  private ruleBasedParser: RuleBasedParser;

  async selectAndExecute<T>(
    task: AuroraTask,
    input: any
  ): Promise<{ result: T; implementation: string }> {
    
    // Level 1: Try Codex
    if (await this.isCodexAvailable()) {
      try {
        const result = await this.executeWithCodex<T>(task, input);
        return { result, implementation: 'codex' };
      } catch (error) {
        console.warn('Codex execution failed, trying Local LLM', error);
      }
    }

    // Level 2: Try Local LLM
    if (await this.isLocalLLMAvailable()) {
      try {
        const result = await this.executeWithLocalLLM<T>(task, input);
        return { result, implementation: 'local_llm' };
      } catch (error) {
        console.warn('Local LLM execution failed, trying rule-based', error);
      }
    }

    // Level 3: Try Rule-Based
    try {
      const result = await this.executeWithRuleBased<T>(task, input);
      return { result, implementation: 'rule_based' };
    } catch (error) {
      console.warn('Rule-based execution failed, using defaults', error);
    }

    // Level 4: Return Defaults
    return {
      result: this.getDefaultResult<T>(task),
      implementation: 'defaults',
    };
  }

  private async isCodexAvailable(): Promise<boolean> {
    try {
      await this.codexClient.healthCheck();
      return true;
    } catch {
      return false;
    }
  }

  private async isLocalLLMAvailable(): Promise<boolean> {
    try {
      await this.localLLMClient.healthCheck();
      return true;
    } catch {
      return false;
    }
  }
}
```

### 9.3 Error Codes

| Error Code | Description | Fallback Action |
|------------|-------------|-----------------|
| `AURORA_CODEX_TIMEOUT` | Codex API timed out | Try Local LLM |
| `AURORA_CODEX_RATE_LIMITED` | Codex rate limit exceeded | Try Local LLM |
| `AURORA_CODEX_UNAVAILABLE` | Codex service unavailable | Try Local LLM |
| `AURORA_LOCAL_LLM_UNAVAILABLE` | Local LLM not running | Try Rule-Based |
| `AURORA_PARSE_ERROR` | Failed to parse response | Retry with simpler prompt |
| `AURORA_VALIDATION_ERROR` | Output validation failed | Use partial results + defaults |

---

## 10. Performance Optimization Strategies

### 10.1 Caching Strategy

Aurora LLM implements multi-level caching:

| Cache Level | Storage | TTL | Contents |
|-------------|---------|-----|----------|
| **L1: Request** | In-memory | Request duration | Intermediate results |
| **L2: Session** | Redis | 1 hour | Enhancement results |
| **L3: Global** | Redis | 24 hours | Embeddings, common prompts |

### 10.2 Parallel Processing

Aurora LLM executes independent modules in parallel:

```typescript
// services/auroraParallelProcessor.ts

export async function processPromptParallel(prompt: string): Promise<ProcessingResult> {
  // Preprocessing is always first (sequential)
  const preprocessed = await preprocess(prompt);

  // Enhancement and initial parsing can run in parallel
  const [enhanceResult, initialParse] = await Promise.all([
    auroraEnhancer.enhance(preprocessed),
    auroraParser.quickParse(preprocessed),
  ]);

  // Use enhanced prompt for detailed parsing
  const detailedParse = await auroraParser.detailedParse(enhanceResult.enhanced_prompt);

  // Structural planning can start as soon as we have basic parameters
  const structuralPlan = await auroraPlanner.plan(detailedParse.parameters);

  return {
    enhancedPrompt: enhanceResult.enhanced_prompt,
    parameters: detailedParse.parameters,
    structuralPlan,
  };
}
```

### 10.3 Latency Budget

| Stage | Target | Max Acceptable |
|-------|--------|----------------|
| Preprocessing | 10ms | 50ms |
| Enhancement (Codex) | 400ms | 1500ms |
| Parsing (Codex) | 300ms | 1000ms |
| Parameter Mapping | 20ms | 100ms |
| Structural Planning (Codex) | 300ms | 1000ms |
| Embedding Generation | 50ms | 200ms |
| **Total Aurora Processing** | **1080ms** | **3850ms** |

---

## 11. Appendices

### Appendix A: Aurora LLM Configuration

```yaml
# aurora_config.yaml

aurora:
  version: "2.0"
  
  implementations:
    primary:
      type: "codex"
      api_url: "${CODEX_API_URL}"
      api_key: "${CODEX_API_KEY}"
      timeout_ms: 10000
      max_retries: 3
    
    secondary:
      type: "local_llm"
      endpoint: "http://localhost:1234/v1"
      model: "aurora-music-7b"
      timeout_ms: 15000
    
    fallback:
      type: "rule_based"
      config_path: "./config/rule_based_parser.json"
  
  modules:
    prompt_enhancer:
      enabled: true
      default_level: "moderate"
    
    semantic_parser:
      enabled: true
      confidence_threshold: 0.6
    
    parameter_mapper:
      enabled: true
      validation_strict: false
    
    structural_planner:
      enabled: true
      default_structure: "verse-chorus"
  
  caching:
    enabled: true
    redis_url: "${REDIS_URL}"
    ttl:
      enhancement: 3600
      embedding: 86400
      suggestion: 300
  
  monitoring:
    enabled: true
    metrics_endpoint: "/metrics"
    log_level: "info"
```

### Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Aurora LLM** | Soundstar's primary semantic intelligence layer that orchestrates all NLP operations |
| **GPT Codex** | OpenAI's model used as the implementation tool for Aurora LLM's NLP tasks |
| **Implementation Tool** | The underlying engine (Codex, Local LLM, or Rule-Based) that Aurora uses to perform tasks |
| **Prompt Enhancement** | The process of adding musical context to raw user prompts |
| **Semantic Parsing** | Extracting structured musical entities from natural language |
| **Parameter Mapping** | Converting extracted entities to Soundstar's internal schema |
| **Structural Planning** | Generating detailed section-by-section plans for music |
| **Unified Control Embedding** | The final numerical representation of all musical parameters |

### Appendix C: Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-07 | Manus AI | Initial release (Codex-centric) |
| 2.0 | 2026-01-07 | Manus AI | Revised to position Aurora LLM as primary architecture, Codex as implementation tool |

---

**End of Document**
