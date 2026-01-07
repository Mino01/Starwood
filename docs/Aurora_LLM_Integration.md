# Integrating Aurora LLM with Soundstar via LM Studio

The **Aurora LLM** is a powerful, context-aware language model that can significantly enhance the **Soundstar** framework's ability to interpret complex natural language prompts and generate sophisticated musical structures. This document outlines how to integrate Aurora, running locally via **LM Studio**, into the Soundstar pipeline.

## I. Architectural Enhancement Points

Aurora LLM is best utilized to augment the **Control Encoder** and the **Structural Transformer** components.

| Component | Current Role | Aurora LLM Enhancement |
| :--- | :--- | :--- |
| **Control Encoder** | Simple feature extraction. | **Semantic Deconstruction**: Deconstructs complex prompts into explicit musical parameters (e.g., "melancholic" $\rightarrow$ minor key, slow tempo, specific instrument timbre). |
| **Structural Transformer** | Generates RAVE latent vector. | **Structural Guidance**: Generates a high-level musical structure (e.g., AABA form, verse/chorus timing) as a conditioning vector for the Transformer. |

## II. Setup: Running Aurora via LM Studio

LM Studio provides a local, OpenAI-compatible API endpoint for running large language models like Aurora.

### A. LM Studio Configuration

1.  **Download and Run Aurora**: In LM Studio, download the Aurora model and load it onto your GPU.
2.  **Start Local Server**: Navigate to the **Local Server** tab in LM Studio.
3.  **Configure Endpoint**: Ensure the server is running on the default host and port (e.g., `http://localhost:1234`).
4.  **API Key**: While not strictly necessary for a local server, a dummy API key can be used if the client requires one.

### B. Soundstar `requirements.txt` Update

The `openai` Python library is the standard way to interact with the LM Studio API.

```bash
# Add to requirements.txt
openai
```

### C. Configuration File (`config.py`)

Create a configuration file to manage the API endpoint:

```python
# musicai/config.py
AURORA_API_BASE = "http://localhost:1234/v1"
AURORA_MODEL_NAME = "Aurora-LLM-GGUF" # Name used in LM Studio
```

## III. Integration into the Control Encoder

The `ControlEncoder` will be updated to use Aurora for **Semantic Deconstruction**.

### A. New `AuroraClient`

A dedicated client class handles communication with the LM Studio API:

```python
# musicai/tools/aurora_client.py
from openai import OpenAI
from musicai.config import AURORA_API_BASE, AURORA_MODEL_NAME

class AuroraClient:
    def __init__(self):
        # Client configured to point to LM Studio's local server
        self.client = OpenAI(base_url=AURORA_API_BASE, api_key="lm-studio") 
        self.model = AURORA_MODEL_NAME

    def deconstruct_prompt(self, prompt: str) -> dict:
        """Uses Aurora to extract explicit musical parameters from a text prompt."""
        system_prompt = (
            "You are an expert music theory and composition assistant. "
            "Analyze the user's music prompt and output a JSON object "
            "containing explicit musical parameters: 'genre', 'tempo', 'key', 'mood', 'instrumentation'."
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse and return the JSON object
        return json.loads(response.choices[0].message.content)
```

### B. Updated `ControlEncoder` Logic

The `ControlEncoder` will call `AuroraClient` to enrich the input features before generating the Unified Control Embedding.

```python
# musicai/core/control_encoder.py (Updated Snippet)
# ... imports ...
from musicai.tools.aurora_client import AuroraClient

class ControlEncoder(nn.Module):
    # ... __init__ ...
    
    def encode(self, prompt: str, **kwargs) -> torch.Tensor:
        # 1. Semantic Deconstruction using Aurora
        aurora_client = AuroraClient()
        musical_params = aurora_client.deconstruct_prompt(prompt)
        
        # 2. Use extracted parameters
        genre = musical_params.get('genre', kwargs.get('genre', 'Unknown'))
        tempo = musical_params.get('tempo', kwargs.get('tempo', 120))
        # ... process key, mood, instrumentation into features ...
        
        # 3. Generate Unified Control Embedding (as before)
        # ...
```

## IV. Integration into the Structural Transformer

Aurora can generate a **Structural Guidance Vector** that dictates the high-level form of the music (e.g., 8 bars of verse, 8 bars of chorus).

### A. Structural Guidance Generation

The `AuroraClient` can be extended to generate a sequence of structural tokens:

```python
# musicai/tools/aurora_client.py (Extended Snippet)
def generate_structure(self, prompt: str, duration: int) -> list:
    """Uses Aurora to generate a high-level musical structure."""
    # ... new system prompt to output a list of (section, duration_in_seconds) tuples ...
    # Example output: [{"section": "verse", "duration": 15}, {"section": "chorus", "duration": 15}]
    # ...
    return structure_list
```

### B. Updated `StructuralTransformerRAVE`

The Transformer will now receive this structural guidance as an additional conditioning input, ensuring the generated RAVE latent vector adheres to the desired form.

```python
# musicai/models/transformer.py (Conceptual Update)
class StructuralTransformerRAVE(nn.Module):
    # ... __init__ ...
    
    def generate_latent_vector(self, control_embeddings, duration):
        # 1. Get Structural Guidance
        # structure_guidance = AuroraClient().generate_structure(prompt, duration)
        
        # 2. Encode Guidance into a vector
        # structural_embedding = self.structure_encoder(structure_guidance)
        
        # 3. Condition Transformer on both embeddings
        # memory = torch.cat([control_embeddings, structural_embedding], dim=-1)
        # ... Transformer logic ...
        pass
```

By leveraging the Aurora LLM through LM Studio, Soundstar gains a powerful, locally-run semantic layer, moving the framework closer to the goal of the "ultimate sound" engine with unparalleled controllability.

***

## References

[1] LM Studio. *Run local LLMs on your desktop*. https://lmstudio.ai/
[2] OpenAI. *API Reference*. https://platform.openai.com/docs/api-reference
[3] Soundstar Project. *Soundstar Architecture: A Hybrid Synthesis Engine*. (See `docs/Soundstar_Architecture.md`)
