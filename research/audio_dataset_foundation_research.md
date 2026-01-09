# Starwood Foundation Research: Audio Datasets, GOAT Integration, and DDSP Baseline

## Research Notes - January 2026

---

## 1. Available Guitar and Music Audio Datasets

### 1.1 Guitar-Specific Datasets

| Dataset | Size | Content | License | Relevance |
|---------|------|---------|---------|-----------|
| **GOAT** | 29.5 hours | Paired guitar audio + tablature | Research | Primary training dataset |
| **GuitarSet** | ~3 hours | Hexaphonic guitar recordings with annotations | CC | Guitar transcription baseline |
| **DadaGP** | 26,181 songs | GuitarPro files + tokenized format | Research | Symbolic music generation |
| **ProgGP** | 173 songs | Progressive metal GuitarPro files | Research | Genre-specific training |
| **IDMT-SMT-Guitar** | Extensive | Guitar recordings with annotations | Research | Transcription research |
| **Acoustic Guitar Notes Dataset** | ~1,500 notes | Individual guitar notes | Kaggle | Note-level synthesis |

### 1.2 General Music Datasets

| Dataset | Size | Content | License | Relevance |
|---------|------|---------|---------|-----------|
| **NSynth** | 305,979 notes | One-shot instrumental notes (incl. guitar) | CC BY 4.0 | Timbre modeling |
| **MAESTRO** | 200+ hours | Piano audio + MIDI | CC BY-NC-SA 4.0 | Audio-MIDI alignment |
| **Slakh2100** | 145 hours | Multi-track synthesized audio | Research | Source separation |
| **MusicNet** | 330 recordings | Classical music with note annotations | CC BY 4.0 | Note-level labels |
| **FMA** | 343 days | Full-length tracks, 161 genres | CC | Genre diversity |
| **AudioSet** | 2M+ clips | 10-second clips, 632 event classes | CC BY 4.0 | Pre-training |
| **MusicCaps** | 5.5k pairs | Music-text descriptions | Research | Captioning |

### 1.3 Key Findings from AI Audio Datasets Repository

The Yuan-ManX/ai-audio-datasets repository (885 stars) provides comprehensive listing of:
- **Speech datasets**: 100+ datasets for TTS/ASR
- **Music datasets**: 80+ datasets for music generation/analysis
- **Sound effects**: 30+ datasets for audio event detection

Critical datasets for Starwood:
1. **GuitarSet** - Guitar transcription ground truth
2. **DadaGP** - Large-scale GuitarPro tokenization
3. **NSynth** - Guitar family timbre modeling
4. **MAESTRO** - Audio-MIDI alignment techniques
5. **Slakh2100** - Multi-instrument synthesis

---

## 2. GOAT Dataset Details (To be researched)

- 29.5 hours of paired guitar audio and tablature
- Multiple guitar timbres (DI, acoustic, amped)
- Synchronized audio and symbolic representations
- Suitable for training tab-to-audio and audio-to-tab models

---

## 3. DDSP Baseline Implementation (To be researched)

- Google Magenta DDSP framework
- Harmonic + noise synthesis
- Differentiable signal processing
- Real-time capable with optimization

---

## 4. Tonewood-Specific Data Collection Strategy (To be researched)

- Recording methodology
- Tonewood labeling
- Acoustic property measurement
- Dataset structure

---

*Research in progress...*


---

## 2. GOAT Dataset - Detailed Analysis

### 2.1 Dataset Overview

**GOAT (Guitar On Audio and Tablatures)** is a comprehensive dataset for guitar-focused MIR research.

| Metric | Value |
|--------|-------|
| **Unique DI Audio** | 5.9 hours |
| **Amplified Audio** | 29.5 hours (with augmentation) |
| **Standard Tuning** | 5.17 hours |
| **Non-Standard Tuning** | 43.85 minutes |
| **Total Notes** | 109,869 |
| **Total Chords** | 13,538 |
| **Individual Files** | 172 |
| **License** | CC BY 4.0 |
| **Distribution** | Zenodo (by request) |

### 2.2 Data Format and Structure

For each data point, GOAT provides:

1. **DI Audio** (.wav) - Raw direct input recordings at 44.1 kHz
2. **Amplified Audio** (.wav) - DI rendered through 5 amps × 5 cabinets = 25 variations
3. **Guitar Pro Tablature** (.gp, .gp5) - Full tablature with techniques
4. **DadaGP Tokens** (.txt) - Text-based encoding for ML models
5. **RSE Rendered Audio** (.wav) - Synthesized version from Guitar Pro
6. **MIDI** (.mid) - Both quantized and fine-aligned versions
7. **Metadata CSV** - File paths and song information

### 2.3 Guitar Types in Dataset

| Guitar | Standard Tuning | Alternative Tuning |
|--------|-----------------|-------------------|
| Stratocaster | 208.82 min | 26.15 min |
| Les Paul | 6.08 min | 7.36 min |
| Jazzmaster | 15.62 min | 10.34 min |
| Strandberg | 79.83 min | - |

### 2.4 Playing Techniques Covered

The dataset includes annotations for:
- **Bends** (most common)
- **Palm Mutes**
- **Legatos** (hammer-on/pull-off)
- **Slides**
- Tapping, vibrato (less common)

### 2.5 Chord Distribution

| Chord Type | Instances |
|------------|-----------|
| 3-note (triads, power chords) | 7,174 |
| 4-note | 4,260 |
| 5-note | 1,065 |
| 6-note | 1,039 |

### 2.6 Data Augmentation Strategy

GOAT uses **reamping** for data augmentation:
1. DI recordings processed through digital amplifier models
2. High-quality amp modeling software (Pedalboard Python package)
3. 5 amplifiers × 5 cabinet IRs = 25 tonal variations
4. Additional effects (25% chance): reverb, delay, chorus
5. Process repeated 5× = 29.5 hours total

**Key Finding**: Reamping improves model generalization:
- AMP-trained models outperform DI-only models
- Multiple reamping passes (AMP-XL) further improve results
- Zero-shot transfer to GuitarSet (acoustic) shows F1 > 0.84

### 2.7 Integration Strategy for Starwood

**Phase 1: Data Acquisition**
```python
# Request GOAT from Zenodo
# Dataset ID: [to be obtained from Zenodo]
# License: CC BY 4.0 (attribution required)
```

**Phase 2: Data Processing Pipeline**
```python
# 1. Load DI audio and corresponding tablature
# 2. Parse DadaGP tokens for technique labels
# 3. Extract tonewood-relevant features (if available)
# 4. Apply reamping augmentation with Starwood tonewoods
```

**Phase 3: Tonewood Augmentation**
- Replace generic amp models with Starwood tonewood models
- Apply tonewood character conditioning during reamping
- Create tonewood-labeled augmented dataset

---

## 3. DDSP Baseline Implementation (To be researched)

*Next section...*


---

## 3. DDSP Baseline Implementation

### 3.1 DDSP Library Overview

**DDSP (Differentiable Digital Signal Processing)** is Google Magenta's library for neural audio synthesis.

| Component | Description |
|-----------|-------------|
| **Core** | Differentiable DSP functions |
| **Processors** | Base classes for signal processing |
| **Synths** | Audio generators (Harmonic, FilteredNoise, Wavetable) |
| **Effects** | Audio transformers (Reverb, FIR Filter) |
| **Losses** | Spectral reconstruction losses |
| **Spectral Ops** | Fourier transforms and helpers |

### 3.2 Core DDSP Architecture

```
Neural Network → Controls → Processor → Audio Signal
                    ↓
              get_controls() → get_signal()
```

**Key Processors:**
1. **Harmonic Synthesizer**: Generates audio from amplitudes, harmonic distribution, and F0
2. **FilteredNoise**: Generates noise shaped by learned filter magnitudes
3. **TrainableReverb**: Learnable impulse response reverb
4. **Add**: Combines multiple audio signals

### 3.3 Training Pipeline

**Data Requirements:**
- 10-20 minutes of **monophonic** audio
- Single instrument recordings work best
- Training time: ~2-3 hours on GPU (Colab free tier)

**Training Process:**
1. Convert audio to dataset format
2. Extract F0 (pitch) and loudness features
3. Train autoencoder model
4. Export to `.tflite` for inference

### 3.4 DDSP-VST Plugin Architecture

**Two Plugin Types:**
1. **DDSP Effect**: Audio-to-audio transformation (timbre transfer)
2. **DDSP Synth**: MIDI-to-audio synthesis with ADSR envelope

**Built-in Models:** 11 pre-trained instrument models

### 3.5 DDSP for Polyphonic Guitar (Jonason et al. 2024)

**Key Innovation:** String-wise MIDI input for polyphonic synthesis

**Architecture:**
```
String-wise MIDI → Control Network → DDSP Synthesizer × 6 → Mix → Output
                         ↓
              F0, Loudness, Periodicity
```

**Control Features:**
- F0 (fundamental frequency) per string
- Loudness per string
- Periodicity (noise vs. harmonic ratio)

**Training Approach:**
1. Hexaphonic recordings (separate pickup per string)
2. String-wise MIDI alignment
3. Joint training of control prediction + synthesis

**Key Findings:**
- Classification task for control features outperforms regression
- Joint training improves results
- Direct MIDI-to-parameters prediction works best

### 3.6 Starwood DDSP Baseline Implementation Plan

**Phase 1: Basic DDSP Setup**
```python
import ddsp

# Initialize processors
harmonic = ddsp.synths.Harmonic()
noise = ddsp.synths.FilteredNoise()
reverb = ddsp.effects.TrainableReverb()

# ProcessorGroup DAG
dag = [
    (harmonic, ['amplitudes', 'harmonic_distribution', 'f0_hz']),
    (noise, ['magnitudes']),
    (ddsp.processors.Add(), ['harmonic/signal', 'filtered_noise/signal']),
    (reverb, ['add/signal'])
]

processor_group = ddsp.processors.ProcessorGroup(dag=dag)
```

**Phase 2: Tonewood Conditioning**
```python
# Add tonewood embedding to control network
class TonewoodConditionedEncoder(tf.keras.Model):
    def __init__(self, tonewood_dim=256):
        self.tonewood_embedding = tf.keras.layers.Embedding(
            num_tonewoods, tonewood_dim
        )
        self.encoder = ddsp.training.encoders.MfccTimeDistributedRnnEncoder()
        
    def call(self, audio, tonewood_id):
        # Get base encoding
        encoding = self.encoder(audio)
        # Add tonewood conditioning
        tonewood_emb = self.tonewood_embedding(tonewood_id)
        return encoding + tonewood_emb
```

**Phase 3: String-wise Polyphonic Extension**
```python
# 6 parallel DDSP synthesizers for guitar
class PolyphonicGuitarDDSP(tf.keras.Model):
    def __init__(self):
        self.string_synths = [
            ddsp.synths.Harmonic() for _ in range(6)
        ]
        self.mixer = ddsp.processors.Add()
        
    def call(self, string_controls):
        # Generate audio for each string
        string_audio = [
            synth(controls) 
            for synth, controls in zip(self.string_synths, string_controls)
        ]
        return self.mixer(string_audio)
```

### 3.7 Training Configuration

**Recommended Settings:**
```python
training_config = {
    'sample_rate': 16000,  # or 44100 for high quality
    'frame_rate': 250,     # control rate
    'n_samples': 64000,    # 4 seconds at 16kHz
    'batch_size': 16,
    'learning_rate': 1e-4,
    'epochs': 10000,
    'loss': 'multi_scale_spectral_loss'
}
```

**Loss Function:**
```python
# Multi-scale spectral loss
loss = ddsp.losses.SpectralLoss(
    loss_type='L1',
    mag_weight=1.0,
    logmag_weight=1.0,
    fft_sizes=[2048, 1024, 512, 256, 128, 64]
)
```

---

## 4. Tonewood-Specific Audio Collection (To be researched)

*Next section...*


---

## 4. Tonewood-Specific Audio Collection Strategies

### 4.1 Professional Tonewood Measurement Methodology

**Maderas Barber MB Sound System** (Giuliano Nicoletti):

| Parameter | Description | Measurement Method |
|-----------|-------------|-------------------|
| **Resonance Frequency** | First longitudinal bending mode (Hz) | Marble excitation + FFT analysis |
| **Density (ρ)** | Mass/Volume (kg/m³) | Physical measurement |
| **Stiffness (E)** | Young's modulus - resistance to deformation | Calculated from resonance |
| **Radiation Coefficient** | √(E/ρ³) - energy to sound conversion | Derived from E and ρ |
| **Q Factor** | Damping - how long wood rings | Peak shape analysis |

### 4.2 Ervin Somogyi Controlled Comparison Method

**Gold Standard for Tonewood Comparison:**
- Build **identical guitars** differing ONLY in soundboard wood
- Same: backs, sides, bracing, tuners, finish, voicing
- Professional player evaluation (Michael Chapdelaine)
- High-quality audio recording + video documentation

**Key Insight:** Factory dimensional consistency ≠ acoustic consistency. Hand-built approach required.

### 4.3 Recommended Audio Collection Protocol for Starwood

**Phase 1: Controlled Recording Setup**

```
Recording Environment:
├── Acoustically treated room (RT60 < 0.3s)
├── Temperature: 20-22°C (68-72°F)
├── Humidity: 45-55% RH
└── Background noise: < 30 dB SPL

Microphone Setup:
├── Primary: Large diaphragm condenser (12th fret, 12" distance)
├── Secondary: Small diaphragm condenser (bridge, 8" distance)
├── Room: Stereo pair (X/Y or ORTF, 3-4 feet)
└── DI: Piezo pickup direct (if available)

Recording Chain:
├── Preamp: High-quality, low-noise (< -128 dBu EIN)
├── Interface: 24-bit/96kHz minimum
├── DAW: Any professional DAW
└── Format: WAV, 24-bit, 96kHz
```

**Phase 2: Standardized Performance Protocol**

```python
recording_protocol = {
    'open_strings': {
        'description': 'Each string played open, let ring to silence',
        'repetitions': 3,
        'dynamics': ['pp', 'mf', 'ff']
    },
    'chromatic_scale': {
        'description': 'Full chromatic scale, each fret',
        'tempo': '60 BPM',
        'articulation': ['legato', 'staccato']
    },
    'chord_progressions': {
        'description': 'Standard chord shapes',
        'chords': ['C', 'G', 'Am', 'F', 'D', 'E', 'Bm'],
        'strumming': ['down', 'up', 'fingerpick']
    },
    'techniques': {
        'hammer_on': True,
        'pull_off': True,
        'slides': True,
        'bends': True,
        'harmonics': ['natural', 'artificial'],
        'palm_mute': True,
        'tremolo': True
    },
    'musical_excerpts': {
        'description': 'Short musical phrases',
        'styles': ['classical', 'fingerstyle', 'strumming', 'lead']
    }
}
```

**Phase 3: Tonewood Metadata Collection**

```python
tonewood_metadata = {
    'wood_species': str,           # e.g., 'Brazilian Rosewood'
    'scientific_name': str,        # e.g., 'Dalbergia nigra'
    'origin': str,                 # e.g., 'Bahia, Brazil'
    'age': int,                    # Years since harvest
    'density': float,              # kg/m³
    'youngs_modulus': float,       # GPa
    'radiation_coefficient': float,
    'q_factor': float,
    'moisture_content': float,     # Percentage
    'grain_orientation': str,      # 'quarter-sawn', 'flat-sawn'
    'guitar_position': str,        # 'top', 'back', 'sides'
    'guitar_model': str,
    'guitar_builder': str,
    'strings': str,                # Brand and gauge
    'recording_date': datetime,
    'recording_engineer': str,
    'room_acoustics': dict,
    'microphone_positions': dict
}
```

### 4.4 Dataset Structure for Starwood Training

```
starwood_tonewood_dataset/
├── metadata/
│   ├── guitars.json           # Guitar specifications
│   ├── tonewoods.json         # Tonewood properties
│   ├── recordings.json        # Recording session info
│   └── annotations.json       # Note-level annotations
├── audio/
│   ├── raw/                   # Original 96kHz recordings
│   │   ├── brazilian_rosewood/
│   │   ├── cocobolo/
│   │   ├── honduran_mahogany/
│   │   └── ...
│   ├── processed/             # Normalized, aligned
│   └── segments/              # Individual note/chord clips
├── features/
│   ├── mfcc/                  # Mel-frequency cepstral coefficients
│   ├── spectrograms/          # Mel spectrograms
│   ├── f0/                    # Pitch contours
│   └── loudness/              # Loudness envelopes
├── midi/
│   ├── aligned/               # MIDI aligned to audio
│   └── tablature/             # Guitar Pro format
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### 4.5 Minimum Viable Dataset Requirements

| Tonewood | Guitars | Hours | Notes |
|----------|---------|-------|-------|
| Brazilian Rosewood | 3-5 | 2-3 | Endangered, limited access |
| Cocobolo | 3-5 | 2-3 | CITES Appendix II |
| East Indian Rosewood | 5-10 | 5-10 | More accessible |
| Honduran Mahogany | 5-10 | 5-10 | Common, good baseline |
| Sitka Spruce (tops) | 5-10 | 5-10 | Most common top wood |
| Adirondack Spruce | 3-5 | 2-3 | Premium, limited |
| European Spruce | 3-5 | 2-3 | Classical standard |
| Cedar | 3-5 | 2-3 | Alternative top wood |
| Koa | 3-5 | 2-3 | Hawaiian specialty |
| Maple | 3-5 | 2-3 | Bright, articulate |

**Total Target: 50-100 hours of high-quality, annotated recordings**

### 4.6 Data Augmentation Strategies

```python
augmentation_pipeline = {
    'pitch_shift': {
        'range': [-2, +2],  # semitones
        'preserve_formants': True
    },
    'time_stretch': {
        'range': [0.9, 1.1],
        'preserve_pitch': True
    },
    'room_simulation': {
        'impulse_responses': ['studio', 'hall', 'room'],
        'wet_dry_mix': [0.1, 0.5]
    },
    'noise_injection': {
        'types': ['white', 'pink', 'room_tone'],
        'snr_range': [20, 40]  # dB
    },
    'eq_variation': {
        'bands': 10,
        'range': [-3, +3]  # dB
    },
    'dynamic_range': {
        'compression': [1.5, 3.0],  # ratio
        'expansion': [0.5, 0.8]
    }
}
```

### 4.7 Partnership Opportunities for Data Collection

| Partner Type | Examples | Value |
|--------------|----------|-------|
| **Guitar Manufacturers** | Taylor, Martin, Gibson, Collings | Access to controlled builds |
| **Luthier Schools** | Roberto-Venn, Galloup, CIPI | Student projects |
| **Tonewood Suppliers** | Maderas Barber, LMI, Allied Lutherie | Wood samples + metadata |
| **Recording Studios** | Sweetwater, Guitar Center | Professional recordings |
| **Universities** | Stanford CCRMA, McGill IDMIL | Research collaboration |
| **YouTube Creators** | Acoustic Letter, Darrell Braun | Existing comparison content |

---

## 5. Integration Summary

### 5.1 Complete Foundation Phase Deliverables

| Component | Source | Integration |
|-----------|--------|-------------|
| **GOAT Dataset** | GitHub/Zenodo | Pre-training for guitar synthesis |
| **GuitarSet** | Zenodo | Polyphonic transcription training |
| **NSynth** | Magenta | Instrument timbre understanding |
| **DDSP Library** | Google Magenta | Baseline synthesis engine |
| **Custom Tonewood Data** | Partnerships + Recording | Tonewood conditioning |

### 5.2 Training Pipeline

```
Phase 1: Pre-training (Months 1-2)
├── Train DDSP baseline on GOAT dataset
├── Validate on GuitarSet
└── Establish baseline metrics

Phase 2: Tonewood Data Collection (Months 2-4)
├── Partner with luthiers/manufacturers
├── Record controlled tonewood comparisons
└── Build annotated dataset

Phase 3: Tonewood Conditioning (Months 4-6)
├── Add tonewood embeddings to DDSP
├── Fine-tune on tonewood dataset
└── Validate tonewood discrimination

Phase 4: Guitar Pro Integration (Months 6-8)
├── Integrate PyGuitarPro parser
├── Map tablature to DDSP controls
└── End-to-end tab-to-audio pipeline
```

---

*Research compiled: January 2026*
*Repository: https://github.com/Mino01/Starwood*
