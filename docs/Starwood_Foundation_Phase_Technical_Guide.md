# Starwood Foundation Phase Technical Guide

## Audio Dataset Collection, GOAT Integration, and DDSP Baseline Implementation

**Version:** 1.0  
**Date:** January 2026  
**Author:** Manus AI  
**Repository:** https://github.com/Mino01/Starwood

---

## Executive Summary

This technical guide provides a comprehensive blueprint for implementing the **Foundation Phase** of the Starwood project. The Foundation Phase establishes the core infrastructure for neural tonewood synthesis by addressing three critical components: audio dataset collection, integration with the GOAT (Guitar-Oriented Audio Tablature) dataset, and implementation of a DDSP (Differentiable Digital Signal Processing) baseline synthesis engine.

The Starwood project aims to revolutionize guitar sound synthesis by enabling any guitar to produce the tonal characteristics of premium tonewoods such as Brazilian rosewood, Cocobolo, and Honduran mahogany. This guide outlines the technical requirements, data pipelines, and implementation strategies necessary to achieve this goal.

---

## Table of Contents

1. [Audio Dataset Landscape](#1-audio-dataset-landscape)
2. [GOAT Dataset Integration](#2-goat-dataset-integration)
3. [DDSP Baseline Implementation](#3-ddsp-baseline-implementation)
4. [Tonewood-Specific Data Collection](#4-tonewood-specific-data-collection)
5. [Training Pipeline Architecture](#5-training-pipeline-architecture)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [References](#references)

---

## 1. Audio Dataset Landscape

### 1.1 Overview of Available Datasets

The development of a neural tonewood synthesis engine requires access to high-quality, annotated audio datasets. The following table summarizes the most relevant datasets for the Starwood project, organized by their primary application.

| Dataset | Size | Content | License | Starwood Application |
|---------|------|---------|---------|---------------------|
| **GOAT** [1] | 29.5 hours | Paired guitar audio + tablature | CC BY-NC-SA 4.0 | Primary pre-training |
| **GuitarSet** [2] | 3 hours | Hexaphonic guitar recordings | CC BY 4.0 | String-wise synthesis |
| **NSynth** [3] | 305,979 samples | Musical instrument notes | CC BY 4.0 | Timbre understanding |
| **MAESTRO** [4] | 200 hours | Piano MIDI + audio | CC BY-NC-SA 4.0 | Synthesis techniques |
| **MusicNet** [5] | 34 hours | Classical music + annotations | CC BY-SA 4.0 | General music understanding |
| **MUSDB18** [6] | 10 hours | Multi-track music | CC BY-NC-SA 4.0 | Source separation |
| **FMA** [7] | 106,574 tracks | Full-length music | Various | General audio features |

### 1.2 Guitar-Specific Datasets

For the Starwood project, guitar-specific datasets are of paramount importance. The **GOAT dataset** stands out as the most comprehensive resource, providing paired audio and tablature data that enables supervised training of tab-to-audio synthesis models.

The **GuitarSet** dataset offers a unique advantage through its hexaphonic recordings, which capture each string independently. This enables string-wise synthesis approaches that are essential for accurate polyphonic guitar modeling, as demonstrated by Jonason et al. in their DDSP-based polyphonic guitar synthesis work [8].

### 1.3 Dataset Selection Criteria

When selecting datasets for Starwood training, the following criteria should be prioritized:

1. **Audio Quality**: Minimum 44.1 kHz sample rate, 16-bit depth, professional recording conditions
2. **Annotation Density**: Note-level or frame-level annotations preferred over song-level labels
3. **Instrument Coverage**: Acoustic guitar recordings with documented instrument specifications
4. **License Compatibility**: Permissive licenses that allow derivative works and commercial use
5. **Metadata Richness**: Information about recording equipment, room acoustics, and instrument characteristics

---

## 2. GOAT Dataset Integration

### 2.1 Dataset Overview

The **GOAT (Guitar-Oriented Audio Tablature)** dataset represents a significant advancement in guitar audio research. Created by researchers at Queen Mary University of London and published in 2024, GOAT provides 29.5 hours of paired audio and tablature data specifically designed for guitar transcription and synthesis tasks [1].

> "GOAT is a large-scale dataset of paired audio and tablature for guitar transcription research. It contains 29.5 hours of audio from 1,337 unique songs, with corresponding Guitar Pro tablature files." [1]

### 2.2 Dataset Structure

The GOAT dataset is organized hierarchically with the following structure:

```
goat_dataset/
├── audio/
│   ├── train/
│   │   ├── song_001.wav
│   │   ├── song_002.wav
│   │   └── ...
│   ├── val/
│   └── test/
├── tablature/
│   ├── train/
│   │   ├── song_001.gp5
│   │   ├── song_002.gp5
│   │   └── ...
│   ├── val/
│   └── test/
├── midi/
│   ├── train/
│   ├── val/
│   └── test/
├── metadata/
│   ├── song_info.json
│   ├── splits.json
│   └── statistics.json
└── README.md
```

### 2.3 Key Statistics

| Metric | Value |
|--------|-------|
| Total Duration | 29.5 hours |
| Number of Songs | 1,337 |
| Audio Format | WAV, 44.1 kHz, 16-bit |
| Tablature Format | Guitar Pro 5 (.gp5) |
| Average Song Length | 79.4 seconds |
| Genres | Rock, Metal, Pop, Blues, Jazz |

### 2.4 Integration Strategy

The integration of GOAT into the Starwood training pipeline involves three stages:

**Stage 1: Data Preprocessing**

```python
import guitarpro
import librosa
import numpy as np

def preprocess_goat_sample(audio_path, tab_path, sample_rate=16000):
    """
    Preprocess a GOAT dataset sample for DDSP training.
    
    Args:
        audio_path: Path to WAV file
        tab_path: Path to Guitar Pro file
        sample_rate: Target sample rate for DDSP
    
    Returns:
        dict: Preprocessed features and controls
    """
    # Load and resample audio
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    
    # Parse Guitar Pro tablature
    song = guitarpro.parse(tab_path)
    
    # Extract note events with timing
    note_events = extract_note_events(song)
    
    # Align tablature to audio
    aligned_events = align_tab_to_audio(note_events, audio, sr)
    
    # Extract DDSP control features
    f0 = extract_f0(audio, sr)
    loudness = extract_loudness(audio, sr)
    
    return {
        'audio': audio,
        'f0': f0,
        'loudness': loudness,
        'note_events': aligned_events,
        'sample_rate': sample_rate
    }
```

**Stage 2: Feature Extraction**

The DDSP framework requires specific control features extracted from the audio:

| Feature | Description | Extraction Method |
|---------|-------------|-------------------|
| **F0 (Pitch)** | Fundamental frequency contour | CREPE or PYIN algorithm |
| **Loudness** | Perceptual loudness envelope | A-weighted power spectrum |
| **Periodicity** | Harmonic vs. noise ratio | Autocorrelation analysis |

**Stage 3: Training Data Generation**

```python
def create_training_batch(goat_samples, batch_size=16, segment_length=64000):
    """
    Create training batches from preprocessed GOAT samples.
    """
    batch = {
        'audio': np.zeros((batch_size, segment_length)),
        'f0_hz': np.zeros((batch_size, segment_length // 256)),
        'loudness_db': np.zeros((batch_size, segment_length // 256)),
        'note_events': []
    }
    
    for i, sample in enumerate(goat_samples[:batch_size]):
        # Random segment extraction
        start = np.random.randint(0, len(sample['audio']) - segment_length)
        batch['audio'][i] = sample['audio'][start:start + segment_length]
        
        # Corresponding control features
        ctrl_start = start // 256
        ctrl_length = segment_length // 256
        batch['f0_hz'][i] = sample['f0'][ctrl_start:ctrl_start + ctrl_length]
        batch['loudness_db'][i] = sample['loudness'][ctrl_start:ctrl_start + ctrl_length]
    
    return batch
```

---

## 3. DDSP Baseline Implementation

### 3.1 DDSP Architecture Overview

**Differentiable Digital Signal Processing (DDSP)** is a library developed by Google Magenta that enables the integration of classical signal processing components into deep learning models [9]. Unlike purely neural approaches, DDSP provides interpretable, controllable synthesis through differentiable implementations of oscillators, filters, and effects.

The core DDSP architecture consists of the following components:

```
Neural Network → Controls → Processor → Audio Signal
                    ↓
              get_controls() → get_signal()
```

### 3.2 Core DDSP Components

| Component | Class | Function |
|-----------|-------|----------|
| **Harmonic Synthesizer** | `ddsp.synths.Harmonic` | Generates audio from amplitudes, harmonic distribution, and F0 |
| **Filtered Noise** | `ddsp.synths.FilteredNoise` | Generates noise shaped by learned filter magnitudes |
| **Trainable Reverb** | `ddsp.effects.TrainableReverb` | Learnable impulse response reverb |
| **Processor Group** | `ddsp.processors.ProcessorGroup` | Chains multiple processors into a DAG |

### 3.3 Baseline Implementation

The Starwood DDSP baseline implementation follows a three-stage architecture:

**Stage 1: Encoder Network**

```python
import ddsp
import tensorflow as tf

class StarwoodEncoder(tf.keras.Model):
    """
    Encoder network that extracts control features from audio.
    """
    def __init__(self, output_splits=(('f0_hz', 1),
                                       ('loudness_db', 1),
                                       ('harmonic_distribution', 60),
                                       ('noise_magnitudes', 65))):
        super().__init__()
        self.output_splits = output_splits
        
        # MFCC-based encoder
        self.encoder = ddsp.training.encoders.MfccTimeDistributedRnnEncoder(
            rnn_channels=512,
            rnn_type='gru',
            z_dims=16,
            z_time_steps=250
        )
        
        # Output dense layers
        self.dense_out = ddsp.training.nn.OutputSplitsController(
            output_splits=output_splits
        )
    
    def call(self, features):
        z = self.encoder(features)
        outputs = self.dense_out(z)
        return outputs
```

**Stage 2: Synthesis Network**

```python
class StarwoodSynthesizer(tf.keras.Model):
    """
    DDSP-based synthesizer for guitar audio generation.
    """
    def __init__(self, sample_rate=16000, n_samples=64000):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        
        # Harmonic synthesizer
        self.harmonic = ddsp.synths.Harmonic(
            n_samples=n_samples,
            sample_rate=sample_rate,
            name='harmonic'
        )
        
        # Filtered noise
        self.noise = ddsp.synths.FilteredNoise(
            n_samples=n_samples,
            window_size=0,
            name='filtered_noise'
        )
        
        # Reverb
        self.reverb = ddsp.effects.Reverb(
            trainable=True,
            reverb_length=48000,
            name='reverb'
        )
    
    def call(self, controls):
        # Generate harmonic component
        harmonic_audio = self.harmonic(
            controls['amplitudes'],
            controls['harmonic_distribution'],
            controls['f0_hz']
        )
        
        # Generate noise component
        noise_audio = self.noise(controls['noise_magnitudes'])
        
        # Combine and apply reverb
        audio = harmonic_audio + noise_audio
        audio = self.reverb(audio)
        
        return audio
```

**Stage 3: Complete Model**

```python
class StarwoodDDSPBaseline(tf.keras.Model):
    """
    Complete DDSP baseline model for Starwood.
    """
    def __init__(self, sample_rate=16000, n_samples=64000):
        super().__init__()
        self.encoder = StarwoodEncoder()
        self.synthesizer = StarwoodSynthesizer(sample_rate, n_samples)
        
        # Multi-scale spectral loss
        self.loss_fn = ddsp.losses.SpectralLoss(
            loss_type='L1',
            mag_weight=1.0,
            logmag_weight=1.0,
            fft_sizes=[2048, 1024, 512, 256, 128, 64]
        )
    
    def call(self, features, training=False):
        controls = self.encoder(features)
        audio = self.synthesizer(controls)
        return audio, controls
    
    def compute_loss(self, audio_target, audio_pred):
        return self.loss_fn(audio_target, audio_pred)
```

### 3.4 Training Configuration

The following configuration is recommended for training the DDSP baseline:

```python
training_config = {
    'sample_rate': 16000,
    'frame_rate': 250,
    'n_samples': 64000,  # 4 seconds at 16kHz
    'batch_size': 16,
    'learning_rate': 1e-4,
    'epochs': 10000,
    'optimizer': 'adam',
    'loss': 'multi_scale_spectral_loss',
    'checkpoint_interval': 1000,
    'validation_interval': 500
}
```

### 3.5 Polyphonic Extension for Guitar

For polyphonic guitar synthesis, the baseline must be extended to handle multiple simultaneous notes. Following the approach of Jonason et al. [8], we implement a string-wise architecture:

```python
class PolyphonicGuitarDDSP(tf.keras.Model):
    """
    String-wise DDSP synthesizer for polyphonic guitar.
    """
    def __init__(self, n_strings=6, sample_rate=16000, n_samples=64000):
        super().__init__()
        self.n_strings = n_strings
        
        # One synthesizer per string
        self.string_synths = [
            StarwoodSynthesizer(sample_rate, n_samples)
            for _ in range(n_strings)
        ]
        
        # Mixing network
        self.mixer = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, string_controls):
        """
        Args:
            string_controls: List of 6 control dictionaries, one per string
        """
        string_audio = []
        for i, controls in enumerate(string_controls):
            audio = self.string_synths[i](controls)
            string_audio.append(audio)
        
        # Stack and mix
        stacked = tf.stack(string_audio, axis=-1)
        mixed = tf.reduce_sum(stacked, axis=-1)
        
        return mixed
```

---

## 4. Tonewood-Specific Data Collection

### 4.1 The Challenge of Tonewood Data

Unlike general guitar audio datasets, tonewood-specific data requires controlled recording conditions where the **only variable** is the tonewood itself. This presents significant challenges:

1. **Instrument Availability**: Premium tonewoods like Brazilian rosewood are rare and expensive
2. **Controlled Builds**: Identical guitars with different tonewoods are uncommon
3. **Recording Consistency**: Same player, same room, same equipment required
4. **Metadata Requirements**: Detailed wood properties must be documented

### 4.2 Professional Measurement Standards

Two industry-standard systems provide the foundation for tonewood characterization:

**TPC (Tonewood Parameters Characterization) System** [10]:

| Parameter | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| Density | ρ | kg/m³ | Mass per unit volume |
| Young's Modulus | E | GPa | Longitudinal stiffness |
| Shear Modulus | G | GPa | Transverse stiffness |
| Q Factor | Q | dimensionless | Damping (higher = longer sustain) |
| Radiation Coefficient | R | m⁴/kg·s | Sound radiation efficiency |

**Pacific Rim Tonewoods Sonic Grading** [11]:

| Grade | Q Factor Range | Characteristics |
|-------|----------------|-----------------|
| Low Q | < 80 | Warm, mellow, quick decay |
| Mid Q | 80-120 | Balanced, versatile |
| High Q | > 120 | Bright, sustaining, projecting |

### 4.3 Recommended Recording Protocol

The following protocol ensures consistent, high-quality tonewood recordings:

**Environment Requirements:**
- Acoustically treated room with RT60 < 0.3 seconds
- Temperature: 20-22°C (68-72°F)
- Humidity: 45-55% relative humidity
- Background noise: < 30 dB SPL

**Microphone Configuration:**
- Primary: Large diaphragm condenser at 12th fret, 12" distance
- Secondary: Small diaphragm condenser at bridge, 8" distance
- Room: Stereo pair (X/Y or ORTF) at 3-4 feet
- DI: Piezo pickup direct (if available)

**Performance Protocol:**

| Segment | Description | Duration |
|---------|-------------|----------|
| Open Strings | Each string played open, pp/mf/ff | 2 min |
| Chromatic Scale | Full fretboard, legato and staccato | 5 min |
| Chord Progressions | Standard shapes, strummed and fingerpicked | 5 min |
| Techniques | Hammer-ons, pull-offs, slides, bends, harmonics | 5 min |
| Musical Excerpts | Short phrases in various styles | 5 min |

### 4.4 Minimum Viable Dataset

For initial Starwood training, the following minimum dataset is recommended:

| Tonewood | Guitars | Hours | Priority |
|----------|---------|-------|----------|
| Brazilian Rosewood | 3-5 | 2-3 | High (reference) |
| Cocobolo | 3-5 | 2-3 | High |
| East Indian Rosewood | 5-10 | 5-10 | Medium |
| Honduran Mahogany | 5-10 | 5-10 | Medium (baseline) |
| Sitka Spruce (tops) | 5-10 | 5-10 | High |
| Koa | 3-5 | 2-3 | Medium |

**Total Target: 50-100 hours of annotated recordings**

### 4.5 Data Augmentation

To expand the effective dataset size, the following augmentation strategies are recommended:

```python
augmentation_config = {
    'pitch_shift': {'range': [-2, +2], 'preserve_formants': True},
    'time_stretch': {'range': [0.9, 1.1], 'preserve_pitch': True},
    'room_simulation': {'impulse_responses': ['studio', 'hall', 'room']},
    'noise_injection': {'snr_range': [20, 40]},
    'eq_variation': {'bands': 10, 'range': [-3, +3]}
}
```

---

## 5. Training Pipeline Architecture

### 5.1 End-to-End Pipeline

The complete Starwood training pipeline integrates all components:

```
┌─────────────────────────────────────────────────────────────────┐
│                    STARWOOD TRAINING PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  GOAT Dataset │    │  GuitarSet   │    │  Tonewood    │       │
│  │  (29.5 hrs)   │    │  (3 hrs)     │    │  Dataset     │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └─────────┬─────────┴─────────┬─────────┘                │
│                   ▼                   ▼                          │
│         ┌─────────────────────────────────────┐                  │
│         │        DATA PREPROCESSING           │                  │
│         │  • Audio resampling (16kHz)         │                  │
│         │  • Feature extraction (F0, loudness)│                  │
│         │  • Tablature alignment              │                  │
│         │  • Tonewood metadata encoding       │                  │
│         └─────────────────┬───────────────────┘                  │
│                           ▼                                      │
│         ┌─────────────────────────────────────┐                  │
│         │        DDSP BASELINE TRAINING       │                  │
│         │  • Encoder: MFCC + GRU              │                  │
│         │  • Synthesizer: Harmonic + Noise    │                  │
│         │  • Loss: Multi-scale spectral       │                  │
│         └─────────────────┬───────────────────┘                  │
│                           ▼                                      │
│         ┌─────────────────────────────────────┐                  │
│         │      TONEWOOD CONDITIONING          │                  │
│         │  • Tonewood embedding (256-dim)     │                  │
│         │  • FiLM conditioning layers         │                  │
│         │  • Fine-tuning on tonewood data     │                  │
│         └─────────────────┬───────────────────┘                  │
│                           ▼                                      │
│         ┌─────────────────────────────────────┐                  │
│         │        EVALUATION & EXPORT          │                  │
│         │  • Objective metrics (FAD, FD)      │                  │
│         │  • Subjective listening tests       │                  │
│         │  • TFLite export for deployment     │                  │
│         └─────────────────────────────────────┘                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Training Stages

| Stage | Duration | Objective | Data |
|-------|----------|-----------|------|
| **Pre-training** | Months 1-2 | Learn general guitar synthesis | GOAT + GuitarSet |
| **Fine-tuning** | Months 3-4 | Adapt to tonewood variations | Custom tonewood data |
| **Conditioning** | Months 5-6 | Enable tonewood selection | All data + embeddings |

### 5.3 Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **FAD** | Fréchet Audio Distance | < 5.0 |
| **FD** | Fréchet Distance (embeddings) | < 10.0 |
| **PESQ** | Perceptual speech quality | > 3.5 |
| **Tonewood Accuracy** | Classification accuracy | > 90% |
| **MOS** | Mean Opinion Score (human) | > 4.0 |

---

## 6. Implementation Roadmap

### 6.1 Phase Timeline

| Phase | Months | Milestones |
|-------|--------|------------|
| **Foundation** | 1-3 | GOAT integration, DDSP baseline, initial training |
| **Data Collection** | 2-4 | Partnership establishment, tonewood recordings |
| **Conditioning** | 4-6 | Tonewood embeddings, FiLM layers, fine-tuning |
| **Integration** | 6-8 | Guitar Pro parser, tab-to-audio pipeline |
| **Optimization** | 8-10 | Real-time inference, TFLite export |
| **Deployment** | 10-12 | VST plugin, hardware prototype |

### 6.2 Resource Requirements

| Resource | Specification | Purpose |
|----------|---------------|---------|
| **GPU** | NVIDIA A100 or equivalent | Training |
| **Storage** | 2 TB SSD | Dataset storage |
| **RAM** | 64 GB | Data preprocessing |
| **Recording Equipment** | Professional studio setup | Data collection |
| **Guitars** | 20-50 instruments | Tonewood coverage |

### 6.3 Risk Mitigation

| Risk | Mitigation Strategy |
|------|---------------------|
| Limited tonewood data | Partner with luthiers, use data augmentation |
| Training instability | Curriculum learning, gradient clipping |
| Real-time performance | Model pruning, quantization, TFLite |
| Subjective quality | Extensive listening tests, iterative refinement |

---

## References

[1] GOAT Dataset. "Guitar-Oriented Audio Tablature Dataset." Queen Mary University of London, 2024. https://github.com/qmul/goat

[2] Xi, Q., et al. "GuitarSet: A Dataset for Guitar Transcription." ISMIR, 2018. https://guitarset.weebly.com/

[3] Engel, J., et al. "NSynth: Neural Audio Synthesis." Magenta, 2017. https://magenta.tensorflow.org/nsynth

[4] Hawthorne, C., et al. "MAESTRO: A Dataset for Music Transcription." ICLR, 2019. https://magenta.tensorflow.org/datasets/maestro

[5] Thickstun, J., et al. "MusicNet: A Dataset for Music Research." NIPS, 2016. https://homes.cs.washington.edu/~thicMDstun/musicnet.html

[6] Rafii, Z., et al. "MUSDB18: A Corpus for Music Separation." ISMIR, 2017. https://sigsep.github.io/datasets/musdb.html

[7] Defferrard, M., et al. "FMA: A Dataset for Music Analysis." ISMIR, 2017. https://github.com/mdeff/fma

[8] Jonason, N., et al. "DDSP-Based Neural Waveform Synthesis of Polyphonic Guitar Performance from String-Wise MIDI Input." DAFx, 2024. https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_49.pdf

[9] Engel, J., et al. "DDSP: Differentiable Digital Signal Processing." ICLR, 2020. https://github.com/magenta/ddsp

[10] Iulius Guitars. "TPC System: Tonewood Parameters Characterization." https://www.iuliusguitars.com/tpc/

[11] Pacific Rim Tonewoods. "Sonic Grading System." https://pacificrimtonewoods.com/pages/sonic-grading

[12] Maderas Barber. "MB Sound: Tonewood Audio Analysis." https://maderasbarber.com/tonewood/

[13] Somogyi, E. "A Systematic Comparison of Tonewoods." https://esomogyi.com/a-systematic-comparison-of-tonewoods/

---

*Document generated by Manus AI for the Starwood project.*
*Repository: https://github.com/Mino01/Starwood*
