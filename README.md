# Awesome fMRI Decoding (Categorized)

A curated list of **fMRI-based brain decoding** resources, focusing on **reconstructing images, videos, language, and audio from brain activity**.

If you find this repo helpful, please give it a ⭐ and consider contributing new papers via pull requests.

---

## Contents

- [0. Survey & Background](#0-survey--background)
  - [0.1 Survey & Overview](#01-survey--overview)
  - [0.2 Tutorials & Intro Reading](#02-tutorials--intro-reading)
- [1. Resources](#1-resources)
  - [1.1 Datasets & Benchmarks](#11-datasets--benchmarks)
  - [1.2 Toolboxes & Codebases](#12-toolboxes--codebases)
  - [1.3 Other Awesome Lists](#13-other-awesome-lists)
- [2. fMRI Decoding Methods (by Task)](#2-fmri-decoding-methods-by-task)
  - [2.1 Brain → Image (Static Vision)](#21-brain--image-static-vision)
    - [2.1.1 Early / Pre-deep](#211-early--pre-deep)
    - [2.1.2 GAN / VAE–based](#212-gan--vae-based)
    - [2.1.3 Diffusion-based](#213-diffusion-based)
    - [2.1.4 Cross-subject / Few-shot / MoE](#214-cross-subject--few-shot--moe)
  - [2.2 Brain → Video / Dynamic Scene](#22-brain--video--dynamic-scene)
  - [2.3 Brain → Text / Narrative](#23-brain--text--narrative)
  - [2.4 Brain → Audio / Music](#24-brain--audio--music)
  - [2.5 Multimodal & Foundation-Model-based Decoding](#25-multimodal--foundation-model-based-decoding)
  - [2.6 Clinical / Cognitive / Mental-State Decoding](#26-clinical--cognitive--mental-state-decoding)
- [3. Related fMRI Modeling](#3-related-fmri-modeling)
  - [3.1 Visual → fMRI Encoding & Data Augmentation](#31-visual--fmri-encoding--data-augmentation)
  - [3.2 Multimodal fMRI + EEG / MEG](#32-multimodal-fmri--eeg--meg)
  - [3.3 Representation Alignment & Analysis](#33-representation-alignment--analysis)
- [Contributing](#contributing)
- [License](#license)

---

## 0. Survey & Background

### 0.1 Survey & Overview

A Survey on fMRI-based Brain Decoding for Reconstructing Multimodal Stimuli  
[Paper](https://arxiv.org/abs/2503.15978) [Project](https://github.com/LpyNow/BrainDecodingImage)

Brain-Conditional Multimodal Synthesis: A Survey and Taxonomy  
[Paper](https://www.computer.org/csdl/journal/ai/2025/05/10798967/22EatqRGQxO)

Deep Generative Models in Brain Encoding and Decoding  
[Paper](https://doi.org/10.1016/j.eng.2019.03.011)

### 0.2 Tutorials & Intro Reading

*(Add tutorials / blog posts / lecture notes here.)*

---

## 1. Resources

### 1.1 Datasets & Benchmarks

Natural Scenes Dataset (NSD)  
[Website](https://naturalscenesdataset.org/) [Data](https://osf.io/9pjky/)

Deep Image Reconstruction (DIR) dataset  
[Data](https://openneuro.org/datasets/ds001506)

Narratives / Story listening datasets  
[Website](https://www.narrativeslab.org/) [Data](https://openneuro.org/datasets/ds002345)

Semantic reconstruction of continuous language – dataset  
[Data](https://openneuro.org/datasets/ds003020)

*(Feel free to add Vim-1, BOLD5000, GOD, movie datasets, etc.)*

### 1.2 Toolboxes & Codebases

DeepImageReconstruction  
[Code](https://github.com/KamitaniLab/DeepImageReconstruction)

semantic-decoding  
[Code](https://github.com/HuthLab/semantic-decoding)

MindReader  
[Code](https://github.com/yuvalsim/MindReader)

MindEye2  
[Code](https://github.com/MedARC-AI/MindEyeV2)

### 1.3 Other Awesome Lists

awesome-brain-decoding  
[GitHub](https://github.com/NeuSpeech/awesome-brain-decoding)

Awesome Brain Encoding & Decoding  
[GitHub](https://github.com/subbareddy248/Awesome-Brain-Encoding--Decoding)

Awesome Brain Graph Learning with GNNs  
[GitHub](https://github.com/XuexiongLuoMQ/Awesome-Brain-Graph-Learning-with-GNNs)

---

## 2. fMRI Decoding Methods (by Task)

> Tags like Brain→Image / Brain→Text / Diffusion / GAN 等可以写在标题行后面的小括号里。

---

### 2.1 Brain → Image (Static Vision)

#### 2.1.1 Early / Pre-deep

Reconstructing Natural Scenes from fMRI Patterns Using Hierarchical Visual Features (Brain→Image)  
[Paper](https://doi.org/10.1016/j.neuroimage.2010.07.063)

Visual Experience Reconstruction from Movie fMRI (Brain→Video)  
[Paper](https://doi.org/10.1016/j.cub.2011.01.031)

#### 2.1.2 GAN / VAE–based

Deep Image Reconstruction from Human Brain Activity (Brain→Image, GAN/VAE)  
[Paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633) [Code](https://github.com/KamitaniLab/DeepImageReconstruction) [Dataset](https://openneuro.org/datasets/ds001506)

From Voxels to Pixels and Back: Self-supervision in Natural-Image Reconstruction from fMRI (Brain→Image, Self-supervised)  
[Paper](https://arxiv.org/abs/1907.02431) [Code](https://github.com/WeizmannVision/ssfmri2im)

Reconstructing Natural Scenes from fMRI Patterns Using BigBiGAN (Brain→Image, BigGAN)  
[Paper](https://arxiv.org/abs/2011.12243)

#### 2.1.3 Diffusion-based

Brain-Diffuser: Natural Scene Reconstruction from fMRI Signals Using Generative Latent Diffusion (Brain→Image, Diffusion)  
[Paper](https://www.nature.com/articles/s41598-023-42891-8) [Code](https://github.com/ozcelikfu/brain-diffuser) [Dataset: NSD](https://naturalscenesdataset.org/)

Reconstructing the Mind’s Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors (MindEye) (Brain→Image, Diffusion)  
[Paper](https://arxiv.org/abs/2305.18274) [Project](https://medarc-ai.github.io/mindeye/) [Code](https://github.com/MedARC-AI/fMRI-reconstruction-NSD)

MindDiffuser: Controlled Image Reconstruction from Human Brain Activity with Semantic and Structural Diffusion (Brain→Image, Diffusion)  
[Paper](https://arxiv.org/abs/2308.04249) [Code](https://github.com/YingxingLu/MindDiffuser)

NeuralDiffuser: Neuroscience-Inspired Diffusion Guidance for fMRI Visual Reconstruction (Brain→Image, Diffusion)  
[Paper](https://arxiv.org/abs/2401.01713) [Code](https://github.com/neu-diffusion/NeuralDiffuser)

Mental Image Reconstruction from Human Brain Activity (Brain→Image, Diffusion)  
[Paper](https://www.sciencedirect.com/science/article/pii/S0893608023006470)

MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data (Brain→Image, Diffusion, Shared-subject)  
[Paper](https://arxiv.org/abs/2403.11207) [Project](https://medarc-ai.github.io/mindeye2/) [Code](https://github.com/MedARC-AI/MindEyeV2)

Bridging Brains and Concepts: Interpretable Visual Decoding from fMRI with Semantic Bottlenecks (Brain→Image, Diffusion, Semantic Bottleneck, NeurIPS 2025)  
[Paper](https://openreview.net/forum?id=K6ijewH34E) [PDF](https://openreview.net/pdf/167d5c3c08cdd7367883eeec0b26002c059215f8.pdf)

#### 2.1.4 Cross-subject / Few-shot / MoE

ZEBRA: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding (Brain→Image, Diffusion, Cross-Subject, NeurIPS 2025)  
[Paper](https://arxiv.org/abs/2510.27128) [PDF](https://openreview.net/pdf/7a4f583ef54685490be5c58986a3ad803aac087c.pdf) [Code](https://github.com/xmed-lab/ZEBRA)

MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding (Brain→Image, MoE, Cross-Subject, NeurIPS 2025)  
[Paper](https://arxiv.org/abs/2505.15946) [OpenReview](https://openreview.net/forum?id=fYSPRGmS6l) [Code](https://github.com/yuxiangwei0808/MoRE-Brain)

---

### 2.2 Brain → Video / Dynamic Scene

Visual Experience Reconstruction from Movie fMRI (Brain→Video)  
[Paper](https://doi.org/10.1016/j.cub.2011.01.031)

CLSR: Decoding Complex Video and Story Stimuli from fMRI (Brain→Video, Brain→Text)  
[Paper](https://doi.org/10.1038/s41593-023-01327-2)

*(Add more movie / video reconstruction or video-caption decoders here.)*

---

### 2.3 Brain → Text / Narrative

Semantic Reconstruction of Continuous Language from Non-Invasive Brain Recordings (Brain→Text, Narrative)  
[Paper](https://www.nature.com/articles/s41593-023-01304-9) [Code](https://github.com/HuthLab/semantic-decoding) [Dataset](https://openneuro.org/datasets/ds003020)

UniCoRN: Unified Cognitive Signal ReconstructioN Bridging Cognitive Signals and Human Language (Brain→Text, EEG+fMRI, LLM)  
[Paper](https://arxiv.org/abs/2307.05355)

Brain-Inspired fMRI-to-Text Decoding via Incremental and Wrap-Up Language Modeling (CogReader) (Brain→Text, LLM, NeurIPS 2025 Spotlight)  
[Paper](https://openreview.net/forum?id=REIo9ZLSYo) [PDF](https://openreview.net/pdf?id=REIo9ZLSYo) [Code](https://github.com/WENXUYUN/CogReader)

---

### 2.4 Brain → Audio / Music

*(Reserved for fMRI decoding of auditory scenes, speech, and music. Add works on music genre/affect decoding, sound category decoding, etc.)*

---

### 2.5 Multimodal & Foundation-Model-based Decoding

MindReader: Reconstructing Complex Images from Brain Activities (Brain→Image, CLIP, StyleGAN2)  
[Paper](https://arxiv.org/abs/2209.12951) [Code](https://github.com/yuvalsim/MindReader)

UMBRAE: Unified Multimodal Brain Decoding (Brain→Image, Brain→Text, Multimodal)  
[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01133.pdf) [Project](https://weihaox.github.io/UMBRAE)

*(Add more brain-conditional multimodal synthesis / VLM / LLM based decoders here.)*

---

### 2.6 Clinical / Cognitive / Mental-State Decoding

*(Reserved for works decoding emotion, cognitive load, disease markers, etc., from fMRI.)*

---

## 3. Related fMRI Modeling

### 3.1 Visual → fMRI Encoding & Data Augmentation

SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning (Visual→fMRI, Encoding, NeurIPS 2025 Poster)  
[Paper](https://arxiv.org/abs/2508.10298) [OpenReview](https://openreview.net/forum?id=ZTHYaSxqmq) [Code](https://github.com/MichaelMaiii/SynBrain)

*(Add more visual→fMRI encoders and synthetic-fMRI data augmentation works here.)*

### 3.2 Multimodal fMRI + EEG / MEG

Joint Modeling of fMRI and EEG Imaging Using Ordinary Differential Equation-Based Hypergraph Neural Networks (FE-NET) (fMRI+EEG, Hypergraph, Neural ODE, NeurIPS 2025)  
[PDF](https://openreview.net/pdf/053f8c5a43f7051852d82cdcb8ab742f69065ea2.pdf)

### 3.3 Representation Alignment & Analysis

*(For encoding-only LM-alignment, RSA / brain-score analysis, etc. To be filled.)*

---

## Contributing

Contributions are welcome!

1. 确认论文 **与 fMRI 强相关**，最好是解码（Brain→Image / Text / Video / Audio），或紧密相关的编码 / 多模态建模。
2. 选择合适的小节。
3. 按下面格式添加一条：

   ```markdown
   Paper Title (简单 tag)
   [Paper](...) [Code](...) [Dataset](...) [Project](...)
