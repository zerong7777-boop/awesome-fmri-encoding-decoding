# Awesome fMRI Decoding (Categorized)

A curated list of **fMRI-based brain decoding** papers and resources, focusing on **reconstructing images / videos / language / audio** from brain activity (plus a few highly related visual-encoding / synthesis works that directly support decoding).

If you find this repo helpful, please give it a ⭐ and consider contributing new papers via pull requests.

---

## Contents

- [1. Surveys and Overviews](#1-surveys-and-overviews)
- [2. Datasets and Benchmarks](#2-datasets-and-benchmarks)
- [3. Language / Narrative Decoding (Brain → Text)](#3-language--narrative-decoding-brain--text)
- [4. Visual Image Reconstruction (Brain → Image)](#4-visual-image-reconstruction-brain--image)
  - [4.1 Classical and Pre-Generative](#41-classical-and-pre-generative)
  - [4.2 GAN / VAE-based](#42-gan--vae-based)
  - [4.3 Diffusion-based Reconstruction](#43-diffusion-based-reconstruction)
  - [4.4 Cross-Subject and Generalizable Decoding](#44-cross-subject-and-generalizable-decoding)
  - [4.5 Interpretability and Concept-Level Decoding](#45-interpretability-and-concept-level-decoding)
  - [4.6 Visual-to-fMRI Synthesis and Data Augmentation](#46-visual-to-fmri-synthesis-and-data-augmentation)
- [5. Video and Dynamic Scene Decoding](#5-video-and-dynamic-scene-decoding)
- [6. Multimodal and Foundation-Model-based Decoding](#6-multimodal-and-foundation-model-based-decoding)
- [7. Audio and Music Decoding](#7-audio-and-music-decoding)
- [8. Clinical / Cognitive and Mental-State Decoding](#8-clinical--cognitive-and-mental-state-decoding)
- [9. Toolboxes and Tutorials](#9-toolboxes-and-tutorials)
- [10. Contributing](#10-contributing)
- [11. License](#11-license)

---

## 1. Surveys and Overviews

A Survey on fMRI-based Brain Decoding for Reconstructing Multimodal Stimuli  
[arXiv 2025](https://arxiv.org/abs/2503.15978) · [Project](https://github.com/LpyNow/BrainDecodingImage)

Brain-Conditional Multimodal Synthesis: A Survey and Taxonomy  
[IEEE TAI 2025](https://www.computer.org/csdl/journal/ai/2025/05/10798967/22EatqRGQxO) · [Project](https://github.com/MichaelMaiii/AIGC-Brain)

Brain Encoding and Decoding in fMRI with Bidirectional Deep Generative Models  
[Engineering 2019](https://www.engineering.org.cn/EN/10.1016/j.eng.2019.03.010)

(欢迎补充其他综述，比如 language-decoding / BCI 方向的 review。)

---

## 2. Datasets and Benchmarks

Natural Scenes Dataset (NSD)  
[Website](https://naturalscenesdataset.org/) · [Data](https://osf.io/9pjky/) · Visual images, large-scale high-res fMRI

Deep Image Reconstruction (DIR) dataset  
[OpenNeuro](https://openneuro.org/datasets/ds001506) · Natural images for the Kamitani 2019 reconstruction paper

Narratives / Story listening datasets  
[Website](https://www.narrativeslab.org/) · [OpenNeuro](https://openneuro.org/datasets/ds002345) · Audio stories, narrative comprehension

Semantic reconstruction of continuous language from non-invasive brain recordings – dataset  
[OpenNeuro](https://openneuro.org/datasets/ds003020) · Spoken stories, fMRI for language reconstruction

(Feel free to add VIM-1, BOLD5000, GOD / THINGS, CelebrityFace, etc.)

---

## 3. Language / Narrative Decoding (Brain → Text)

Semantic reconstruction of continuous language from non-invasive brain recordings  
[Nature Neuroscience 2023](https://www.nature.com/articles/s41593-023-01304-9) · [Code](https://github.com/HuthLab/semantic-decoding) · [Dataset](https://openneuro.org/datasets/ds003020)

Brain-Inspired fMRI-to-Text Decoding via Incremental and Wrap-Up Language Modeling (CogReader)  
[NeurIPS 2025 Spotlight](https://openreview.net/forum?id=REIo9ZLSYo) · [Code](https://github.com/WENXUYUN/CogReader)

(可以继续补充 BP-GPT、UniCoRN 等 LLM-based fMRI-to-text 工作。)

---

## 4. Visual Image Reconstruction (Brain → Image)

### 4.1 Classical and Pre-Generative

Reconstructing Natural Scenes from fMRI Patterns using Hierarchical Visual Features  
[NeuroImage 2011](https://doi.org/10.1016/j.neuroimage.2010.07.063)

### 4.2 GAN / VAE-based

Deep image reconstruction from human brain activity  
[PLoS Comput Biol 2019](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633) · [Code](https://github.com/KamitaniLab/DeepImageReconstruction) · [Dataset](https://openneuro.org/datasets/ds001506)

From voxels to pixels and back: Self-supervision in natural-image reconstruction from fMRI  
[NeurIPS 2019](https://arxiv.org/abs/1907.02431) · [Code](https://github.com/WeizmannVision/ssfmri2im)

Reconstructing Natural Scenes from fMRI Patterns using BigBiGAN  
[arXiv 2020](https://www.researchgate.net/publication/347156512_Reconstructing_Natural_Scenes_from_fMRI_Patterns_using_BigBiGAN)

### 4.3 Diffusion-based Reconstruction

High-resolution image reconstruction with latent diffusion models from human brain activity  
[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Takagi_High-Resolution_Image_Reconstruction_With_Latent_Diffusion_Models_From_Human_Brain_CVPR_2023_paper.html) · [Project](https://sites.google.com/view/stablediffusion-with-brain/) · [Code](https://github.com/yu-takagi/StableDiffusionReconstruction)

Reconstructing the Mind’s Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors (MindEye)  
[NeurIPS 2023 Spotlight](https://openreview.net/forum?id=rwrblCYb2A) · [Project](https://medarc-ai.github.io/mindeye/) · [Code](https://github.com/MedARC-AI/fMRI-reconstruction-NSD)

Contrast, Attend and Diffuse to Decode High-Resolution Visual Experiences from fMRI  
[NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/file/28dad4a70f748a2980998d3ed0f1b8d2-Paper-Conference.pdf) · [Code](https://github.com/soinx0629/vis_dec_neurips)

Brain-Diffuser: Natural scene reconstruction from fMRI signals using generative latent diffusion  
[Scientific Reports 2023](https://www.nature.com/articles/s41598-023-42891-8) · [Code](https://github.com/ozcelikfu/brain-diffuser) · [Dataset: NSD](https://naturalscenesdataset.org/)

MindDiffuser: Controlled Image Reconstruction from Human Brain Activity with Semantic and Structural Diffusion  
[ACM MM 2023](https://dl.acm.org/doi/10.1145/3581783.3613819) · [arXiv](https://arxiv.org/abs/2308.04249) · [Code](https://github.com/YingxingLu/MindDiffuser) · [Dataset: NSD](https://github.com/MichaelMaiii/AIGC-Brain?tab=readme-ov-file#dataset)

NeuralDiffuser: Neuroscience-Inspired Diffusion Guidance for fMRI Visual Reconstruction  
[IEEE TIP 2025](https://ieeexplore.ieee.org/document/10749645) · [arXiv](https://arxiv.org/abs/2401.01713) · [Code](https://github.com/HaoyyLi/NeuralDiffuser) · [Dataset: NSD](https://naturalscenesdataset.org/)

Mental image reconstruction from human brain activity  
[Neural Networks 2024](https://www.sciencedirect.com/science/article/pii/S0893608023006470)

### 4.4 Cross-Subject and Generalizable Decoding

MindEye2: Shared-subject models enable fMRI-to-image with one hour of data  
[ICML 2024](https://proceedings.mlr.press/v235/scotti24a.html) · [Project](https://medarc-ai.github.io/mindeye2/)  

ZEBRA: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding  
[NeurIPS 2025 Poster] (OpenReview / arXiv link to be added)

MoRE-Brain: Routed Mixture of Experts for Cross-Subject fMRI-to-Image Decoding  
[NeurIPS 2025 Poster] (OpenReview / arXiv link to be added)

(这一类主要关注 **跨被试泛化 / 少样本适配 / 共享模型** 等主题。)

### 4.5 Interpretability and Concept-Level Decoding

Mind Reader: Reconstructing complex images from brain activities  
[NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/bee5125b773414d3d6eeb4334fbc5453-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2210.01769) · [Code](https://github.com/sklin93/mind-reader) · [Dataset: NSD](https://github.com/sklin93/mind-reader#dataset)

Bridging Brains and Concepts: Interpretable Visual Decoding via Concept Bottlenecks  
[NeurIPS 2025 Poster] (OpenReview / arXiv link to be added)

### 4.6 Visual-to-fMRI Synthesis and Data Augmentation

SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning  
[NeurIPS 2025 Poster] (OpenReview / arXiv link to be added)

(这一小节偏 **image→fMRI 合成 / 数据增强 / 对比学习**，但与解码性能提升直接相关。)

---

## 5. Video and Dynamic Scene Decoding

Visual experience reconstruction from movie fMRI  
[Current Biology 2011](https://doi.org/10.1016/j.cub.2011.01.031)

CLSR: Decoding complex video and story stimuli from fMRI  
[Nature Neuroscience 2023](https://doi.org/10.1038/s41593-023-01327-2)

(可继续加入 fMRI-to-video、fMRI-conditioned video diffusion 等工作。)

---

## 6. Multimodal and Foundation-Model-based Decoding

MindReader: Reconstructing complex images from brain activities  
[NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/bee5125b773414d3d6eeb4334fbc5453-Abstract-Conference.html) · [Code](https://github.com/sklin93/mind-reader) · CLIP + StyleGAN2, multimodal latent space

UMBRAE: Unified Multimodal Brain Decoding  
[ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01133.pdf) · [Project](https://weihaox.github.io/UMBRAE)

(这里可以收集更多利用 CLIP / LMM / VLM 等大模型做多模态 brain decoding 的工作。)

---

## 7. Audio and Music Decoding

(占位：添加基于 fMRI 的音乐 / 音频类别 / 情绪解码等工作。)

---

## 8. Clinical / Cognitive and Mental-State Decoding

(占位：情绪、心理状态、疾病 marker 等基于 fMRI decoding 的应用。)

---

## 9. Toolboxes and Tutorials

DeepImageReconstruction codebase  
[GitHub](https://github.com/KamitaniLab/DeepImageReconstruction) – End-to-end pipeline for visual fMRI → image reconstruction.

awesome-brain-decoding (general, multi-modality)  
[GitHub](https://github.com/NeuSpeech/awesome-brain-decoding)

Awesome Brain Encoding & Decoding  
[GitHub](https://github.com/subbareddy248/Awesome-Brain-Encoding--Decoding)

(也可以列出预处理工具箱、NSD 读取工具、教学 tutorial 等。)

---

## 10. Contributing

Contributions are welcome!  
If you want to add or update a paper:

1. 确认它主要是 **fMRI-based decoding**（至少有比较核心的 fMRI decoding 组件）。
2. 选择正确的一级 & 二级类别。
3. 按照下面的格式添加到对应小节：

   ```markdown
   Paper Title  
   [Venue Year](paper_link) · [Code](code_link) · [Project](project_link) · [Dataset](dataset_link)
