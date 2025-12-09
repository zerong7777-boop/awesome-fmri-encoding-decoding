# Awesome fMRI Encoding, Decoding, and Representations

A curated list of fMRI-centric **encoding models, decoding frameworks, and representational analyses**, covering language/narrative, visual reconstruction, video, audio/music, mental state, and BCI applications.

If you find this repo helpful, please give it a ⭐ and consider contributing new papers via pull requests.

---

## Contents
- [0. Tag legend](#0-tag-legend)
- [1. Surveys and Overviews](#1-surveys-and-overviews)
  - [1.1 Visual and Multimodal fMRI Decoding & Reconstruction](#11-visual-and-multimodal-fmri-decoding--reconstruction)
  - [1.2 Language and Narrative Neural Decoding](#12-language-and-narrative-neural-decoding)
  - [1.3 General Brain Decoding, BCI, and Deep Learning Perspectives](#13-general-brain-decoding-bci-and-deep-learning-perspectives)
- [2. Datasets and Benchmarks](#2-datasets-and-benchmarks)
  - [2.1 Static Visual Image Datasets](#21-static-visual-image-datasets)
  - [2.2 Language and Narrative Datasets](#22-language-and-narrative-datasets)
  - [2.3 Video, Affective, and Multimodal Naturalistic Datasets](#23-video-affective-and-multimodal-naturalistic-datasets)
  - [2.4 Large-Scale Population Imaging Cohorts (Pretraining-Oriented)](#24-large-scale-population-imaging-cohorts-pretraining-oriented)
  - [2.5 Clinical, Psychiatric, and Cognitive/Developmental Cohorts (Downstream Tasks)](#25-clinical-psychiatric-and-cognitivedevelopmental-cohorts-downstream-tasks) 
- [3. Language / Narrative Decoding (Brain → Text)](#3-language--narrative-decoding-brain--text)
  - [3.1 Encoding-Model & Candidate-Scoring Decoders (fMRI)](#31-encoding-model--candidate-scoring-decoders-fmri)
  - [3.2 Representation-Alignment & Embedding-Space Decoders (fMRI)](#32-representation-alignment--embedding-space-decoders-fmri)
  - [3.3 Generative & LLM-Based Brain-to-Text Decoders (fMRI)](#33-generative--llm-based-brain-to-text-decoders-fmri)
  - [3.4 Non-fMRI but Influential Brain-to-Text Decoding](#34-non-fmri-but-influential-brain-to-text-decoding)
- [4. Visual Image Reconstruction (Brain → Image)](#4-visual-image-reconstruction-brain--image)
  - [4.1 Classical and Pre-Deep-Learning Reconstruction](#41-classical-and-pre-deep-learning-reconstruction)
  - [4.2 Deep Generative Reconstruction with Learned Image Priors](#42-deep-generative-reconstruction-with-learned-image-priors)
  - [4.3 Cross-Subject and Universal Visual Decoders / Encoders](#43-cross-subject-and-universal-visual-decoders--encoders)
  - [4.4 Interpretability and Concept-Level Decoding](#44-interpretability-and-concept-level-decoding)
- [5. Video and Dynamic Scene Decoding](#5-video-and-dynamic-scene-decoding)
  - [5.1 Classical Encoding and Semantic Decoding Frameworks for Movies](#51-classical-encoding-and-semantic-decoding-frameworks-for-movies)
  - [5.2 Representation-Alignment and Retrieval-Based Video Decoders](#52-representation-alignment-and-retrieval-based-video-decoders)
  - [5.3 Deep Generative fMRI-to-Video Reconstruction](#53-deep-generative-fmri-to-video-reconstruction)
- [6. Multimodal and Foundation-Model-based Decoding](#6-multimodal-and-foundation-model-based-decoding)
  - [6.1 Unified Vision–Language / Multimodal Decoders](#61-unified-visionlanguage--multimodal-decoders)
  - [6.2 Video-Oriented and Retrieval-Based Multimodal Decoding](#62-video-oriented-and-retrieval-based-multimodal-decoding)
- [7. Audio and Music Decoding](#7-audio-and-music-decoding)
  - [7.1 fMRI-Based Music and Audio Decoding](#71-fmri-based-music-and-audio-decoding)
  - [7.2 Non-fMRI but Influential Audio / Music Decoding](#72-non-fmri-but-influential-audio--music-decoding)
- [8. Mental-State, Cognitive, and Clinical Decoding (Cross-Modality)](#8-mental-state-cognitive-and-clinical-decoding-cross-modality)
  - [8.1 Affective and Emotion State Decoding](#81-affective-and-emotion-state-decoding)
  - [8.2 Cognitive Task and Performance Decoding](#82-cognitive-task-and-performance-decoding)
  - [8.3 Clinical and Biomarker-Oriented fMRI Decoding](#83-clinical-and-biomarker-oriented-fmri-decoding)
  - [8.4 Methodological and Conceptual Perspectives on Mental-State Decoding](#84-methodological-and-conceptual-perspectives-on-mental-state-decoding)
- [9. Toolboxes and Awesome Lists](#9-toolboxes-and-awesome-lists)
  - [9.1 Decoding / Reconstruction Codebases](#91-decoding--reconstruction-codebases)
  - [9.2 Preprocessing, Analysis, and Utility Libraries](#92-preprocessing-analysis-and-utility-libraries)
  - [9.3 Awesome Lists and Related Curations](#93-awesome-lists-and-related-curations)
- [10. Contributing](#10-contributing)



---
## 0. Tag legend

- [FM] foundation-style / multi-task / subject-agnostic models
- [X-SUBJ] explicit cross-subject / subject-agnostic decoders
- [CLINICAL] clinical / disease-focused cohorts or decoders
- [AFFECT] emotion / affective state decoding
- [COG] cognitive task / performance decoding
- [BCI] communication / control-oriented brain–computer interfaces

---


## 1. Surveys and Overviews

> **Scope:** Global surveys / reviews / tutorials that describe the overall landscape of fMRI decoding and brain-conditioned generative modeling (images, videos, language, multimodal, BCIs, etc.).

### 1.1 Visual and Multimodal fMRI Decoding & Reconstruction

A Survey on fMRI-based Brain Decoding for Reconstructing Multimodal Stimuli  
[[arXiv 2025](https://arxiv.org/abs/2503.15978)]

Brain-Conditional Multimodal Synthesis: A Survey and Taxonomy  
[[IEEE TAI 2025](https://www.computer.org/csdl/journal/ai/2025/05/10798967/22EatqRGQxO)] [[Project](https://github.com/MichaelMaiii/AIGC-Brain)]

Visual Image Reconstruction from Brain Activity via Latent Representation  
[[Annual Review of Vision Science 2025](https://www.annualreviews.org/content/journals/10.1146/annurev-vision-110423-023616)]

Review of visual neural encoding and decoding methods in fMRI  
[[Journal of Image and Graphics 2023](https://www.cjig.cn/en/article/doi/10.11834/jig.220525)]

Visualizing the mind’s eye: a future perspective on image reconstruction from brain signals  
[[Psychoradiology 2023](https://doi.org/10.1093/psyrad/kkad022)]

Deep Generative Models in Brain Encoding and Decoding  
[[Engineering 2019](https://doi.org/10.1016/j.eng.2019.03.011)]

---

### 1.2 Language and Narrative Neural Decoding

Progress, challenges and future of linguistic neural decoding with deep learning  
[[Communications Biology 2025](https://www.nature.com/articles/s42003-025-08511-z)]

Artificial intelligence based multimodal language decoding from brain activity: A review  
[[Brain Research Bulletin 2023](https://doi.org/10.1016/j.brainresbull.2023.110713)]

---

### 1.3 General Brain Decoding, BCI, and Deep Learning Perspectives

Non-Invasive Brain-Computer Interfaces: State of the Art and Trends  
[[IEEE Reviews in Biomedical Engineering 2025](https://doi.org/10.1109/RBME.2024.3449790)]

Deep learning approaches for neural decoding across multiple scales  
[[Briefings in Bioinformatics 2021](https://doi.org/10.1093/bib/bbaa053)]

A Survey on Brain Encoding and Decoding  
[[IJCAI 2021](https://www.ijcai.org/proceedings/2021/594)]

---

## 2. Datasets and Benchmarks

> **Scope:** Public fMRI datasets / benchmarks (vision, language, audio, etc.), organized by dataset rather than method, plus large-scale cohorts for pretraining and downstream cognitive/clinical tasks.

### 2.1 Static Visual Image Datasets

Natural Scenes Dataset (NSD) – 7T high-resolution fMRI responses to tens of thousands of natural images.  
[[Nature Neuroscience 2022](https://www.nature.com/articles/s41593-021-00962-x)] [[Website](https://naturalscenesdataset.org/)] [[Data](https://osf.io/9pjky/)]

Natural Object Dataset (NOD) – large-scale fMRI dataset with 57k naturalistic images (ImageNet / COCO) from 30 participants.  
[[Scientific Data 2023](https://www.nature.com/articles/s41597-023-02471-x)] [[OpenNeuro ds004496](https://openneuro.org/datasets/ds004496)]

THINGS-data / THINGS-fMRI – multimodal object-vision dataset (fMRI, MEG, behavior) over ~1.8k object concepts.  
[[eLife 2023](https://elifesciences.org/articles/82580)] [[OpenNeuro ds004192](https://openneuro.org/datasets/ds004192)] [[Collection](https://doi.org/10.25452/figshare.plus.c.6161151)]

BOLD5000 – slow event-related fMRI dataset for ~5k images drawn from COCO / ImageNet / SUN.  
[[Scientific Data 2019](https://www.nature.com/articles/s41597-019-0052-3)] [[Website](https://bold5000-dataset.github.io/website/)] [[OpenNeuro ds001499](https://openneuro.org/datasets/ds001499)]

Deep Image Reconstruction (DIR) dataset – single-subject fMRI for natural images used in the Kamitani deep image reconstruction work.  
[[PLoS Comput Biol 2019](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633)] [[OpenNeuro ds001506](https://openneuro.org/datasets/ds001506)]

---

### 2.2 Language and Narrative Datasets

Narratives / Story listening datasets – multi-subject naturalistic spoken-story fMRI.  
[[Scientific Data 2021](https://www.nature.com/articles/s41597-021-01033-3)] [[Website](https://www.narrativeslab.org/)] [[OpenNeuro ds002345](https://openneuro.org/datasets/ds002345)]

Semantic reconstruction of continuous language – dataset used in the Nature Neuroscience 2023 semantic decoding paper.  
[[Nature Neuroscience 2023](https://www.nature.com/articles/s41593-023-01304-9)] [[OpenNeuro ds003020](https://openneuro.org/datasets/ds003020)]

Natural language fMRI dataset for voxelwise encoding models – five multi-session natural-language listening datasets for voxelwise encoding.  
[[Scientific Data 2023](https://www.nature.com/articles/s41597-023-02437-z)] [[GitHub](https://github.com/HuthLab/deep-fMRI-dataset)]

---

### 2.3 Video, Affective, and Multimodal Naturalistic Datasets

BOLD Moments Dataset (BMD) – video fMRI responses to ~1k short naturalistic clips with rich object / action / text metadata.  
[[Nature Communications 2024](https://www.nature.com/articles/s41467-024-50310-3)] [[OpenNeuro ds005165](https://openneuro.org/datasets/ds005165)] [[Code](https://github.com/blahner/BOLDMomentsDataset)]

[AFFECT] [COG] Spacetop – multimodal fMRI dataset with >100 participants, combining movie viewing with a broad battery of cognitive / affective tasks and physiology.  
[[Scientific Data 2025](https://www.nature.com/articles/s41597-025-05154-x)] [[OpenNeuro ds005256](https://openneuro.org/datasets/ds005256)]

[AFFECT] Emo-FilM – film-based fMRI with dense emotion annotations and concurrent physiological recordings.  
[[Scientific Data 2025](https://www.nature.com/articles/s41597-025-04803-5)] [[OpenNeuro ds004892](https://openneuro.org/datasets/ds004892)]

---


### 2.4 Large-Scale Population Imaging Cohorts (Pretraining-Oriented)

[COG] Human Connectome Project (HCP, Young Adult S1200) – multimodal MRI for ~1.2k healthy young adults (3T structural, resting-state and task fMRI, diffusion MRI), with a subset scanned at 7T and some MEG; widely used for connectivity, representation learning, and population-based decoding.  
[[NeuroImage 2013](https://doi.org/10.1016/j.neuroimage.2013.05.041)] [[S1200 Data Releases](https://www.humanconnectome.org/study/hcp-young-adult/data-releases)] [[S1200 Reference Manual (PDF)](https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf)]

[CLINICAL] UK Biobank Imaging – very large population cohort (>100k participants targeted) with structural MRI, resting-state and task fMRI, and other imaging (cardiac, abdominal, DXA, carotid ultrasound); primarily designed for population health and genetics, but increasingly used for large-scale brain encoder pretraining and brain–phenotype prediction.  
[[NeuroImage 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5086094/)] [[Nature Communications 2020 imaging enhancement](https://www.nature.com/articles/s41467-020-15948-9)] [[Imaging project overview](https://www.ukbiobank.ac.uk/taking-part/participant-opportunities/imaging-project/)]

[COG] ABCD Study (Adolescent Brain Cognitive Development) – longitudinal cohort (~10k+ children/adolescents) with structural MRI, diffusion, resting-state and task fMRI (e.g., MID, SST, n-back), plus rich behavioral, cognitive, and environmental measures; useful for developmental decoding and pretraining.  
[[Dev Cogn Neurosci 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5999559/)] [[ABCD Study website](https://abcdstudy.org/)] [[Imaging documentation](https://docs.abcdstudy.org/latest/documentation/imaging/)]

---

### 2.5 Clinical, Psychiatric, and Cognitive/Developmental Cohorts (Downstream Tasks)

[CLINICAL] PPMI (Parkinson’s Progression Markers Initiative) – longitudinal, multi-center cohort with extensive clinical, multi-modal imaging (structural MRI, DaTscan, and additional MRI sequences at some sites), biospecimens and genetics for Parkinson’s disease and at-risk individuals; standard benchmark for PD progression modeling and biomarker discovery.  
[[Prog Neurobiol 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6292383/)] [[Study website](https://www.ppmi-info.org/)] [[Data access](https://www.ppmi-info.org/access-data-specimens/download-data)]

[CLINICAL] ADNI (Alzheimer’s Disease Neuroimaging Initiative) – multi-center longitudinal study with structural MRI, PET, some resting-state fMRI derivatives, cognitive assessments, genetics and CSF/blood biomarkers for MCI / Alzheimer’s and controls; widely used for neurodegenerative disease prediction and progression modeling.  
[[ADNI neuroimaging overview](https://adni.loni.usc.edu/data-samples/adni-data/neuroimaging/)] [[ADNI MRI component](https://adni.loni.usc.edu/data-samples/adni-data/neuroimaging/mri/)] [[ADNI Data portal](https://adni.loni.usc.edu/data-samples/adni-data/)]

[CLINICAL] [COG] ABIDE I / II (Autism Brain Imaging Data Exchange) – multi-site repositories aggregating structural MRI and resting-state fMRI for individuals with autism spectrum disorder and controls; standard benchmark for ASD classification, connectome-based decoding, and cross-site generalization.  
[[ABIDE overview (NeuroImage 2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4162310/)] [[ABIDE portal](http://fcon_1000.projects.nitrc.org/indi/abide/)] [[ABIDE II](http://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html)] [[ABIDE Preprocessed](https://preprocessed-connectomes-project.org/abide/)]

[CLINICAL] [COG] ADHD-200 – multi-site structural MRI and resting-state fMRI dataset for ADHD and typically-developing controls, originally released for the ADHD-200 Global Competition; extensively used as a benchmark for rs-fMRI–based psychiatric diagnosis and generalization across scanners/sites.  
[[ADHD-200 portal](http://fcon_1000.projects.nitrc.org/indi/adhd200/)] [[Global Competition summary](https://pmc.ncbi.nlm.nih.gov/articles/PMC3460316/)] [[ADHD-200 Preprocessed](https://preprocessed-connectomes-project.org/adhd200/)]

---

## 3. Language / Narrative Decoding (Brain → Text)

> **Scope:** Brain → text decoding, where the output is language (words, sentences, continuous narratives) rather than images, video, or other modalities.  
> Sections 3.1–3.3 cover **fMRI-based** approaches, grouped by decoding strategy:
> - **§3.1** – likelihood-based / **encoding-model + candidate scoring** decoders for a discrete set of words/sentences;
> - **§3.2** – decoders that map brain activity into **continuous semantic embedding spaces** and perform retrieval or linear decoding;
> - **§3.3** – **open-vocabulary generative** decoders that use pretrained sequence models / LLMs.
>
> We place a work in **§3.1** if it explicitly uses an encoding model to score a discrete candidate set (even if embeddings are used internally).  
> Works that directly regress to a continuous representation space and then perform retrieval or linear decoding, **without explicit candidate scoring**, are grouped in **§3.2**.  
> **§3.4** collects influential **non-fMRI (EEG/MEG)** brain-to-text works that are methodologically relevant for non-invasive decoding.

### 3.1 Encoding-Model & Candidate-Scoring Decoders (fMRI)

Toward a universal decoder of linguistic meaning from brain activation  
[[Nature Communications 2018](https://www.nature.com/articles/s41467-018-03068-4)] [[OSF project](https://osf.io/crwz7/)]

Semantic reconstruction of continuous language from non-invasive brain recordings  
[[Nature Neuroscience 2023](https://www.nature.com/articles/s41593-023-01304-9)] [[Code](https://github.com/HuthLab/semantic-decoding)] [[Dataset](https://openneuro.org/datasets/ds003020)]

How Many Bytes Can You Take Out Of Brain-To-Text Decoding?  
*(Information-theoretic evaluation and augmentation of fMRI→text decoders)*  
[[arXiv 2024](https://arxiv.org/abs/2405.14055)]

Mind captioning: Evolving descriptive text of mental content from human brain activity  
[[Science Advances 2025](https://www.science.org/doi/10.1126/sciadv.adw1464)] [[Code](https://github.com/horikawa-t/MindCaptioning)] [[OpenNeuro ds005191](https://openneuro.org/datasets/ds005191)]


---

### 3.2 Representation-Alignment & Embedding-Space Decoders (fMRI)

Decoding naturalistic experiences from human brain activity via distributed representations of words  
[[NeuroImage 2018](https://www.sciencedirect.com/science/article/pii/S105381191730664X)]

Towards Sentence-Level Brain Decoding with Distributed Representations  
[[AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4685)]

Fine-grained Neural Decoding with Distributed Word Representations  
[[Information Sciences 2020](https://www.sciencedirect.com/science/article/pii/S0020025519307820)]

Neural Encoding and Decoding With Distributed Sentence Representations  
[[IEEE TNNLS 2021](https://doi.org/10.1109/TNNLS.2020.3027595)]

MapGuide: A Simple yet Effective Method to Reconstruct Continuous Language from Brain Activities  
[[NAACL 2024](https://aclanthology.org/2024.naacl-long.211/)] 

High-level visual representations in the human brain are aligned with large language models  
[[Nature Machine Intelligence 2025](https://www.nature.com/articles/s42256-025-01072-0)] [[arXiv](https://arxiv.org/abs/2209.11737)] [[Code](https://github.com/adriendoerig/visuo_llm)]

---

### 3.3 Generative & LLM-Based Brain-to-Text Decoders (fMRI)

Towards Brain-to-Text Generation: Neural Decoding with Pre-trained Encoder-Decoder Models  
[[NeurIPS 2021 (AI4Science Workshop](https://openreview.net/forum?id=13IJlk221xG)]

[FM] UniCoRN: Unified Cognitive Signal ReconstructioN bridging cognitive signals and human language  
[[ACL 2023](https://aclanthology.org/2023.acl-long.741/)] 

[X-SUBJ] Decoding Continuous Character-based Language from Non-invasive Brain Recordings  
[[bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2024.03.19.585656v1)] [[arXiv](https://arxiv.org/abs/2403.11183)] [[Dataset](https://openneuro.org/datasets/ds006630)]

[FM] BrainDEC: A Multimodal LLM for the Non-Invasive Decoding of Text from Brain Recordings  
[[Information Fusion 2025](https://doi.org/10.1016/j.inffus.2025.103589)] [[Code](https://github.com/Hmamouche/brain_decode)]

Brain-Inspired fMRI-to-Text Decoding via Incremental and Wrap-Up Language Modeling (CogReader)  
[[NeurIPS 2025 Spotlight](https://openreview.net/forum?id=REIo9ZLSYo)] [[Code](https://github.com/WENXUYUN/CogReader)]

[FM] [X-SUBJ] MindLLM: A Subject-Agnostic and Versatile Model for fMRI-to-Text Decoding  
[[ICML 2025 (poster)](https://openreview.net/forum?id=EiAQrilPYP)] [[arXiv 2025](https://arxiv.org/abs/2502.15786)] [[Code](https://github.com/Graph-and-Geometric-Learning/MindLLM)]

MindGPT: Interpreting What You See With Non-Invasive Brain Recordings  
[[IEEE TIP 2025](https://ieeexplore.ieee.org/document/11018227)] [[Code](https://github.com/JxuanC/MindGPT)]

Open-vocabulary Auditory Neural Decoding Using fMRI-prompted LLM (Brain Prompt GPT / BP-GPT)  
[[ICASSP 2025 preprint](https://arxiv.org/abs/2405.07840)] [[Code](https://github.com/1994cxy/BP-GPT)]

[FM] Generative language reconstruction from brain recordings (BrainLLM)  
[[Communications Biology 2025](https://www.nature.com/articles/s42003-025-07731-7)] [[Code](https://github.com/YeZiyi1998/Brain-language-generation)]

[FM] [X-SUBJ] fMRI-LM: Towards a Universal Foundation Model for Multi-Task Brain Decoding  
[[arXiv](https://www.arxiv.org/abs/2511.21760)]

[FM] Brain-language fusion enables interactive neural readout and in-silico experimentation (CorText / CorText-AMA)  
[[arXiv](https://arxiv.org/abs/2509.23941)]

---

### 3.4 Non-fMRI but Influential Brain-to-Text Decoding

[X-SUBJ] Decoding speech perception from non-invasive brain recordings *(MEG/EEG contrastive decoding of perceived speech, strong reference for non-invasive language decoding)*  
[[Nature Machine Intelligence 2023](https://www.nature.com/articles/s42256-023-00714-5)] [[Code](https://github.com/facebookresearch/brainmagick)]

[X-SUBJ] Towards decoding individual words from non-invasive brain recordings *(EEG/MEG – non-fMRI but highly influential for non-invasive brain-to-text)*  
[[Nature Communications 2025](https://www.nature.com/articles/s41467-025-65499-0)]

[BCI] [X-SUBJ] Brain-to-Text Decoding: A Non-invasive Approach via Typing (Brain2Qwerty) *(sentence-level typing decoded from EEG/MEG)*  
[[arXiv 2025](https://arxiv.org/abs/2502.17480)] [[Project page](https://ai.meta.com/research/publications/brain-to-text-decoding-a-non-invasive-approach-via-typing/)]


---

## 4. Visual Image Reconstruction (Brain → Image)

> **Scope:** Static image reconstruction from brain activity (mostly fMRI).  
> Subsections **4.1–4.2** are organized by the main **image prior** (pre-deep vs deep generative models), while **4.3–4.4** cut across these backbones to highlight **cross-subject / universal decoders** and **interpretability / concept-level analyses**.  
> Some works naturally fit multiple views (e.g., diffusion-based and cross-subject); we cross-reference them via tags such as **[FM]** and **[X-SUBJ]** rather than duplicating full entries.

---

### 4.1 Classical and Pre-Deep-Learning Reconstruction

> Early approaches that do **not** rely on modern deep generative image models, often based on hand-crafted features or simpler encoding/decoding pipelines.

Visual image reconstruction from human brain activity using a combination of multiscale local image decoders  
[[Neuron 2008](https://doi.org/10.1016/j.neuron.2008.11.004)]

Reconstructing Natural Scenes from fMRI Patterns using Hierarchical Visual Features  
[[NeuroImage 2011](https://doi.org/10.1016/j.neuroimage.2010.07.063)]

---

### 4.2 Deep Generative Reconstruction with Learned Image Priors

> fMRI→image reconstruction that uses **deep generative models** as image priors (GAN, latent diffusion, Stable Diffusion variants, etc.).

Deep image reconstruction from human brain activity  
[[PLoS Comput Biol 2019](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633)] [[Code](https://github.com/KamitaniLab/DeepImageReconstruction)] [[Dataset](https://openneuro.org/datasets/ds001506)]

From voxels to pixels and back: Self-supervision in natural-image reconstruction from fMRI  
[[NeurIPS 2019](https://arxiv.org/abs/1907.02431)] [[Code](https://github.com/WeizmannVision/ssfmri2im)]

High-resolution image reconstruction with latent diffusion models from human brain activity  
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Takagi_High-Resolution_Image_Reconstruction_With_Latent_Diffusion_Models_From_Human_Brain_CVPR_2023_paper.html)] [[Project](https://sites.google.com/view/stablediffusion-with-brain/)] [[Code](https://github.com/yu-takagi/StableDiffusionReconstruction)]

Seeing Beyond the Brain: Conditional Diffusion Model with Sparse Masked Modeling for Vision Decoding (MinD-Vis)  
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Seeing_Beyond_the_Brain_Conditional_Diffusion_Model_With_Sparse_Masked_CVPR_2023_paper.pdf)] [[Project](https://mind-vis.github.io/)]

Reconstructing the Mind’s Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors (MindEye)  
[[NeurIPS 2023](https://arxiv.org/abs/2305.18274)] [[Project](https://medarc-ai.github.io/mindeye/)] [[Code](https://github.com/MedARC-AI/fMRI-reconstruction-NSD)]

MindDiffuser: Controlled Image Reconstruction from Human Brain Activity with Semantic and Structural Diffusion  
[[ACM MM 2023](https://dl.acm.org/doi/10.1145/3581783.3613819)] [[arXiv](https://arxiv.org/abs/2308.04249)] [[Code](https://github.com/YingxingLu/MindDiffuser)]

Dual-Guided Brain Diffusion Model: Natural Image Reconstruction from Human Visual Stimulus fMRI (DBDM)  
[[Bioengineering 2023](https://www.mdpi.com/2306-5354/10/10/1117)]

Mental image reconstruction from human brain activity  
[[Neural Networks 2024](https://www.sciencedirect.com/science/article/pii/S0893608023006470)]

NeuralDiffuser: Neuroscience-Inspired Diffusion Guidance for fMRI Visual Reconstruction  
[[IEEE TIP 2025](https://ieeexplore.ieee.org/document/10749645)] [[arXiv](https://arxiv.org/abs/2401.01713)]

Balancing Semantic and Structural Decoding for fMRI-to-Image Reconstruction  
[[Expert Systems with Applications 2025](https://www.sciencedirect.com/science/article/abs/pii/S0957417425034517)]

---

### 4.3 Cross-Subject and Universal Visual Decoders / Encoders

> Brain→image decoders and image→fMRI encoders that explicitly target **cross-subject / cross-site generalization**, few-shot adaptation, or universal representations.  
> Tagged with **[X-SUBJ]** when cross-subject generalization is a core focus, and **[FM]** when the model is positioned as a more general-purpose brain encoder/decoder.  
> Some of these also belong conceptually to §4.2 (diffusion-based reconstruction) or §6 (foundation-style encoders); we list them here when **population-level modeling** is a central contribution.

[X-SUBJ] MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data  
[[ICML 2024](https://proceedings.mlr.press/v235/scotti24a.html)] [[arXiv](https://arxiv.org/abs/2403.11207)] [[Project](https://medarc-ai.github.io/mindeye2/)] [[Code](https://github.com/MedARC-AI/MindEyeV2)]

[X-SUBJ] ZEBRA: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding  
[[NeurIPS 2025](https://openreview.net/pdf/7a4f583ef54685490be5c58986a3ad803aac087c.pdf)] [[Code](https://github.com/xmed-lab/ZEBRA)]

[FM] [X-SUBJ] Psychometry: An Omnifit Model for Image Reconstruction from Human Brain Activity  
[[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Quan_Psychometry_An_Omnifit_Model_for_Image_Reconstruction_from_Human_Brain_CVPR_2024_paper.html)] [[arXiv](https://arxiv.org/abs/2403.20022)]

[X-SUBJ] NeuroPictor: Refining fMRI-to-Image Reconstruction via Multi-individual Pretraining and Multi-level Modulation  
[[ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06746.pdf)] [[arXiv](https://arxiv.org/abs/2403.18211)] [[Project](https://jingyanghuo.github.io/neuropictor/)]

[X-SUBJ] Wills Aligner: Multi-Subject Collaborative Brain Visual Decoding  
[[AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33554)] [[arXiv](https://arxiv.org/abs/2404.13282)]

[X-SUBJ] BrainGuard: Privacy-Preserving Multisubject Image Reconstructions from Brain Activities  
[[AAAI 2025 (Oral)](https://ojs.aaai.org/index.php/AAAI/article/view/33579)] [[arXiv](https://arxiv.org/abs/2501.14309)] [[Project](https://github.com/kunzhan/brainguard)]

[X-SUBJ] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding  
[[NeurIPS 2025](https://openreview.net/forum?id=fYSPRGmS6l)] [[arXiv](https://arxiv.org/abs/2505.15946)] [[Code](https://github.com/yuxiangwei0808/MoRE-Brain)]

[X-SUBJ] Inter-individual and inter-site neural code conversion without shared stimuli  
*(General-purpose cross-subject / cross-site alignment that can support various decoding tasks beyond visual reconstruction.)*  
[[Nature Computational Science 2025](https://doi.org/10.1038/s43588-025-00826-5)]

Self-Supervised Natural Image Reconstruction and Large-Scale Semantic Classification from Brain Activity  
[[NeuroImage 2022](https://www.sciencedirect.com/science/article/pii/S105381192200249X)]

[FM] [X-SUBJ] The Wisdom of a Crowd of Brains: A Universal Brain Encoder  
[[arXiv 2024](https://arxiv.org/abs/2406.12179)]

SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning  
[[arXiv 2025](https://arxiv.org/abs/2508.10298)] [[NeurIPS 2025](https://openreview.net/forum?id=ZTHYaSxqmq)]

---

### 4.4 Interpretability and Concept-Level Decoding

> Brain→image pipelines that explicitly emphasize **interpretability**, concept-level representations, or analysis of how much information generative priors actually extract from the brain (e.g., concept bottlenecks, probing, attribution analyses).

MindReader: Reconstructing complex images from brain activities  
[[NeurIPS 2022](https://arxiv.org/abs/2209.12951)] [[Code](https://github.com/yuvalsim/MindReader)]

Bridging Brains and Concepts: Interpretable Visual Decoding from fMRI with Semantic Bottlenecks  
[[NeurIPS 2025 Poster](https://openreview.net/forum?id=K6ijewH34E)] [[PDF](https://openreview.net/pdf?id=K6ijewH34E)]

BrainBits: How Much of the Brain are Generative Reconstruction Methods Using?  
[[NeurIPS 2024](https://openreview.net/forum?id=KAAUvi4kpb)] [[arXiv](https://arxiv.org/abs/2411.02783)] [[Code](https://github.com/czlwang/BrainBits)]


---

## 5. Video and Dynamic Scene Decoding

> **Scope:** Decoding **continuous movies / dynamic visual scenes** from brain activity.  
> This includes **movie encoding/decoding frameworks** and feature-based models, **representation-alignment and retrieval-based decoders**, and **deep generative fMRI-to-video reconstruction**.  
> Video-oriented pipelines that heavily rely on **multimodal LMMs / foundation models** are cross-referenced in **§6.2**, while works focusing on higher-level **affect or cognitive state trajectories** induced by movies are mostly indexed in **§8**.

---

### 5.1 Classical Encoding and Semantic Decoding Frameworks for Movies

> Encoding-model and decoding pipelines for natural movies, often predicting voxel responses from visual features and then decoding semantic content or categories.

Reconstructing visual experiences from brain activity evoked by natural movies  
[[Current Biology 2011](https://www.sciencedirect.com/science/article/pii/S0960982211009377)]

Neural encoding and decoding with deep learning for dynamic natural vision  
[[Cerebral Cortex 2018](https://academic.oup.com/cercor/article/28/12/4136/4560155)]

The Algonauts Project 2021 Challenge: How the Human Brain Makes Sense of a World in Motion  
*(Benchmark challenge for predicting fMRI responses to >1k short everyday videos.)*  
[[arXiv 2021](https://arxiv.org/abs/2104.13714)] [[Challenge](http://algonauts.csail.mit.edu/)]

---

### 5.2 Representation-Alignment and Retrieval-Based Video Decoders

> Approaches that map fMRI into a **shared embedding space** (e.g., clip-level or text-level representations) and then perform video **retrieval** or matching, often with the help of multimodal large models.

Mind2Word: Towards Generalized Visual Neural Representations for High-Quality Video Reconstruction  
– Maps fMRI into a sequence of pseudo-words in a text embedding space, and then uses a video generator for high-quality reconstruction.  
[[Expert Systems with Applications 2025](https://www.sciencedirect.com/science/article/pii/S095741742502771X)]

Decoding the Moving Mind: Multi-Subject fMRI-to-Video Retrieval with MLLM Semantic Grounding  
– Multi-subject fMRI-to-video retrieval using multimodal large language models to ground semantic similarity between brain activity and candidate clips.  
[[bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.04.07.647335v1)]

> These works also appear in **§6.2 Video-Oriented and Retrieval-Based Multimodal Decoding**, where their multimodal / foundation-model aspects are emphasized.

---

### 5.3 Deep Generative fMRI-to-Video Reconstruction

> Models that aim to **reconstruct full video sequences** (or high-frame-rate approximations) from fMRI, typically using deep video generators or diffusion models conditioned on brain activity.

Reconstructing rapid natural vision with fMRI-conditional video generative adversarial network  
[[Cerebral Cortex 2022](https://academic.oup.com/cercor/article/32/20/4502/6515038)]

A Penny for Your (visual) Thoughts: Self-Supervised Reconstruction of Natural Movies from Brain Activity  
[[arXiv 2022](https://arxiv.org/abs/2206.03544)]

Cinematic Mindscapes: High-quality Video Reconstruction from Brain Activity (Mind-Video)  
[[NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/4e5e0daf4b05d8bfc6377f33fd53a8f4-Paper-Conference.pdf)] [[Project](https://www.mind-video.com/)]

Animate Your Thoughts: Decoupled Reconstruction of Dynamic Natural Vision from Slow Brain Activity (Mind-Animator)  
[[ICLR 2025](https://openreview.net/forum?id=BpfsxFqhGa)] [[arXiv](https://arxiv.org/abs/2405.03280)] [[Project](https://mind-animator-design.github.io/)]

---

## 6. Multimodal and Foundation-Model-based Decoding

> **Scope:** Methods that treat brain activity as one modality inside **large pretrained models** (CLIP, Stable Diffusion, VLMs, LMMs, etc.) and/or build **foundation-style brain encoders** designed to work across tasks, datasets, or modalities.  
> We list a work here if (i) a large pretrained model is **central** to the decoding pipeline, and (ii) the model is explicitly **multi-task**, **multimodal**, or **subject-/dataset-agnostic** beyond a single specific task.  
> This section is **method-oriented** and mainly serves as a **second index**: concrete models remain organized under modality-specific sections (§3–§5) and are tagged with [FM] / [X-SUBJ].

### 6.1 Unified Vision–Language / Multimodal Decoders

> Subsection 6.1 groups **foundation-style brain decoders** that unify multiple output modalities (e.g., images + text) or tasks within one model.  
> The entries below are **indexed here** from §§3–4; please refer to those sections for task-specific context and details.

**Language / narrative decoders (see §3.3)**  

- [FM] BrainLLM – *Generative language reconstruction from brain recordings*  
  → listed under [§3.3](#33-generative--llm-based-brain-to-text-decoders-fmri)  

- [FM] BrainDEC – *A Multimodal LLM for the Non-Invasive Decoding of Text from Brain Recordings*  
  → see [§3.3](#33-generative--llm-based-brain-to-text-decoders-fmri)  

- [FM] [X-SUBJ] MindLLM – *A Subject-Agnostic and Versatile Model for fMRI-to-Text Decoding*  
  → see [§3.3](#33-generative--llm-based-brain-to-text-decoders-fmri)  

- [FM] UniCoRN – *Unified Cognitive Signal ReconstructioN bridging cognitive signals and human language*  
  → see [§3.3](#33-generative--llm-based-brain-to-text-decoders-fmri)  

- [FM] fMRI-LM – *Towards a Universal Foundation Model for Multi-Task Brain Decoding*  
  → see [§3.3](#33-generative--llm-based-brain-to-text-decoders-fmri)  

- [FM] Brain-language fusion (CorText / CorText-AMA) – *interactive neural readout and in-silico experimentation with LLMs*  
  → see [§3.3](#33-generative--llm-based-brain-to-text-decoders-fmri)


**Visual reconstruction & universal encoders (see §4.3–§4.4)**  

- [FM] [X-SUBJ] Psychometry – *An Omnifit Model for Image Reconstruction from Human Brain Activity*  
  → listed under [§4.3](#43-cross-subject-and-universal-visual-decoders--encoders)  

- [X-SUBJ] MindEye2 – *Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data*  
  → see [§4.3](#43-cross-subject-and-universal-visual-decoders--encoders)  

- [X-SUBJ] ZEBRA – *Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding*  
  → see [§4.3](#43-cross-subject-and-universal-visual-decoders--encoders)  

- [X-SUBJ] NeuroPictor / Wills Aligner / BrainGuard / MoRE-Brain – cross-subject visual decoders and privacy-preserving or MoE-based variants  
  → see [§4.3](#43-cross-subject-and-universal-visual-decoders--encoders)  

- [FM] [X-SUBJ] The Wisdom of a Crowd of Brains – *A Universal Brain Encoder*  
  → listed under [§4.3](#43-cross-subject-and-universal-visual-decoders--encoders)  

- Self-Supervised Natural Image Reconstruction and Large-Scale Semantic Classification from Brain Activity  
  → see [§4.3](#43-cross-subject-and-universal-visual-decoders--encoders)  

- SynBrain – *Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning*  
  → see [§4.3](#43-cross-subject-and-universal-visual-decoders--encoders)  

> For task-specific details and links, please refer back to the corresponding sections (§3–§4). Here we group them by their **foundation-style / cross-subject** perspective.

---

### 6.2 Video-Oriented and Retrieval-Based Multimodal Decoding

> Models that explicitly use **vision–language / multimodal foundation models** (CLIP, VLMs, LMMs) and do not fit cleanly into a single output-modality section.  
> Many of these bridge **video reconstruction / retrieval** with **textual or conceptual grounding**, and so are cross-referenced with §5.

[FM] [X-SUBJ] UMBRAE: Unified Multimodal Brain Decoding  
– Unified decoder that aligns fMRI with CLIP-like multimodal representations, supporting image, text, and category decoding within one framework.  
[[ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01133.pdf)] [[Project](https://weihaox.github.io/UMBRAE/)] [[Code](https://github.com/weihaox/UMBRAE)]

BrainCLIP: Bridging Brain and Visual-Linguistic Representation via CLIP for Generic Natural Visual Stimulus Decoding  
– Uses CLIP as a shared vision–language embedding space, mapping fMRI into CLIP space to enable image and concept decoding.  
[[arXiv 2023](https://arxiv.org/abs/2302.12971)] [[Code](https://github.com/YulongBonjour/BrainCLIP)]

Modality-Agnostic fMRI Decoding of Vision and Language  
– Single decoder jointly trained on image- and text-evoked fMRI, operating in a shared embedding space for modality-agnostic decoding.  
[[ICLR 2024 Workshop on Representational Alignment](https://openreview.net/forum?id=7gWQL0hTrX)] [[arXiv](https://arxiv.org/abs/2403.11771)]

[FM] [X-SUBJ] Brain Harmony: A Multimodal Foundation Model Unifying Morphology and Function into 1D Tokens  
– Tokenizes structural and functional brain data into a shared sequence space, enabling multi-task decoding and transfer across cohorts.  
[[NeurIPS 2025](https://openreview.net/pdf/80edac1ff79b10252bcd8be5794855fadbd39ea9.pdf)] [[Code](https://github.com/hzlab/Brain-Harmony)]

Mind2Word: Towards Generalized Visual Neural Representations for High-Quality Video Reconstruction  
– Maps fMRI into a pseudo-word sequence in a text embedding space, then uses a video generator / diffusion model for high-quality **video reconstruction**.  
[[Expert Systems with Applications 2025](https://www.sciencedirect.com/science/article/pii/S095741742502771X)]

Decoding the Moving Mind: Multi-Subject fMRI-to-Video Retrieval with MLLM Semantic Grounding [X-SUBJ]  
– Multi-subject fMRI-to-video retrieval using multimodal LLMs to ground semantic similarity between brain activity and candidate clips.  
[[bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.04.07.647335v1)]

> For additional video-focused methods that **do not** rely heavily on multimodal LMMs or foundation models, see **§5. Video and Dynamic Scene Decoding**.

---

## 7. Audio and Music Decoding

> **Scope:** Decoding approaches where the **output is sound** (music, general audio). We focus on **fMRI-based** music and audio decoders, and list a few **non-fMRI but closely related** works (iEEG / EEG / MEG) when they have become field-shaping references. Speech and narrative decoding that produce **text** is covered in §3, and multimodal VLM / diffusion pipelines in §6.

### 7.1 fMRI-Based Music and Audio Decoding

Brain2Music: Reconstructing Music from Human Brain Activity  
[[arXiv 2023](https://arxiv.org/abs/2307.11078)] [[Project](https://google-research.github.io/seanet/brain2music/)]

Reconstructing Music Perception from Brain Activity Using a Prior-Guided Diffusion Model  
[[Scientific Reports 2025](https://www.nature.com/articles/s41598-025-26095-w)]

R&B – Rhythm and Brain: Cross-Subject Music Decoding from fMRI via Prior-Guided Diffusion Model  
[[Preprint 2025](https://doi.org/10.21203/rs.3.rs-7301336/v1)]

Identifying Musical Pieces from fMRI Data Using Encoding and Decoding Models  
[[Scientific Reports 2018](https://www.nature.com/articles/s41598-018-20732-3)]

### 7.2 Non-fMRI but Influential Music / Audio Decoding

Music Can Be Reconstructed from Human Auditory Cortex Activity Using Nonlinear Decoding Models  
*(iEEG; music reconstruction, often cited together with fMRI work)*  
[[PLOS Biology 2023](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002176)]

Neural Decoding of Music from the EEG  
*(EEG combined with fMRI-informed source localisation for music reconstruction and identification)*  
[[Scientific Reports 2023](https://www.nature.com/articles/s41598-022-27361-x)]

Decoding Reveals the Neural Representation of Perceived and Imagined Musical Sounds  
*(MEG; decoding perceived and imagined melodies)*  
[[PLOS Biology 2024](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002858)]

---


## 8. Mental-State, Cognitive, and Clinical Decoding (Cross-Modality)

> **Scope:** fMRI-based decoding of **mental states** (emotion, affect, spontaneous thought, cognitive performance) and **clinically relevant phenotypes** (diagnosis, risk, progression), plus methodological work that directly targets mental-state decoding.  

### 8.1 Affective and Emotion State Decoding  [AFFECT]

Brain Decoding of Spontaneous Thought: Predictive Modeling of Self-Relevance and Valence Using Personal Narratives  
[[PNAS 2024](https://www.pnas.org/doi/10.1073/pnas.2401959121)]

Machine Learning for Classifying Affective Valence from fMRI: A Systematic Review  
[[Affective Science 2025](https://link.springer.com/article/10.1007/s44163-025-00377-8)]

### 8.2 Cognitive Task and Performance Decoding  [COG]

BrainCodec: Neural fMRI Codec for the Decoding of Cognitive Brain States  
[[arXiv 2024](https://arxiv.org/abs/2410.04383)] [[Code](https://github.com/amano-k-lab/BrainCodec)]

Brain Decoding of the Human Connectome Project Tasks in a Dense Individual fMRI Dataset  
[[NeuroImage 2023](https://doi.org/10.1016/j.neuroimage.2023.120395)]

Explainable Deep-Learning Framework: Decoding Brain Task and Predicting Individual Performance in False-Belief Tasks at Early Childhood Stage  
[[Preprint 2024](https://www.biorxiv.org/content/10.1101/2024.02.29.582682v1)]

Scaling Vision Transformers for Functional MRI with Flat Maps  
*(Provides a scalable ViT-based backbone that has been applied to a variety of decoding tasks, including cognitive-state prediction.)*  
[[NeurIPS 2025 (Foundation Models for the Brain and Body Workshop)](https://openreview.net/forum?id=L0CpmKEVHw)] [[arXiv](https://arxiv.org/abs/2510.13768)] [[Code](https://github.com/MedARC-AI/fmri-fm)]

### 8.3 Clinical and Biomarker-Oriented fMRI Decoding  [CLINICAL]

Advances in Functional Magnetic Resonance Imaging-Based Brain Decoding and Its Clinical Applications  
[[Psychoradiology 2025](https://doi.org/10.1093/psyrad/kkaf007)]

*(See also §2.5 for large-scale clinical / psychiatric cohorts that are commonly used as downstream benchmarks.)*

### 8.4 Methodological and Conceptual Perspectives on Mental-State Decoding

Benchmarking Explanation Methods for Mental State Decoding with Deep Learning Models  
[[NeuroImage 2023](https://doi.org/10.1016/j.neuroimage.2023.120109)] [[Code](https://github.com/athms/xai-brain-decoding-benchmark)]

Limits of Decoding Mental States with fMRI  
*(Foundational cautionary perspective on what can and cannot be inferred from decoding models.)*  
[[NeuroImage 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9238276/)]

---

---

## 9. Toolboxes and Awesome Lists

> **Scope:** General-purpose codebases for brain decoding and fMRI analysis, preprocessing pipelines, and other curated awesome lists relevant to fMRI-based brain decoding.

### 9.1 Decoding / Reconstruction Codebases

DeepImageReconstruction codebase  
[[GitHub](https://github.com/KamitaniLab/DeepImageReconstruction)]

End-to-end Deep Image Reconstruction  
[[GitHub](https://github.com/KamitaniLab/End2EndDeepImageReconstruction)]

Inter-individual Deep Image Reconstruction  
[[GitHub](https://github.com/KamitaniLab/InterIndividualDeepImageReconstruction)]

semantic-decoding (language reconstruction)  
[[GitHub](https://github.com/HuthLab/semantic-decoding)]

MindEye (fMRI-to-image with contrastive + diffusion priors)  
[[GitHub](https://github.com/MedARC-AI/fMRI-reconstruction-NSD)]

MindEye2 implementation (shared-subject fMRI-to-image)  
[[GitHub](https://github.com/MedARC-AI/MindEyeV2)]

Brain-Diffuser implementation  
[[GitHub](https://github.com/ozcelikfu/brain-diffuser)]

UMBRAE implementation (unified multimodal brain decoding)  
[[GitHub](https://github.com/weihaox/UMBRAE)]

BrainCLIP implementation  
[[GitHub](https://github.com/YulongBonjour/BrainCLIP)]

Brain2Music implementation  
[[GitHub](https://github.com/google-research/google-research/tree/master/brain2music)]

BrainCodec: neural fMRI codec for cognitive-state decoding  
[[GitHub](https://github.com/amano-k-lab/BrainCodec)]

---

### 9.2 Preprocessing, Analysis, and Utility Libraries

fMRIPrep – robust preprocessing pipeline for task / resting-state fMRI  
[[GitHub](https://github.com/nipreps/fmriprep)] [[Docs](https://fmriprep.org/)]

XCP-D – post-processing (denoising, connectivity, QA) for fMRIPrep outputs  
[[GitHub](https://github.com/PennLINC/xcp_d)]

NiBabies – fMRIPrep-style preprocessing for infant / neonatal MRI  
[[GitHub](https://github.com/nipreps/nibabies)]

Nilearn – machine learning & decoding tools for neuroimaging in Python  
[[GitHub](https://github.com/nilearn/nilearn)] [[Docs](https://nilearn.github.io/)]

BrainIAK – Brain Imaging Analysis Kit (advanced fMRI analyses)  
[[GitHub](https://github.com/brainiak/brainiak)] [[Docs](https://brainiak.org/docs/)] [[Tutorials](https://brainiak.org/tutorials/)]

fmralign – functional alignment and inter-subject mapping  
[[GitHub](https://github.com/Parietal-INRIA/fmralign)]

bdpy – Brain Decoder Toolbox in Python  
[[GitHub](https://github.com/KamitaniLab/bdpy)]

NextBrain: A probabilistic histological atlas of the human brain for MRI segmentation 
[[Nature 2025](https://doi.org/10.1038/s41586-025-09708-2)]

---

### 9.3 Awesome Lists and Related Curations

awesome-brain-decoding (general, multi-modality)  
[[GitHub](https://github.com/NeuSpeech/awesome-brain-decoding)]

Awesome Brain Encoding & Decoding  
[[GitHub](https://github.com/subbareddy248/Awesome-Brain-Encoding--Decoding)]

Awesome Brain Graph Learning with GNNs  
[[GitHub](https://github.com/XuexiongLuoMQ/Awesome-Brain-Graph-Learning-with-GNNs)]

Awesome Neuroimaging in Python (nibabel, nilearn, MNE, etc.)  
[[GitHub](https://github.com/ofgulban/awesome-neuroimaging-in-python)]

*(You can further extend this section with visualization tools, BIDS utilities, and more decoding-specific codebases as the ecosystem grows.)*

---

## 10. Contributing

Contributions are welcome! 🎉  

**Recommended entry format:**

```markdown
Paper Title  
[[Venue Year](paper_link)] [[Code](code_link)] [[Project](project_link)] [[Dataset](dataset_link)]
