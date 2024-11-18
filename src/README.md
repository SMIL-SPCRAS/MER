# Multi-Corpus Emotion Recognition Method based on Cross-Modal Gated Attention Fusion

## Overview

This study addresses key challenges in **automatic emotion recognition (ER)**, particularly the limitations of single-corpus training that hinder generalizability. To overcome these issues, the authors introduce a novel **multi-corpus, multimodal ER method** evaluated using a **leave-one-corpus-out (LOCO) protocol**. This approach incorporates fine-tuned encoders for audio, video, and text, combined through a context-independent gated attention mechanism for cross-modal feature fusion.

## Pipeline Overview

![Pipeline Diagram](https://smil-spcras.github.io/MER/static/img/pipeline.png)

## Components of the Proposed Method

- **Visual Encoder**: Fine-tuned ResNet-50.
- **Acoustic Encoder**: Fine-tuned emotional wav2vec2.
- **Linguistic Encoder**: Fine-tuned RoBERTa model.
- **Audio/Video Segment Aggregation**: Extraction of feature statistics (mean and standard deviation (STD) values).
- **Multimodal Feature Aggregation**: Cross-modal, context-independent gated attention mechanism.

## Key Results

The proposed method achieves **state-of-the-art performance** across multiple benchmark corpora, including **MOSEI, MELD, IEMOCAP, and AFEW**. The study reveals that models trained on **MELD** demonstrate superior cross-corpus generalization. Additionally, **AFEW** annotations show strong alignment with other corpora, resulting in the best cross-corpus performance. These findings validate the robustness and applicability of the method across diverse real-world scenarios.

## User Guide

### Accessing the Models

The pre-trained models are available [here](https://drive.google.com/drive/u/2/folders/1veuXhNBj0OKT-oa9-ge6UU1NxAVZGfNr).


### Inference

To predict emotions for your multimodal files, configure the `config.toml` file with paths to the models and files, and then run `python src/inference.py`.

### Training

To train the multimodal model, first extract features from your data `python src/avt_feature_extraction.py`.

Then, initiate training with `python src/train_avt_model.py`.

Ensure the `config.toml` file is properly configured for both steps.

## Citation

If you are using our models in your research, please consider to cite research:

<div class="highlight highlight-text-bibtex notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-k">@article</span>{<span class="pl-en">RYUMINA2024</span>,
  <span class="pl-s">title</span>        = <span class="pl-s"><span class="pl-pds">{</span>Multi-Corpus Emotion Recognition Method based on Cross-Modal Gated Attention Fusion<span class="pl-pds">}</span></span>,
  <span class="pl-s">author</span>       = <span class="pl-s"><span class="pl-pds">{</span>Elena Ryumina and Dmitry Ryumin and Alexandr Axyonov and Denis Ivanko and Alexey Karpov<span class="pl-pds">}</span></span>,
  <span class="pl-s">journal</span>      = <span class="pl-s"><span class="pl-pds">{</span>Pattern Recognition Letters<span class="pl-pds">}</span></span>,
  <span class="pl-s">year</span>         = <span class="pl-s"><span class="pl-pds">{</span>2024<span class="pl-pds">}</span></span>,
}</div>