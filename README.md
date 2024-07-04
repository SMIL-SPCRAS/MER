# Multi-Corpus Emotion Recognition Method based on Cross-Modal Gated Attention Fusion

The official repository for "Multi-Corpus Emotion Recognition Method based on Cross-Modal Gated Attention Fusion", [Pattern Recognition Letters](https://www.sciencedirect.com/journal/pattern-recognition-letters) (submitted)

## Abstract

Automatic emotion recognition methods are critical to human-computer interaction. However, current methods suffer from limited applicability due to their tendency to overfit on single-corpus datasets. This overfitting reduces real-world effectiveness of the methods when faced with new unseen corpora. We propose the first multi-corpus multimodal emotion recognition method with high generalizability evaluated through a leave-one-corpus-out protocol. The method uses three fine-tuned encoders per modality (audio, video, and text) and a decoder employing context-independent gated attention to combine features from all three modalities. The research is conducted on four benchmark corpora: MOSEI, MELD, IEMOCAP, and AFEW. The proposed method achieves the state-of-the-art results on the corpora and establishes the first baselines for multi-corpus studies. We demonstrate that due to the MELD rich emotional expressiveness across three modalities, the models trained on it exhibit the best generalization ability when applied to corpora used. We also reveal that the AFEW annotation better correlates with the annotations of MOSEI, MELD, IEMOCAP and shows the best cross-corpus performance as it is consistent with the widely-accepted concepts of basic emotions.

## Acknowledgments

Parts of this project page were adopted from the [Nerfies](https://nerfies.github.io/) page.

## Website License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
