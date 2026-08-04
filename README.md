# IGDM
**Official code**: _Instruction-guided distribution maximization for general few-shot intent recognition_ (IEEE Transactions on Pattern Analysis and Machine Intelligence).  
## Abstract
Intent recognition captures the deep needs of users from natural language, and few-shot intent recognition addresses data scarcity. Existing methods treat single- and multi-label intent recognition as separate classification tasks, and discriminative models typically require fine-tuning when generalizing to new intents, which limits generality. To this end, in this study, we re-think the single- and multi-label intent recognition task as a unified natural language generation task rather than a discriminative classification problem. After instruction pre-training on a general intent dataset, the language model (LM) directly generalizes to few-shot target data without fine-tuning. However, limited data and large distribution shifts severely challenge generalization. Therefore, in this paper, we propose an Instruction-Guided Distribution Maximization (IGDM) method to improve the robustness of the model and enhance generalization. IGDM prompts the model to learn a broader decision boundary by enlarging the maximum error of LM between the user utterance and the corresponding intent. Formally, we perform a two-level optimization strategy to obtain the inner instruction-guided sample distribution maximization and the outer model error minimization, respectively. The inner optimization objective is the sample distribution maximization, which is achieved by performing gradient ascent according to LM in instruction learning. The outer optimization objective is the LM, whose optimization is implemented by gradient descent in instruction learning combined with samples from the distribution maximized according to the LM. Extensive theoretical and inference proofs demonstrate the superior robustness and generalization capabilities of IGDM. To verify this, we conduct comprehensive experiments across 20 single-label and 7 multi-label widely used intent recognition benchmarks. The results demonstrate that IGDM achieves superior performance over existing methods across all benchmarks.

## Distributional Difference
Feature distribution differences. Red points represent target datasets (ATIS, SNIPS, Bank77) and blue points represent general datasets (ACID, BCS, HINT3, MCID, XSID). The T-SNE tool is used to visualize the distribution deviation, and the convex hull contour is drawn and the contour line based on Gaussian kernel density estimation is added to describe the distribution boundary of the data more clearly. This significant distribution divergence poses a major challenge to generalization on new tasks.
<div align="center">
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/distribution_g_atis_page-0001.jpg" width="260" />
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/distribution_g_snips_page-0001.jpg" width="260" />
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/distribution_g_banking_page-0001.jpg" width="260" />
</div>

## Framework
![image](https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/method_page-0001.jpg)

## How to use
You can directly run meta_train.py to train a new model, and then use meta_test.py to conduct the generalization test. Alternatively, you can directly use the pre-trained model weights that we have provided. It can be downloaded from <a href="https://pan.baidu.com/s/1RH8aRFLjySL1c4J5imfcXA?pwd=igdm" target="_blank">Link</a>.

## Results
### Single-label fuzzy matching
<div align="center">
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/single_cosine_page-0001.jpg" width="260" />
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/single_jaccard_page-0001.jpg" width="260" />
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/single_leven_page-0001.jpg" width="260" />
</div>

### Multi-label fuzzy matching
<div align="center">
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/multi_cosine_page-0001.jpg" width="260" />
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/multi_jaccard_page-0001.jpg" width="260" />
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/multi_leven_page-0001.jpg" width="260" />
</div>

### Cosine accuracy of new intent recognition
Left is IGDM, and Right is GenPINT.
<div align="center">
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/new_intent_igdm_page-0001.jpg" width="300" />
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/new_intent_genpint_page-0001.jpg" width="300" />
</div>

## Citation
If you use this work in your research, please cite it as follows:

```bibtex
@ARTICLE{11516309,
  author={Yang, Shun and Du, YaJun and He, XiaoFei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Instruction-Guided Distribution Maximization for General Few-Shot Intent Recognition}, 
  year={2026},
  volume={},
  number={},
  pages={1-18},
  doi={10.1109/TPAMI.2026.3692528},
}
```
