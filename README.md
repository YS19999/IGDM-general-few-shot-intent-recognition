# IGDM
**Official code**: _Instruction-guided distribution maximization for general few-shot intent recognition_ (IEEE Transactions on Pattern Analysis and Machine Intelligence).  
## Abstract
Intent recognition captures the deep needs of users from natural language, and few-shot intent recognition addresses data scarcity. Existing methods treat single- and multi-label intent recognition as separate classification tasks, and discriminative models typically require fine-tuning when generalizing to new intents, which limits generality. To this end, in this study, we re-think the single- and multi-label intent recognition task as a unified natural language generation task rather than a discriminative classification problem. After instruction pre-training on a general intent dataset, the language model (LM) directly generalizes to few-shot target data without fine-tuning. However, limited data and large distribution shifts severely challenge generalization. Therefore, in this paper, we propose an Instruction-Guided Distribution Maximization (IGDM) method to improve the robustness of the model and enhance generalization. IGDM prompts the model to learn a broader decision boundary by enlarging the maximum error of LM between the user utterance and the corresponding intent. Formally, we perform a two-level optimization strategy to obtain the inner instruction-guided sample distribution maximization and the outer model error minimization, respectively. The inner optimization objective is the sample distribution maximization, which is achieved by performing gradient ascent according to LM in instruction learning. The outer optimization objective is the LM, whose optimization is implemented by gradient descent in instruction learning combined with samples from the distribution maximized according to the LM. Extensive theoretical and inference proofs demonstrate the superior robustness and generalization capabilities of IGDM. To verify this, we conduct comprehensive experiments across 20 single-label and 7 multi-label widely used intent recognition benchmarks. The results demonstrate that IGDM achieves superior performance over existing methods across all benchmarks.

## Distributional Difference
Feature distribution differences. Red points represent target datasets (ATIS, SNIPS, Bank77) and blue points represent general datasets (ACID, BCS, HINT3, MCID, XSID). The T-SNE tool is used to visualize the distribution deviation, and the convex hull contour is drawn and the contour line based on Gaussian kernel density estimation is added to describe the distribution boundary of the data more clearly. This significant distribution divergence poses a major challenge to generalization on new tasks.
<p style="text-align: center;">
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/distribution_g_atis_page-0001.jpg" width="300" />
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/distribution_g_snips_page-0001.jpg" width="300" />
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/distribution_g_banking_page-0001.jpg" width="300" />
</p>

## Framework
![image](https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/method_page-0001.jpg)

## Results
### Single-label fuzzy matching
<p style="text-align: center;">
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/single_cosine_page-0001.jpg" width="300" />
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/single_jaccard_page-0001.jpg" width="300" />
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/single_level_page-0001.jpg" width="300" />
</p>

### Multi-label fuzzy matching
<p style="text-align: center;">
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/multi_cosine_page-0001.jpg" width="300" />
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/multi_jaccard_page-0001.jpg" width="300" />
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/multi_level_page-0001.jpg" width="300" />
</p>

### Cosine accuracy of new intent recognition
<p style="text-align: center;">
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/new_intent_igdm_page-0001.jpg" width="450" />
  <img src="https://github.com/YS19999/IGDM-general-few-shot-intent-recognition/blob/main/images/new_intent_genpint_page-0001.jpg" width="450" />
</p>
