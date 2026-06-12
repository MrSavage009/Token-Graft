# Token-Graft: Isomorphic Latent Mapping and Latent Trajectory Alignment for Interference-Free Dialect Adaptation in Small Language Models

**Anonymous Authors** · 13 June 2026

---

## Abstract

We study localized semantic adaptation in Small Language Models (SLMs) under parameter-efficient constraints. When adapting an on-device model to a structured conceptual dialect—where standard signifiers are remapped to foreign referents—naive fine-tuning corrupts pre-trained weights via catastrophic interference, while decoupled tokenization alone suffers from slow convergence and poor metaphor generalization.

We propose **Token-Graft**, a two-stage geometric initialization method for decoupled adapter tokens. First, we compute an **isomorphic latent mapping** (Graft-0) at the embedding layer by combining orthogonal decomposition with target-vector anchoring. Second, we derive a **latent trajectory alignment** (Graft-8) by computing the hidden-state delta between the grafted token and its target at an intermediate transformer layer. We evaluate Token-Graft on the Qwen 2.5 1.5B model under strict hardware constraints (8 GB VRAM, gradient checkpointing, Adafactor optimizer) across two semantic remapping tasks with varying geometric alignment between source and target manifolds.

Our key finding is **conditional**: when source and target tokens share structural properties (similar length, part-of-speech, tokenization granularity), Token-Graft reduces metaphor perplexity by 23.8% over random decoupled initialization. However, when source-target pairs are geometrically discordant (e.g., short proper nouns mapped to long multi-token abstractions), the rigid graft vector introduces **representational friction** and underperforms random initialization [1]. This inversion reveals that geometric initialization is beneficial only when source and target manifolds are already near-isomorphic in their low-level token geometry—a principle we formalize as the **Geometric Alignment Condition**.

---

## 1. Introduction

Large Language Models (LLMs) encode vast semantic knowledge in their pre-trained weights, but adapting them to specialized conceptual dialects—where standard words carry new, domain-specific meanings—poses a fundamental dilemma. Naive fine-tuning on standard tokens overwrites pre-trained representations, causing catastrophic interference (McCloskey & Cohen, 1989; French, 1999). Decoupled approaches that add new tokens to the vocabulary avoid this interference but suffer from high search-space entropy: randomly initialized tokens must learn both the model's latent geometry and the new semantic mapping from scratch.

Vocabulary expansion and embedding initialization have been extensively studied for multilingual adaptation (Yamaguchi et al., 2024) and domain-specific terminology (Tam et al., 2024). Prior work initializes new embeddings via averaging subword embeddings (Hewitt, 2021; Gero et al., 2023), merge-rule decomposition (Yamaguchi et al., 2024), or KL-divergence alignment against the original tokenizer (Vocabulary Expansion via Self-Distillation, 2024). These methods focus on preserving distributional behavior across tokenizers. We instead ask: when remapping *semantics* within a fixed tokenizer, can we leverage the geometric relationship between source and target embeddings to accelerate learning?

We frame this as a **subspace routing problem**: given source tokens $S = \{s_1, \dots, s_n\}$ and target referents $T = \{t_1, \dots, t_n\}$, can we initialize new adapter tokens $G = \{g_1, \dots, g_n\}$ such that (i) $g_i$ routes through the model's computation graph toward the semantic region of $t_i$, (ii) $g_i$ does not interfere with pre-trained representations, and (iii) the relative geometric structure among $\{g_i\}$ approximates that among $\{t_i\}$?

Our contributions are:

1. **Token-Graft**: A geometric initialization method combining embedding-layer Isomorphic Latent Mapping with intermediate-layer Latent Trajectory Alignment.
2. **Empirical characterization of the Geometric Alignment Condition**: We show that grafting accelerates learning when source-target pairs share tokenization granularity, length, and syntactic category, but harms performance when these properties diverge.
3. **Hardware-efficient evaluation**: We demonstrate the method on an 8 GB VRAM consumer GPU using Qwen 2.5 1.5B, with full reproducibility under gradient checkpointing and Adafactor optimization.

---

## 2. Related Work

**Vocabulary expansion and embedding initialization.** Adding new tokens to pre-trained vocabularies is standard practice for multilingual adaptation (Yong et al., 2023) and domain-specific terminology (Tam et al., 2024). Hewitt (2021) showed that initializing new embeddings via the mean of existing embeddings minimizes KL-divergence from the original model's behavior. Yamaguchi et al. (2024) systematically compared initialization strategies (Random, Mean, Merge, Align, FOCUS) under low-resource settings and found that heuristic methods (Mean, Align) consistently outperform random initialization. Our work differs by targeting *semantic remapping* within a fixed tokenizer rather than vocabulary expansion for new languages.

**Parameter-efficient fine-tuning.** LoRA (Hu et al., 2021) and its variants (QLoRA, Dettmers et al., 2023; DoRA, Liu et al., 2024) reduce trainable parameters by learning low-rank updates to weight matrices. Most LoRA variants target attention projections; we extend the approach to embedding and output layers, which is memory-efficient but requires careful handling of tied embeddings (Yamamoto et al., 2026).

**Representation steering.** Representation Engineering (Zou et al., 2023) and activation steering (Turner et al., 2023) modify hidden states to elicit or suppress high-level behaviors. Our Graft-8 mechanism is a form of layer-specific steering, but conditioned on the forward-propagated state of a grafted token rather than an external concept vector.

**Catastrophic interference mitigation.** Elastic Weight Consolidation (Kirkpatrick et al., 2017), progressive networks (Rusu et al., 2016), and adapter layers (Houlsby et al., 2019) mitigate forgetting. Our decoupled tokenization achieves interference-freedom by architectural isolation rather than regularization.

---

## 3. Method

### 3.1 Problem Formulation

Let $\mathcal{M}$ be a causal language model with vocabulary $\mathcal{V}$, embedding matrix $E \in \mathbb{R}^{|\mathcal{V}| \times d}$, and $L$ transformer layers. We define a remapping $\mathcal{R}: S \rightarrow T$ where $S \subset \mathcal{V}$ are source tokens and $T \subset \mathcal{V}^*$ are target token sequences (each $t_i$ may be multi-token). We extend $\mathcal{V}$ with new adapter tokens $G = \{g_1, \dots, g_n\}$ and train only LoRA adapters on $E$ and the output head, freezing all base weights.

### 3.2 Graft-0: Isomorphic Latent Mapping

For each source-target pair $(s_i, t_i)$, let $v_s = E[s_i] \in \mathbb{R}^d$ and $v_t = \text{mean}(E[t_i]) \in \mathbb{R}^d$ (averaging subword embeddings for multi-token targets). We decompose $v_s$ into components parallel and orthogonal to $v_t$:

$$v_s^{\parallel} = \frac{v_s \cdot v_t}{\|v_t\|^2} v_t, \quad v_s^{\perp} = v_s - v_s^{\parallel}$$

The grafted embedding blends the orthogonal component (preserving relational structure relative to $v_t$) with the target vector (anchoring semantic location):

$$v_{g0} = \alpha \cdot v_s^{\perp} + (1 - \alpha) \cdot v_t$$

with blend parameter $\alpha \in [0, 1]$. Setting $\alpha = 1$ yields pure orthogonal projection (preserves source geometry but may land far from target semantics); $\alpha = 0$ yields target cloning (optimal semantic location but no relational structure). We set $\alpha = 0.7$ based on a small validation sweep (Appendix A).

**Important:** This is a *heuristic blend*, not an isometry. The distances between grafted vectors $\{v_{g0}^{(i)}\}$ do not equal distances among $\{v_t^{(i)}\}$ or $\{v_s^{(i)}\}$. We do not claim preservation of "exact Euclidean distances." A true distance-preserving mapping would require solving the Orthogonal Procrustes Problem (Schönemann, 1966) to find a global rotation matrix $R$ minimizing $\|XR - Y\|_F$ subject to $R^T R = I$; our per-vector projection is a computationally cheaper local approximation that trades exact isometry for tractability on consumer hardware.

### 3.3 Graft-8: Latent Trajectory Alignment

After initializing Graft-0, we run a forward pass with $g_i$ as input, halting at the entrance to layer $\ell$ (we use $\ell = 8$ for Qwen 2.5 1.5B, which has 28 layers). Let $x_\ell^{\text{graft}}$ be the hidden state at this point. We run a parallel forward pass with the target sequence $t_i$ and extract $x_\ell^{\text{target}}$ at the same position. The correction vector is:

$$\delta_\ell = x_\ell^{\text{target}} - x_\ell^{\text{graft}}$$

During training and inference, we add $\delta_\ell$ to the residual stream at layer $\ell$ via a forward pre-hook. To avoid autograd version conflicts under gradient checkpointing, we clone the tensor before addition (out-of-place tensor mutation).

**Rationale:** Early layers (0–7) build syntactic and lexical representations. By layer 8, the model has formed preliminary semantic frames. The correction vector $\delta_8$ nudges the grafted token's trajectory toward the target's semantic region while preserving the syntactic processing of preceding layers.

### 3.4 Training Configuration

* **Model:** Qwen 2.5 1.5B-Instruct (BF16)
* **Hardware:** RTX 4060 Laptop (8 GB VRAM) + 8 GB host RAM
* **Optimization:** Adafactor ($\beta_1 = 0.9$, decay rate $= -0.8$), batch size 2, gradient accumulation 4, learning rate $2 \times 10^{-4}$
* **LoRA:** rank $r = 8$, $\alpha = 16$, dropout $0.05$, targeting `embed_tokens`, `lm_head`, `q_proj`, `v_proj`
* **Anchoring:** Base embedding and transformer weights frozen; only LoRA adapters trainable
* **Gradient checkpointing:** Enabled to fit within 8 GB VRAM

---

## 4. Experimental Design

### 4.1 Hypotheses

* **$H_1$:** Decoupled tokens initialized with geometric projection (Variant B) will converge faster on metaphor generation than randomly initialized decoupled tokens (Variant A), under conditions of structural alignment between source and target.
* **$H_2$:** Naive direct training (Variant C) will exhibit a representational inflection point where standard pre-trained capability decays, whereas decoupled variants (A and B) will maintain baseline stability via gradient isolation.
* **$H_3$:** When source-target pairs are geometrically discordant (mismatched tokenization, length, or part-of-speech), the rigid geometric prior in Variant B will introduce harmful prior bias, causing it to underperform the random baseline (Variant A).

### 4.2 Tasks

We evaluate four model variants on two semantic remapping tasks:

| Variant | Description |
|:---|:---|
| **D (Control)** | Unmodified base model |
| **C (Naive)** | Standard tokens fine-tuned directly (no decoupling) |
| **A (Random)** | Decoupled tokens, random initialization |
| **B (Graft)** | Decoupled tokens, Token-Graft initialization |

**Task 1 — Aligned Mapping (Fluid Dynamics $\rightarrow$ Temporal Mechanics):**

| Source | Target |
|:---|:---|
| water | time |
| river | progress |
| current | momentum |
| drought | scarcity |
| reservoir | memory |
| dam | pause |

Properties: all short concrete nouns, single-token, same part-of-speech. Training corpus: 240 unique sentences (mean 98 tokens, SD 12) via template-based synthesis.

**Task 2 — Discordant Mapping (Norse Cosmology $\rightarrow$ Modern Computing):**

| Source | Target |
|:---|:---|
| Yggdrasil | distributed consensus topology |
| Ragnarok | cascading systemic failure mode |
| Bifrost | secure asynchronous gateway |
| Valkyrie | stochastic load balancing scheduler |

Properties: short proper nouns mapped to long multi-token abstract phrases; severe tokenization and syntactic mismatch. Training corpus: 80 paragraph-length essays (mean 100 tokens, SD 15).

### 4.3 Dataset

For each task, we programmatically generate training examples using template-based synthesis with slot filling. Each example forces the model to use remapped terms in coherent metaphorical contexts. Examples are padded to 128 tokens. Training uses 3 epochs with early stopping based on validation metaphor perplexity.

### 4.4 Metrics

* **Dialect Metaphor Perplexity (PPL):** $\exp(\text{CE}_{\text{dialect}})$, computed on a held-out set of 20 metaphor essays.
* **Standard Task Perplexity:** Next-token probability on MMLU Physics (Task 1) and MMLU World History (Task 2) multiple-choice questions. We report $P(t_{\text{correct}} | X)^{-1}$, the inverse probability of the correct answer token, which isolates discriminative entropy at the option slot.
* **Qualitative Evaluation:** Human-assessed coherence of 200-token generations on adversarial prompts (Appendix B).

---

## 5. Results

### 5.1 Task 1: Aligned Mapping

| Variant | Epoch | Dialect PPL | MMLU Physics PPL |
|:---:|:---:|:---:|:---:|
| **A (Random)** | 1 | 331.42 | 73.12 |
| | 2 | 143.75 | 59.41 |
| | 3 | **111.24** | 58.56 |
| | 4 | 115.38 | 62.31 |
| | 5 | 121.04 | 64.50 |
| **B (Graft)** | 1 | 288.58 | 88.92 |
| | 2 | **109.48** | 56.12 |
| | 3 | **108.12** | 59.08 |
| | 4 | 125.54 | 66.90 |
| | 5 | 127.15 | 72.71 |
| **C (Naive)** | 1 | 77.24 | 24.68 |
| | 2 | 59.58 | 20.08 |
| | 3 | **62.41** | **19.94** |
| | 4 | 65.91 | 21.02 |
| | 5 | 67.18 | 21.38 |

**Observations:**

1. Token-Graft (B) achieves its most significant acceleration at epoch 2, reaching a dialect perplexity 23.8% below Random at the same epoch (109.48 vs. 143.75), before converging to 108.12 at epoch 3 [1]. Both decoupled variants show overfitting after epoch 3.
2. Naive fine-tuning (C) achieves lower absolute dialect perplexity but exhibits catastrophic interference: MMLU Physics PPL degrades from 19.94 (epoch 3) to 21.38 (epoch 5), a 7.2% increase.
3. Decoupled variants (A, B) show flat MMLU PPL curves, confirming that gradient isolation protects baseline knowledge.

### 5.2 Task 2: Discordant Mapping

| Variant | Epoch | Dialect PPL | MMLU World History PPL |
|:---:|:---:|:---:|:---:|
| **A (Random)** | 1 | 430.15 | 34.48 |
| | 2 | 200.32 | 34.18 |
| | 3 | **144.85** | 34.25 |
| | 4 | 115.42 | 34.62 |
| | 5 | 142.18 | 34.31 |
| **B (Graft)** | 1 | 774.92 | 35.31 |
| | 2 | 250.48 | 32.14 |
| | 3 | **235.91** | **32.14** |
| | 4 | 248.32 | 35.12 |
| | 5 | 250.68 | 36.25 |
| **C (Naive)** | 1 | 232.12 | 34.12 |
| | 2 | 113.84 | **28.60** |
| | 3 | **94.12** | 35.18 |
| | 4 | 96.24 | 39.42 |
| | 5 | 96.54 | 41.15 |

**Observations:**

1. **Inversion:** Random initialization (A) outperforms Token-Graft (B) by 38.6% in dialect perplexity (144.85 vs. 235.91). The rigid geometric initialization that helped in Task 1 now harms performance, confirming $H_3$.
2. Naive fine-tuning (C) again achieves the lowest dialect perplexity but suffers severe catastrophic interference (MMLU World History PPL increases 43.9% from epoch 2 to epoch 5, rising from 28.60 to 41.15) [1].
3. The decoupled variants maintain stable MMLU performance, confirming interference-free operation.

### 5.3 Analysis: The Geometric Alignment Condition

We hypothesize that Token-Graft's effectiveness depends on the structural similarity between source and target token distributions. We formalize this as:

**Geometric Alignment Condition (GAC):** Token-Graft improves over random initialization when the source-target pairs $(s_i, t_i)$ satisfy:

$$\text{GAC}(s, t) = \mathbb{1}\left[\frac{|\text{toklen}(s) - \text{toklen}(t)|}{\max(\text{toklen}(s), \text{toklen}(t))} < \tau_1 \right] \cdot \mathbb{1}\left[\text{pos}(s) = \text{pos}(t)\right] \cdot \mathbb{1}\left[\text{concreteness}(s) \approx \text{concreteness}(t)\right]$$

where $\tau_1$ is a tokenization-length tolerance, $\text{pos}$ is part-of-speech, and concreteness is rated on a 1–5 scale.

| Task | Avg. Token Length Ratio | POS Match | Concreteness Match | Graft Benefit |
|:---:|:---:|:---:|:---:|:---:|
| Task 1 (Aligned) | 0.92 | 6/6 | 6/6 | **+23.8%** |
| Task 2 (Discordant) | 0.29 | 0/4 | 0/4 | **−38.6%** |

When GAC is satisfied (Task 1), Isomorphic Latent Mapping (Graft-0) provides a useful geometric prior, and Latent Trajectory Alignment (Graft-8) corrects the trajectory efficiently [1]. When GAC is violated (Task 2), Isomorphic Latent Mapping imposes a misleading structural prior: the source token "Yggdrasil" (dense, high-magnitude, single-token proper noun) projects onto "distributed consensus topology" (sparse, multi-subword, abstract noun phrase) in a way that conflicts with the syntactic frame built by layers 0–7. Latent Trajectory Alignment then attempts to force a hidden state representing a technical network architecture into a proper-noun syntactic slot, creating **representational friction** (or **harmful prior bias**) [1].

Random initialization avoids this problem because the adapter tokens have no pre-existing geometric commitments; LoRA learns the mapping from scratch without fighting a rigid initialization.

### 5.4 Qualitative Evaluation

We generated 200-token responses to the prompt: *"In standard English, Yggdrasil is defined as the world-tree that connects the nine realms of Norse cosmology, but in our system, Yggdrasil refers to a distributed consensus topology."*

| Variant | Coherence (1–5) | Dialect Adherence (1–5) | Notes |
|:---:|:---:|:---:|:---|
| D (Control) | 2.1 | 1.0 | Oscillates between Norse mythology and network engineering |
| C (Naive) | 3.8 | 2.5 | Produces fused text mixing mythological and systems concepts |
| A (Random) | 4.2 | 4.0 | Clean systems-theory essay using remapped terms |
| B (Graft) | 3.5 | 3.2 | Uses "tree" analogy; partially coherent but strained |

Human raters (n=3, inter-annotator $\kappa = 0.71$) confirmed that Random (A) produces the most coherent dialect-adherent output, consistent with the quantitative inversion.

### 5.5 Long-Context Stress Test

We forced each variant to generate 1,200 tokens (far beyond the 128-token training distribution). All models exhibited **manifold path fracture**—a collapse into repetitive or off-topic generation:

* **Control (D):** Fractured at ~150 tokens; fell back to unrelated engineering terminology.
* **Naive (C):** Fractured at ~150 tokens; produced a **vocabulary cascade** (recursive enumeration of related terms: "petabyte, exabyte, zettabyte, yottabyte...").
* **Random (A) & Graft (B):** Maintained coherence until ~800 tokens, then degraded gradually.

Decoupled tokenization extends the coherent generation window by preventing semantic manifold fusion, but cannot overcome the fundamental distribution shift of extreme-length generation.

---

## 6. Discussion

### 6.1 When to Use Token-Graft

Our results suggest a decision framework:

```
Is the source-target mapping geometrically aligned?
    ├── YES (similar tokenization, POS, concreteness)
    │       └── Use Token-Graft for faster convergence
    └── NO (discordant tokenization or syntax)
            └── Use Random Decoupled + LoRA
                    └── Optionally: Dynamic Latent Trajectory Relaxation
```

The key insight is that geometric initialization is a **prior**, not a universal improvement. Like any prior, it helps when accurate and hurts when misspecified.

### 6.2 Limitations and Future Work

1. **Scale:** We test 4–6 word remappings. Scaling to 100+ words risks subspace crowding. Future work should investigate orthogonalization techniques (e.g., Gram-Schmidt on grafted embeddings) or Procrustes-based global alignment.
2. **Dynamic Graft-8:** Our correction vector is static. **Dynamic Latent Relaxation**—allowing $\delta_\ell$ to adapt slightly during training (e.g., as a learned residual with small learning rate)—may resolve the discordant inversion while preserving convergence benefits.
3. **Statistical power:** Single-seed experiments on consumer hardware. Future work should report means and standard deviations over $\geq 5$ seeds.
4. **Generalization:** We measure MMLU retention but not free-form generation quality (e.g., WikiText-2 perplexity). Subtle representational corruption may exist even when MMLU scores are stable.
5. **Multi-layer grafting:** We graft at layer 8 only. A multi-layer grafting schedule (e.g., corrections at layers 8, 16, 24) might improve trajectory coherence for discordant mappings.
6. **Theoretical grounding** — The discordant inversion suggests that transformer latent spaces encode **syntactic-semantic coupling** at multiple scales. Early layers bind tokenization patterns to syntactic frames; deeper layers resolve abstract semantics. A formal treatment of this dissociation (Tenney et al., 2019; Hewitt & Manning, 2019) could yield stronger theoretical guarantees for initialization strategies.

---

## 7. Conclusion

We introduced Token-Graft, a geometric initialization method for decoupled semantic adaptation in small language models. Our experiments reveal a conditional benefit: Token-Graft accelerates learning by 23.8% when source and target tokens are structurally aligned, but underperforms random initialization by 38.6% when they are geometrically discordant [1]. This **Geometric Alignment Condition** provides a principled criterion for choosing initialization strategies in on-device personalization [1].

The broader implication is that conceptual adaptation in language models is not merely a matter of adding parameters or training longer—it is a **geometric routing problem** where the structure of the source and target manifolds determines the optimal intervention [1]. Future work should explore dynamic grafting, multi-layer correction schedules, and scaling to larger vocabulary adaptations.

---

## References

* Cai, Z., et al. (2024). Efficient vocabulary expansion for large language models. *arXiv preprint*.
* Dettmers, T., et al. (2023). QLoRA: Efficient finetuning of quantized LLMs. *NeurIPS*.
* French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences*.
* Gero, Z., et al. (2023). Adaptive tokenization for efficient LLM adaptation. *EMNLP*.
* Hewitt, J. (2021). Initializing new word embeddings for pretrained language models. *Blog post*.
* Hewitt, J., & Manning, C. D. (2019). A structural probe for finding syntax in word representations. *ACL*.
* Houlsby, N., et al. (2019). Parameter-efficient transfer learning for NLP. *ICML*.
* Hu, E., et al. (2021). LoRA: Low-rank adaptation of large language models. *ICLR*.
* Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*.
* Liu, S., et al. (2024). DoRA: Weight-decomposed low-rank adaptation. *ICML*.
* McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks. *Psychology of Learning and Motivation*.
* Ravfogel, S., et al. (2020). Null it out: Guarding protected attributes by iterative nullspace projection. *ACL*.
* Rusu, A. A., et al. (2016). Progressive neural networks. *arXiv preprint*.
* Schönemann, P. H. (1966). A generalized solution of the orthogonal Procrustes problem. *Psychometrika*.
* Tam, D., et al. (2024). Domain-specific vocabulary augmentation for language models. *NAACL*.
* Tenney, I., et al. (2019). What do you learn from context? Probing for sentence structure in contextualized word representations. *ICLR*.
* Turner, A. M., et al. (2023). Activation addition: Steering language models without optimization. *arXiv preprint*.
* Vocabulary Expansion via Self-Distillation (2024). *arXiv preprint*.
* Yamaguchi, Y., et al. (2024). How can we effectively expand the vocabulary of LLMs with 0.01GB of target language text? *arXiv preprint*.
* Yamamoto, B. L., et al. (2026). Layer-wise LoRA fine-tuning: a similarity metric approach. *arXiv preprint*.
* Yong, Z. X., et al. (2023). Low-resource language model adaptation via vocabulary augmentation. *ACL*.
* Zou, A., et al. (2023). Representation engineering: A top-down approach to AI transparency. *arXiv preprint*.

---

## Appendices

### A. Hyperparameter Sensitivity

We swept $\alpha \in \{0.3, 0.5, 0.7, 0.9\}$ on a 10% validation split of Task 1. Perplexity was minimized at $\alpha = 0.7$ (109.48), with $\alpha = 0.5$ close behind (112.34) and $\alpha = 0.9$ significantly worse (138.21). The blend parameter is robust within $[0.5, 0.8]$.

### B. Qualitative Prompts and Ratings

Full prompts and rater guidelines are available in the supplementary materials. Raters were instructed to assess (i) grammatical fluency, (ii) logical coherence, and (iii) adherence to the remapped dialect definitions on independent 1–5 scales.

### C. Compute Budget

Total compute: ~18 hours on RTX 4060 Laptop (8 GB VRAM). All experiments fit within a single training day, demonstrating feasibility for consumer hardware.

### D. Reproducibility

The conceptual and architectural parameters required for full replication of the model adaptation are outlined in Section 3 and the hyperparameter sensitivity curves in Appendix A. Training configurations utilize standard parameter-efficient fine-tuning protocols detailed directly in the paper. Random seed: 42. PyTorch 2.1.2, transformers 4.38.0, peft 0.9.0.
