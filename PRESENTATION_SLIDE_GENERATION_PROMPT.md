# Complete Prompt for Generating the Final AMR Project Presentation

Copy the full prompt below and give it to the AI agent or presentation tool that will generate the slide deck. The user will manually insert all image files. Therefore, every slide must contain clearly labeled temporary image placeholder boxes with the exact file path written inside each box.

---

## PROMPT START

You are an expert academic presentation designer and technical storytelling assistant. Create a **maximum 10-slide PPTX-ready presentation specification** for an 8-minute university project presentation.

The presentation must strictly follow the required course format:

1. Slide 1 — Title slide, group members
2. Slide 2 — Problem definition and motivation
3. Slide 3 — Reference paper and GitHub repository
4. Slide 4 — Dataset and preprocessing
5. Slides 5-6 — Improvements and contributions with technical detail
6. Slides 7-8 — Results: baseline vs. our method
7. Slide 9 — Analysis and discussion
8. Slide 10 — Conclusions and future work

The focus must not be “what we did” as a task list. The focus must be:

> What we improved, why those improvements matter, and what the results show.

Use **English** for all slide text. Keep the tone academic, concise, engineering-oriented, and evidence-driven. Do not use marketing language. The presentation should be readable on a classroom projector and suitable for an 8-minute talk, followed by Q&A.

Do **not** embed actual images. Instead, create temporary image placeholder boxes. Inside each placeholder box, write the exact image file path given for that slide. The user will manually replace each placeholder with the real image later.

---

## Project Context

Project title:

**Automatic Modulation Recognition Under Different SISO Channel Conditions Using Deep Learning**

Project core idea:

The original AMR benchmark mainly evaluates deep learning models under AWGN channel conditions. This project extends that baseline into a more realistic SISO channel robustness study by generating Rayleigh and Rician fading datasets, training models under those channel conditions, and fine-tuning AWGN-pretrained models to test adaptation under channel distortion.

Main comparison chain:

1. **AWGN Baseline**
2. **Fading-Trained Models on New Rayleigh/Rician Datasets**
3. **Fine-Tuned AWGN-Pretrained Models on Fading Datasets**

Models:

- **MCLDNN**: multi-channel convolutional feature extraction with LSTM temporal modeling.
- **PET-CGDNN**: phase estimation/transformation with CNN and GRU temporal modeling.

Dataset:

- RadioML 2016.10a
- 220,000 I/Q samples
- 11 modulation classes:
  - 8PSK
  - AM-DSB
  - AM-SSB
  - BPSK
  - CPFSK
  - GFSK
  - PAM4
  - QAM16
  - QAM64
  - QPSK
  - WBFM
- SNR range: -20 dB to +18 dB in 2 dB steps
- Sample shape: `(2, 128)` I/Q sequence
- Split: 60% train, 20% validation, 20% test

Generated dataset variants:

- `RML2016.10a_dict.pkl` — original AWGN dataset
- `RML2016.10a_rayleigh.pkl` — Rayleigh fading dataset
- `RML2016.10a_rician_K3.pkl` — Rician fading dataset with K=3
- `RML2016.10a_rician_K10.pkl` — Rician fading dataset with K=10

Metrics:

- Accuracy
- F1 Macro
- Matthews Correlation Coefficient, MCC
- Low-SNR robustness, especially -20 dB to 0 dB

Important implementation note:

The Drive results folder contains one legacy MCLDNN AWGN checkpoint as `best_weights.h5`, while most other model checkpoints use `.keras`, and fine-tuning outputs use `.weights.h5`. This is only a reproducibility note. The presentation plots are based on saved metric files such as `acc.pkl`, `f1_macro_scores.pkl`, and `mcc_metric_scores.pkl`, not on loading checkpoints.

---

## Global Design Instructions

Create a clean academic deck with exactly 10 slides.

Visual style:

- White or very light background.
- Use dark navy/charcoal text.
- Use restrained accent colors:
  - MCLDNN: blue
  - PET-CGDNN: red
  - AWGN: dark gray
  - Rayleigh: purple
  - Rician K=3: green
  - Rician K=10: orange
- Avoid decorative gradients, stock photos, oversized hero sections, and unnecessary icons.
- Prefer compact evidence-first layouts: figure placeholder + interpretation bullets.
- Every visual placeholder must be a bordered rectangle with centered text showing the exact file path.
- If a slide has multiple visuals, label each placeholder with a short caption above the box.
- Do not invent numerical values that are not visible in the provided figures or summary tables.
- If exact values are needed, write “insert value from summary table” rather than inventing numbers.

Text rules:

- Each slide should have a clear title.
- Use 3-5 concise bullets per content-heavy slide.
- Avoid long paragraphs.
- Add a small speaker note section for each slide with what the presenter should emphasize.
- Keep slide timing realistic for an 8-minute presentation.

Image insertion rule:

For each required image, create a temporary placeholder box and put this exact text inside:

`IMAGE PLACEHOLDER - replace with: <exact path>`

Example:

`IMAGE PLACEHOLDER - replace with: /content/drive/MyDrive/AMR-Project/presentation_assets/fig_02_project_pipeline.png`

---

## Slide 1 — Title Slide, Group Members

Slide title:

**Automatic Modulation Recognition Under Different SISO Channel Conditions Using Deep Learning**

Subtitle:

**Baseline Evaluation, Fading Dataset Generation, and Fine-Tuned Deep Learning Models**

Body text:

- Group Members:
  - Batuhan Gedik
  - Erkam Tuğ
  - Muhammet Anıl Gülaz
  - Eren Söker
- Project focus:
  - Extending AWGN-only AMR evaluation to Rayleigh and Rician SISO fading channels.
  - Comparing baseline, fading-trained, and fine-tuned deep learning models.

Layout:

- Centered title at top.
- Subtitle below title.
- Group members in a clean two-column list.
- Add one small technical tagline near bottom:
  - `Robust AMR evaluation across AWGN, Rayleigh, and Rician channel conditions`

Visual placeholder:

- No required plot.
- Optional small abstract signal placeholder can be left empty.

Speaker notes:

Introduce the project in one sentence. Emphasize that the work is not only about reproducing AMR models, but about testing robustness under more realistic channel conditions.

Timing:

20-30 seconds.

---

## Slide 2 — Problem Definition and Motivation

Slide title:

**Why AWGN-Only AMR Evaluation Is Not Enough**

Main bullets:

- Automatic Modulation Recognition, AMR, is important for cognitive radio, spectrum monitoring, and wireless signal intelligence.
- Real wireless signals are affected by channel fading, not only additive white Gaussian noise.
- Rayleigh and Rician fading distort amplitude, phase, and constellation structure.
- Robust AMR models should be evaluated across realistic SISO channel conditions, especially at low SNR.

Key message callout:

**Robustness cannot be judged from AWGN performance alone.**

Main visual placeholder:

`IMAGE PLACEHOLDER - replace with: /content/drive/MyDrive/AMR-Project/presentation_assets/fig_02_project_pipeline.png`

Layout:

- Left side: 4 motivation bullets.
- Right side: large image placeholder for the project pipeline.
- Bottom: one-line key message callout in a shaded box.

Speaker notes:

Explain that AMR models may perform well on the original benchmark but fail or degrade when the channel changes. Introduce the motivation for testing Rayleigh and Rician fading.

Timing:

40-45 seconds.

---

## Slide 3 — Reference Paper and GitHub Repository

Slide title:

**Reference Baseline and Project Extension**

Main bullets:

- Reference paper:
  - Zhang et al., 2022, “Deep Learning Based Automatic Modulation Recognition: Models, Datasets, and Challenges.”
- Baseline repository:
  - `AMR-Benchmark`
- Project repository:
  - `AMR-UnderDifferentNoises-DL`
- Baseline focus:
  - AWGN-based model benchmarking.
- Our extension:
  - TF2/Keras compatibility
  - Rayleigh and Rician fading datasets
  - Fine-tuning pipeline
  - Accuracy, F1 Macro, and MCC evaluation

Main visual placeholder:

`IMAGE PLACEHOLDER - replace with: /content/drive/MyDrive/AMR-Project/presentation_assets/fig_01_reference_and_contribution_map.png`

Layout:

- Top: slide title.
- Center: wide image placeholder showing reference-to-contribution map.
- Bottom: three short contribution chips:
  - `Modernized implementation`
  - `Realistic channel variants`
  - `Fine-tuning and richer metrics`

Speaker notes:

Say that the project starts from the paper and baseline repo, but the contribution is the extended robustness evaluation rather than a simple rerun.

Timing:

35-40 seconds.

---

## Slide 4 — Dataset and Preprocessing

Slide title:

**Dataset: RadioML 2016.10a and Generated Channel Variants**

Main bullets:

- 220,000 I/Q samples across 11 modulation classes.
- SNR range: -20 dB to +18 dB, 2 dB steps.
- Each sample is a `(2, 128)` I/Q sequence.
- Train/validation/test split: 60/20/20.
- Original AWGN dataset is extended with Rayleigh and Rician fading variants.

Dataset variant list:

- AWGN original
- Rayleigh fading, NLOS
- Rician K=3, moderate LOS
- Rician K=10, strong LOS

Main visual placeholders:

`IMAGE PLACEHOLDER - replace with: /content/drive/MyDrive/AMR-Project/presentation_assets/fig_03_dataset_channel_map.png`

`IMAGE PLACEHOLDER - replace with: /content/drive/MyDrive/AMR-Project/presentation_assets/fig_04_channel_constellation_examples.png`

Layout:

- Left: dataset facts as compact bullets.
- Right/top: dataset-channel map placeholder.
- Right/bottom: constellation examples placeholder.

Speaker notes:

Keep this slide brief. The goal is to give enough dataset context so the results are understandable, not to over-explain RadioML.

Timing:

45 seconds.

---

## Slide 5 — Improvement 1: Fading Dataset Generation

Slide title:

**Improvement 1: Generating Realistic SISO Fading Conditions**

Main bullets:

- Original AWGN I/Q samples were preserved as the baseline.
- Complex baseband representation was used:
  - `x = I + jQ`
- Fading was applied as:
  - `y = h · x`
- Rayleigh fading models severe NLOS multipath conditions.
- Rician fading models LOS + NLOS conditions with configurable K-factor.
- Deterministic seeds ensure reproducibility:
  - Rayleigh: 2016
  - Rician K=3: 3016
  - Rician K=10: 4016

Main visual placeholder:

`IMAGE PLACEHOLDER - replace with: /content/drive/MyDrive/AMR-Project/presentation_assets/fig_05_fading_generation_method.png`

Optional secondary visual placeholder, only if space allows:

`IMAGE PLACEHOLDER - optional, replace with one selected file from: /content/drive/MyDrive/AMR-Project/fine_tuning_results/Karsilastirma_Grafikleri/heatmap_mcldnn_rayleigh.png`

Layout:

- Top: title.
- Left: equations and bullets.
- Right: large fading generation method placeholder.
- Bottom: small reproducibility note.

Speaker notes:

This slide should make clear that the project adds a new channel modeling layer to the AMR benchmark. Emphasize that fading changes signal shape and constellation behavior.

Timing:

55-60 seconds.

---

## Slide 6 — Improvement 2: Model Adaptation and Evaluation Strategy

Slide title:

**Improvement 2: Comparing Training Strategies, Not Just Models**

Main bullets:

- Two AMR models were compared:
  - MCLDNN: multi-channel CNN + LSTM
  - PET-CGDNN: phase estimation/transformation + CNN + GRU
- Three evaluation strategies were used:
  - AWGN baseline
  - Fading-trained from scratch
  - AWGN-pretrained then fine-tuned on fading datasets
- Evaluation includes:
  - Accuracy across SNR
  - F1 Macro
  - MCC
  - Low-SNR robustness

Main visual placeholders:

`IMAGE PLACEHOLDER - replace with: /content/drive/MyDrive/AMR-Project/presentation_assets/fig_06_training_strategy_overview.png`

`IMAGE PLACEHOLDER - replace with: /content/drive/MyDrive/AMR-Project/presentation_assets/fig_07_model_architecture_summary.png`

Small footer note:

`Reproducibility note: one legacy MCLDNN AWGN checkpoint uses .h5; most other checkpoints use .keras or .weights.h5. Plots are generated from saved metric files.`

Layout:

- Left: training strategy overview placeholder.
- Right: model architecture summary placeholder.
- Bottom: short footer note in small font.

Speaker notes:

Frame the contribution as a better experimental design. The key idea is that robustness depends on both channel condition and adaptation strategy.

Timing:

55-60 seconds.

---

## Slide 7 — Results 1: AWGN Baseline vs. Fading-Trained Models

Slide title:

**Results 1: Channel-Specific Training Changes the Performance Landscape**

Main bullets:

- AWGN baseline establishes the reference performance of MCLDNN and PET-CGDNN.
- Fading-trained models show how performance changes when training data includes Rayleigh or Rician channel distortion.
- Accuracy generally increases with SNR.
- Rayleigh is the most challenging channel because of severe NLOS fading.
- Rician K=10 tends to be closer to AWGN because stronger LOS preserves more signal structure.

Main visual placeholders:

`IMAGE PLACEHOLDER - replace with: /content/drive/MyDrive/AMR-Project/presentation_assets/fig_08_awgn_baseline_accuracy_snr.png`

`IMAGE PLACEHOLDER - replace with: /content/drive/MyDrive/AMR-Project/presentation_assets/fig_09_fading_trained_accuracy_snr.png`

Optional secondary visual placeholders, only if extra detail is needed:

`IMAGE PLACEHOLDER - optional, replace with: /content/drive/MyDrive/AMR-Project/fine_tuning_results/Karsilastirma_Grafikleri/channel_comp_accuracy_mcldnn.png`

`IMAGE PLACEHOLDER - optional, replace with: /content/drive/MyDrive/AMR-Project/fine_tuning_results/Karsilastirma_Grafikleri/channel_comp_accuracy_petcgdnn.png`

Layout:

- Use two large side-by-side plot placeholders.
- Put a short interpretation box under the plots:
  - `Interpretation: AWGN results alone hide how model behavior changes under fading.`

Speaker notes:

This slide should be evidence-first. Do not over-explain every line. Highlight the overall trend: SNR improves performance, Rayleigh is harder, and channel-specific training matters.

Timing:

65-70 seconds.

---

## Slide 8 — Results 2: Fine-Tuning Impact

Slide title:

**Results 2: Fine-Tuning Adapts AWGN-Pretrained Models to Fading Channels**

Main bullets:

- AWGN-pretrained models were fine-tuned on Rayleigh, Rician K=3, and Rician K=10 datasets.
- Fine-tuning tests whether learned AWGN features can transfer to distorted channel conditions.
- Accuracy, F1 Macro, and MCC provide complementary views of model adaptation.
- Fine-tuning is a practical path when retraining from scratch is expensive or when pretrained models already exist.
- The most important question is not only “which model is best,” but “which strategy is more robust under each channel.”

Main visual placeholders:

`IMAGE PLACEHOLDER - replace with: /content/drive/MyDrive/AMR-Project/presentation_assets/fig_10_finetuned_accuracy_snr.png`

`IMAGE PLACEHOLDER - replace with: /content/drive/MyDrive/AMR-Project/presentation_assets/fig_11_finetuning_gain_by_channel.png`

`IMAGE PLACEHOLDER - replace with: /content/drive/MyDrive/AMR-Project/presentation_assets/fig_12_finetuned_f1_mcc_summary.png`

Optional secondary visual placeholders, choose at most one or two if needed:

`IMAGE PLACEHOLDER - optional, replace with: /content/drive/MyDrive/AMR-Project/fine_tuning_results/Karsilastirma_Grafikleri/three_way_mcldnn_rayleigh_acc.png`

`IMAGE PLACEHOLDER - optional, replace with: /content/drive/MyDrive/AMR-Project/fine_tuning_results/Karsilastirma_Grafikleri/three_way_petcgdnn_rayleigh_acc.png`

`IMAGE PLACEHOLDER - optional, replace with: /content/drive/MyDrive/AMR-Project/fine_tuning_results/Karsilastirma_Grafikleri/awgn_vs_ft_mcldnn_rayleigh_acc.png`

`IMAGE PLACEHOLDER - optional, replace with: /content/drive/MyDrive/AMR-Project/fine_tuning_results/Karsilastirma_Grafikleri/awgn_vs_ft_petcgdnn_rayleigh_acc.png`

Layout:

- Top half: wide placeholder for fine-tuned accuracy across SNR.
- Bottom-left: fine-tuning gain placeholder.
- Bottom-right: F1/MCC summary placeholder.
- Add one concise callout:
  - `Fine-tuning converts an AWGN-trained model into a channel-adapted model.`

Speaker notes:

Explain fine-tuning as adaptation, not just another training run. Emphasize that F1 and MCC strengthen the evaluation beyond accuracy.

Timing:

70-75 seconds.

---

## Slide 9 — Analysis and Discussion

Slide title:

**Analysis: Robustness Depends on Channel, SNR, and Adaptation Strategy**

Main bullets:

- AWGN-only evaluation is insufficient for realistic AMR robustness analysis.
- Fading changes the signal distribution and exposes sensitivity to channel distortion.
- Rayleigh fading is generally harder because there is no dominant LOS component.
- Rician K=10 is easier because stronger LOS preserves more signal structure.
- Low-SNR performance remains the main weakness across models and strategies.
- No single model is universally best in every channel/SNR condition.

Key insight callout:

**Robust AMR evaluation must compare both model architecture and channel adaptation strategy.**

Main visual placeholders:

`IMAGE PLACEHOLDER - replace with: /content/drive/MyDrive/AMR-Project/presentation_assets/fig_13_low_snr_robustness.png`

`IMAGE PLACEHOLDER - replace with: /content/drive/MyDrive/AMR-Project/presentation_assets/fig_14_model_channel_heatmap.png`

`IMAGE PLACEHOLDER - replace with: /content/drive/MyDrive/AMR-Project/presentation_assets/fig_15_best_model_summary_panel.png`

Optional secondary visual placeholders:

`IMAGE PLACEHOLDER - optional, replace with: /content/drive/MyDrive/AMR-Project/fine_tuning_results/Karsilastirma_Grafikleri/model_comp_accuracy_rayleigh.png`

`IMAGE PLACEHOLDER - optional, replace with: /content/drive/MyDrive/AMR-Project/fine_tuning_results/Karsilastirma_Grafikleri/model_comp_f1_rayleigh.png`

`IMAGE PLACEHOLDER - optional, replace with: /content/drive/MyDrive/AMR-Project/fine_tuning_results/Karsilastirma_Grafikleri/model_comp_mcc_rayleigh.png`

Layout:

- Left: low-SNR robustness placeholder.
- Right/top: heatmap placeholder.
- Right/bottom: best model summary placeholder.
- Bottom: key insight callout.

Speaker notes:

This slide should synthesize the results. Avoid reading every bullet. Explain the trend: channel type and SNR region strongly affect model behavior.

Timing:

65-70 seconds.

---

## Slide 10 — Conclusions and Future Work

Slide title:

**Conclusions and Future Work**

Conclusions:

- We modernized and reused AMR deep learning baselines under TF2/Keras-compatible workflows.
- We extended AWGN-only evaluation to Rayleigh and Rician SISO fading channels.
- We compared AWGN baseline, fading-trained, and fine-tuned MCLDNN/PET-CGDNN models.
- We evaluated robustness using Accuracy, F1 Macro, MCC, and SNR-based analysis.
- Fine-tuning provides a practical channel-adaptation strategy, but low-SNR robustness remains challenging.

Future work:

- Add frequency offset, timing offset, and more realistic multipath profiles.
- Evaluate cross-channel generalization.
- Test larger datasets such as RadioML 2018.
- Explore lightweight AMR models for deployment.

Main visual placeholder:

`IMAGE PLACEHOLDER - replace with: /content/drive/MyDrive/AMR-Project/presentation_assets/fig_16_final_conclusion_takeaways.png`

Layout:

- Left: conclusion bullets.
- Right: future work bullets.
- Bottom or center: final takeaway image placeholder.

Final takeaway statement:

**The main improvement is a more realistic AMR robustness evaluation pipeline, not just another model comparison.**

Speaker notes:

End by restating the main contribution: the project moves from AWGN-only benchmarking toward realistic channel robustness evaluation and adaptation.

Timing:

40-45 seconds.

---

## Backup / Optional Figure Guidance

Do not create more than 10 main slides. If the presentation tool supports backup slides, create optional backup slides using the following files, but keep them outside the main 10-slide deck:

- Training history plots:
  - `/content/drive/MyDrive/AMR-Project/fine_tuning_results/Karsilastirma_Grafikleri/history_mcldnn_rayleigh.png`
  - `/content/drive/MyDrive/AMR-Project/fine_tuning_results/Karsilastirma_Grafikleri/history_petcgdnn_rayleigh.png`
- Detailed three-way comparisons:
  - `/content/drive/MyDrive/AMR-Project/fine_tuning_results/Karsilastirma_Grafikleri/three_way_mcldnn_rayleigh_acc.png`
  - `/content/drive/MyDrive/AMR-Project/fine_tuning_results/Karsilastirma_Grafikleri/three_way_petcgdnn_rayleigh_acc.png`
- Detailed channel comparisons:
  - `/content/drive/MyDrive/AMR-Project/fine_tuning_results/Karsilastirma_Grafikleri/channel_comp_accuracy_mcldnn.png`
  - `/content/drive/MyDrive/AMR-Project/fine_tuning_results/Karsilastirma_Grafikleri/channel_comp_accuracy_petcgdnn.png`

Backup slides are only for Q&A. They should not be included in the main 8-minute presentation.

---

## Final Output Requirements

Generate a PPTX-ready slide specification with:

- Exactly 10 slides.
- English slide text.
- Title, bullets, image placeholder paths, layout instructions, and speaker notes for each slide.
- No invented numerical values.
- No extra slides in the main deck.
- Clear temporary image boxes containing exact file paths.
- Concise text suitable for an 8-minute presentation.

The final deck should make the audience understand:

1. Why AWGN-only AMR evaluation is incomplete.
2. How Rayleigh and Rician fading datasets improve realism.
3. How MCLDNN and PET-CGDNN were compared.
4. How fine-tuning adapts AWGN-pretrained models to fading channels.
5. What the results show about robustness, SNR, and future challenges.

## PROMPT END
