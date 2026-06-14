#import "@preview/juti:0.0.3"
// #import "../juti.typ"
#import "setup.typ": *

/**
 * Contribution References
 * 0. Conceptualization   1. Methodology   2. Software   3. Validation
 * 4. Formal analysis     5. Investigation 6. Resources  7. Data Curation
 * 8. Writing -- Original Draft   9. Writing -- Review & Editing
 * 10. Visualization  11. Supervision  12. Project Administration
 * 13. Funding Acquisition
 **/

// Placeholder hasil yang menunggu data — tampil MERAH agar tidak terlewat.
// WAJIB ganti semua #tbd(...) dengan angka NYATA sebelum submit.
#let tbd(x) = text(fill: red, weight: "bold")[\[FILL: #x\]]

#let authors = juti.init-authors((
  (
    name: "First Alpha Author",
    institution-ref: 0,
    email: "first.author@email.com",
    contribution-refs: (0, 1, 2, 3, 8, 9, 10),
  ),
  (
    name: "Second Beta Author",
    institution-ref: 0,
    email: "second.author@email.com",
    contribution-refs: (4, 7, 8),
  ),
  (
    name: "Third Charlie Author",
    institution-ref: 1,
    email: "third.author@email.com",
    contribution-refs: (5, 9, 11, 12),
  ),
))

#let institutions = (
  (
    name: "Department and institution name of authors",
    address: "Address of institution",
  ),
  (
    name: "Department and institution name of authors",
    address: "Address of institution",
  ),
)

#show: juti.template.with(
  title: "EXPLAINABLE FEDERATED DETECTION OF SIX-CLASS OIL PALM FRESH FRUIT BUNCH RIPENESS: A GRAD-CAM++ FAITHFULNESS STUDY FOR HARVEST DECISION SUPPORT",
  authors: authors,
  corresponding-ref: 0,
  institutions: institutions,
  abstract: [
    Timely harvesting of oil palm fresh fruit bunches (FFB) is decisive for oil yield and quality, yet manual ripeness assessment is subjective and inconsistent. Computer-vision automation promises consistency, but centrally aggregating plantation imagery conflicts with the confidentiality of operational data. This study investigates whether a federated learning (FL) approach -- training a shared model without moving raw imagery across estates -- can deliver six-class FFB ripeness detection that is both accurate and explainable, making it a trustworthy harvest decision-support tool. A YOLOv11n detector is trained federatively using FedAvg over five communication rounds on Non-IID Dirichlet partitions derived from a leakage-free bunch-identity split (training: 9,094 images from 72 bunches; validation: 769 from 8; test: 951 from 9). All BatchNorm layers are converted to GroupNorm for cross-experiment consistency. Explanation quality is assessed with Grad-CAM++ via per-class Average Drop (AD) and Focus Retention Rate (FRR), the latter doubling as an agronomic alignment proxy measuring heatmap intensity within fruit bounding boxes. The federated model attains mAP\@0.5 = #tbd("B2 mAP50") (centralized reference: 0.787), with global AD = #tbd("AD")% and global FRR = #tbd("FRR"). The per-class analysis reveals that #tbd("best/worst class"). The results indicate FL can deliver privacy-preserving palm ripeness detection without sacrificing decision explainability.
  ],
  keywords: (
    [explainable artificial intelligence],
    [federated learning],
    [Grad-CAM++],
    [oil palm ripeness],
    [YOLOv11],
  ),
  bib: bibliography("references.bib"),
  ..setup,
)

= Introduction

The oil palm industry is a cornerstone of Indonesian and Malaysian agribusiness, with the two countries together accounting for over 85% of the world's palm oil supply @usda2024. At the operational level, the single most influential decision in determining oil yield and quality is the timing of fresh fruit bunch (FFB) harvest. Bunches harvested too early contain insufficient mesocarp oil, whereas bunches harvested too late accumulate Free Fatty Acid (FFA), degrading Crude Palm Oil (CPO) quality. Industry practice distinguishes six commercially relevant ripeness classes: _Unripe_, _Underripe_, _Ripe_, _Overripe_, _Empty Bunch_, and _Abnormal_ @junos2022. Traditional assessment by harvest foremen, based on bunch color and loose fruitlet count, is subjective and has reported misclassification rates of 15--25% under suboptimal lighting @sabri2017.

Modern object detectors such as YOLOv11 can localize and classify multiple bunches in a single frame in real time, making vision-based automation feasible at the plantation scale @yolov11. However, training accurate detectors requires diverse imagery spanning seasons, lighting, and geographic conditions. Centralizing such imagery clashes with a practical reality: plantation images are commercially sensitive -- revealing block locations, production volumes, and operational practices -- and estates are reluctant to ship raw data to a shared server.

Federated learning (FL) @fedavg provides an architectural answer: multiple estates collaboratively train a shared model while raw imagery remains on local infrastructure; only model parameters traverse the network. In the taxonomy of Kairouz et al. @kairouz2021, the plantation-consortium scenario is _cross-silo_ -- few clients, each holding large amounts of data, with stable availability.

Accuracy alone, however, is insufficient for field adoption. An agricultural decision-support system must be trustworthy: operators need to understand _why_ the model labels a bunch as ripe or overripe. Without auditable visual explanations, predictions risk being either rejected outright or trusted uncritically -- both failure modes carry real economic cost. Explainable AI (XAI) methods such as Grad-CAM++ @gradcampp generate heatmaps highlighting the image regions most responsible for a prediction, and the quality of such heatmaps can be measured quantitatively through faithfulness metrics.

Distinct from the authors' prior work that characterizes the behavior of differential privacy on this detector @paperA, the present study addresses an application-oriented question: *can federated learning deliver oil palm ripeness detection that is simultaneously accurate and explainable enough to serve as a credible harvest decision-support tool?* This paper makes three contributions: (1) an in-depth, per-class evaluation of Grad-CAM++ faithfulness on a federated palm ripeness detector, going beyond a single global metric; (2) a bunch-identity-based anti-leakage split protocol for palm detection datasets; and (3) empirical evidence that federation does not significantly degrade the quality of model explanations relative to centralized training.

= Literature Review

== Vision-Based Oil Palm Ripeness Assessment

Several studies have investigated automated palm ripeness assessment using deep learning. Color-feature classifiers @sabri2017 preceded modern CNN approaches. Junos et al. @junos2022 applied a YOLO variant on UAV imagery for fruit-level detection. #tbd("add 2-3 more recent studies with one-line summaries"). A common limitation across these works is the assumption of centralized training and the use of only a single mAP number for validation -- neither the trustworthiness of the visual explanation nor the privacy implications of centralized aggregation is examined. The present work addresses both gaps.

== Federated Learning for Cross-Silo Scenarios

McMahan et al. @fedavg introduced the canonical FedAvg algorithm: clients perform local stochastic gradient descent and the server aggregates updates by a data-size-weighted average. Kairouz et al. @kairouz2021 distinguish _cross-device_ (millions of unreliable clients with little data each) from _cross-silo_ (few reliable clients with large data each) -- a plantation consortium clearly falls in the latter regime. Data heterogeneity between sites is commonly simulated via Dirichlet partitioning @hsu2019, where a concentration parameter $alpha$ controls per-client class imbalance.

== Explainable AI and Faithfulness Metrics

Grad-CAM @gradcam weights each feature-map channel by the spatially-averaged gradient of the target-class score, producing a class activation heatmap. Grad-CAM++ @gradcampp refines this using a weighted combination of positive partial derivatives, providing more accurate localization when multiple instances of the same class are present -- a condition that routinely occurs in FFB imagery containing several bunches per frame. Two complementary faithfulness metrics are used here: Average Drop (AD), measuring the confidence lost when the input is restricted to the heatmap region, and Focus Retention Rate (FRR), measuring the fraction of heatmap intensity falling inside the ground-truth fruit boxes. FRR thus doubles as an agronomic alignment proxy.

= Methodology

== Dataset and Anti-Leakage Split

The dataset is the palm-fruit-ripeness-detection collection (Roboflow, version 2), comprising 10,814 images from 89 unique bunches, annotated across six classes in alphabetical order: _Abnormal_, _Empty Bunch_, _Overripe_, _Ripe_, _Underripe_, _Unripe_. A naive random split risks leakage at the bunch level: multiple frames of the same bunch could be assigned to both training and test partitions, allowing the model to recognize the bunch rather than learn generalizable ripeness cues. To prevent this, a stratified group split keyed by `bunch_id` is applied: each bunch is assigned entirely to one partition, with stratification preserving class balance (@tab-split). A leakage audit confirmed zero bunch-identity overlap between any two partitions.

#figure(
  table(
    columns: 3,
    align: (x, y) => if x > 0 and y > 0 { center } else { center },
    table.header(
      table.hline(),
      [*Partition*], [*Images*], [*Unique bunches*],
      table.hline(),
    ),
    [Train], [9,094], [72],
    [Validation], [769], [8],
    [Test (held out)], [951], [9],
    table.hline(stroke: 0.5pt),
    [*Total*], [*10,814*], [*89*],
    table.hline(),
  ),
  caption: [Leakage-free dataset partition by bunch identity.],
) <tab-split>

== Non-IID Federation Setup

The training partition is divided among #tbd("K, e.g., 4") clients using a Dirichlet distribution with per-client concentration parameters spanning $alpha = 0.1$ (highly skewed) to $alpha = 0.8$ (near uniform), simulating realistic heterogeneity across estates. Validation and test partitions are kept global so that metrics are directly comparable across configurations.

== Detector and Architectural Modification

The base detector is YOLOv11n (~2.6M parameters), initialized from COCO pre-trained weights. All 81 BatchNorm layers are converted in place to GroupNorm (eight groups; auto-reduced when it does not divide a layer's channel count), with affine parameters copied to preserve pre-trained initialization. This ensures consistency with the companion differential-privacy study @paperA, whose per-sample gradient computation requires per-sample-independent normalization. After conversion, zero BatchNorm layers remain.

== Federated Training Procedure

Training uses FedAvg over $T = 5$ communication rounds. Each round, every client receives the global parameters $w_t$, performs $E = 2$ local epochs of SGD (learning rate 0.01, momentum 0.937, weight decay $5 times 10^(-4)$, batch size 16, image size $640 times 640$), and returns $w_t^k$. The server aggregates by data-size weighting (@eq-fedavg):

$ w_(t+1) = sum_(k=1)^(K) n_k / n thin w_t^k $ <eq-fedavg>

where $n_k$ is the local shard size of client $k$ and $n = sum_k n_k$. The loss is the standard YOLOv11 composite of CIoU box loss (weight 7.5), classification loss (0.5), and Distribution Focal Loss (1.5). A centralized reference is trained for 50 epochs on the entire training partition under identical hyperparameters. Formal privacy is treated in detail in the companion paper @paperA; here, privacy is addressed architecturally (raw imagery never leaves the client), keeping the focus on explainability.

== Explanation Quality Metrics

Grad-CAM++ is applied at the last convolutional layer before the detection head. Two faithfulness metrics are computed on $N =$ #tbd("e.g., 120") test images, both globally and per class. Average Drop (AD, %), lower is better, quantifies confidence lost when the input is restricted to the heatmap region (@eq-ad):

$ "AD" = 1/N sum_(i=1)^(N) (max(0, Y_i^c - O_i^c)) / (Y_i^c) times 100% $ <eq-ad>

where $Y_i^c$ is the model's confidence on the original image $i$ for class $c$ and $O_i^c$ its confidence when only the masked region is visible. Focus Retention Rate (FRR), higher is better, gives the fraction of heatmap intensity inside the ground-truth boxes (@eq-frr):

$ "FRR" = (sum_(p in "ROI") I(p)) / (sum_(p in "Image") I(p)) $ <eq-frr>

where $I(p)$ is heatmap intensity at pixel $p$ and ROI is the union of ground-truth boxes for the target class. FRR thus measures both faithfulness and agronomic alignment: does the model attend to the fruit, or to the background?

= Results and Discussion

== Detection Accuracy

@tab-acc compares the federated model against the centralized reference. #tbd("1-2 sentences: report the FL-cost gap and whether the federated model stays in a useful operating range for decision support.")

#figure(
  table(
    columns: 5,
    align: (x, y) => if y == 0 { center } else if x == 0 { left } else { center },
    table.header(
      table.hline(),
      [*Model*], [*mAP\@0.5*], [*mAP\@0.5:0.95*], [*Precision*], [*Recall*],
      table.hline(),
    ),
    [Centralized (reference)], [0.787], [0.672], [0.815], [0.833],
    [Federated (FedAvg)], tbd("v"), tbd("v"), tbd("v"), tbd("v"),
    [FL cost (absolute)], tbd("v"), tbd("v"), tbd("v"), tbd("v"),
    table.hline(),
  ),
  caption: [Detection performance on the held-out test set.],
) <tab-acc>

== Per-Class Faithfulness

@tab-xai reports AD and FRR per ripeness class plus the global average. The best-explained class is #tbd("class") (lowest AD, highest FRR), consistent with the strong color cue characterizing that ripeness stage. The most challenging class is #tbd("class"), where higher AD and lower FRR reflect visual ambiguity -- #tbd("agronomic explanation, e.g., Empty Bunch lacks loose fruitlets so the model latches onto bunch silhouette overlapping with background foliage; Abnormal exhibits high intra-class variation").

#figure(
  table(
    columns: 4,
    align: (x, y) => if y == 0 { center } else if x == 0 { left } else { center },
    table.header(
      table.hline(),
      [*Class*], [*n*], [*AD (%)*], [*FRR*],
      table.hline(),
    ),
    [Abnormal], tbd("n"), tbd("v"), tbd("v"),
    [Empty Bunch], tbd("n"), tbd("v"), tbd("v"),
    [Overripe], tbd("n"), tbd("v"), tbd("v"),
    [Ripe], tbd("n"), tbd("v"), tbd("v"),
    [Underripe], tbd("n"), tbd("v"), tbd("v"),
    [Unripe], tbd("n"), tbd("v"), tbd("v"),
    table.hline(stroke: 0.5pt),
    [*Global*], tbd("n"), tbd("v"), tbd("v"),
    table.hline(),
  ),
  caption: [Per-class Grad-CAM++ faithfulness. AD lower-is-better (%); FRR higher-is-better (0--1). #emph[n]: test images per class.],
) <tab-xai>

This breakdown carries direct operational value. Estates can calibrate operator review effort by ripeness class: classes with strong faithfulness can be acted on with higher automation confidence, while weaker classes are flagged for manual confirmation. A single global number would have obscured this distinction.

== Effect of Federation on Explanation Quality

#tbd("1 paragraph: compare federated vs centralized AD/FRR. If the gap is small (<5% AD increase, <0.05 FRR drop), state FL does not materially harm explanation quality -- the headline finding justifying FL adoption. If large, discuss client drift on Non-IID partitions and mitigations.")

== Qualitative Analysis

@img-xai shows representative Grad-CAM++ heatmaps, one per ripeness class. #tbd("1 paragraph: verify highlights fall on bunch surface and loose fruitlets (correct) rather than canopy/background (incorrect); reference specific subfigures.")

#figure(
  rect(width: 80%, height: 4cm, stroke: 0.5pt + gray)[
    #align(center + horizon)[#tbd("insert figures/xai_per_class.png: 2x3 grid of representative Grad-CAM++ heatmaps, one per ripeness class")]
  ],
  caption: [Representative Grad-CAM++ heatmaps for the six ripeness classes. Highlight intensity (red = high) indicates the image regions most influential to the per-class prediction.],
) <img-xai>

== Threats to Validity

A single dataset source bounds generalization; multi-estate validation is left to future work. Heatmap faithfulness is one operationalization of trustworthiness; complementary user studies with harvest foremen would strengthen the agronomic claim. FRR rewards heatmaps concentrated inside boxes, which may under-credit explanations correctly attending to nearby cues such as fallen fruitlets; per-class AD provides a complementary signal.

= Conclusion

This paper investigated whether federated learning can deliver six-class oil palm ripeness detection that is simultaneously accurate and explainable enough to serve as a trustworthy harvest decision-support tool. On a leakage-free bunch-identity split, a federated YOLOv11n--GroupNorm detector trained with FedAvg attains mAP\@0.5 = #tbd("B2 mAP50") -- a #tbd("small/modest/large") gap from the centralized reference (0.787). The per-class Grad-CAM++ evaluation yields global AD = #tbd("v")% and FRR = #tbd("v"), with #tbd("class") most reliably explained and #tbd("class") most challenging. Federation #tbd("does not materially / only mildly") degrades explanation quality, supporting privacy-preserving collaborative training in plantation consortia. Future work includes integration of per-sample differential privacy @paperA, multi-estate validation across cultivars and climates, and edge-deployment benchmarks for in-field inference.

#set heading(numbering: none)

= CRediT authorship contribution statement

#juti.credits(authors)

= Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

= Acknowledgement

#tbd("optional: funding/thanks, or remove this section")

= Declaration of Generative AI and AI-assisted Technologies in the Writing Process

The authors used #tbd("tool & version, e.g., Claude Sonnet 4.6") to improve the writing clarity of this paper and to assist with formatting. The authors reviewed and edited the AI-assisted content and take full responsibility for the final publication.

= Data availability

The dataset analyzed in this study is openly available from Roboflow (palm-fruit-ripeness-detection, version 2). The leakage-free `bunch_id`-keyed split protocol and partition lists used in this study are available from the corresponding author upon reasonable request.
