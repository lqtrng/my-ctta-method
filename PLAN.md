The biggest gap: the paper rotates the source domain
The paper doesn't just train on domain A. "Each domain is used in turn as the source, and the model is evaluated on all remaining target domains." Its headline numbers are the Average over all source domains:

Source domains	ReDi-FM "Average"	Its "Do. A" (the only part we have)
OD/OC	A–E (5)	73.59	73.11
Prostate	A–F (6)	61.81	66.10
Polyp	A–D (4)	74.84	77.47
Every number we have is from source domain A only, mostly with a single seed. So the core of the plan is training and running everything for every source domain, with 3 seeds.

What the paper reports, and the Crop-Teach equivalent
#	Paper	Crop-Teach equivalent	Status
T1	OD/OC and prostate: Dice for each source domain plus Average, 3 seeds, R34 target	Crop-Teach (paper loss, native-res) for each source, 3 seeds, plus source-only	Source A only, mostly seed 0
T2	Polyp: Dice, Eφmax and Sα for each source domain, Segformer-B0 target	Same with Segformer-B0, plus the two extra metrics	Missing: no Segformer, no Eφmax or Sα
T3	Long-term: 3 rounds; average, gain over source-only, degradation (OD/OC, prostate)	Same	Only prostate source A, old loss (GPU queue item 1)
T4	Ablation of loss components (OD/OC, 5 source domains, 3 seeds)	Ablation of Crop-Teach's terms, below	Missing
Fig. 3	Prompt strategies for SAM	Crop-placement strategies	Partial: all vs largest regions and native-res vs low-res, source A only
T5	Latency vs Dice (polyp)	ms per image for Crop-Teach	Missing
T6	Class-balancing variants	Only if we keep the paper's soft loss (we now do)	Missing
T7	Foundation-model branch alone vs the full method	Crop-Infer (no adaptation) vs Crop-Teach, all 3 problems	Partial: OD/OC and polyp, source A
T5 (last column)	SAM ViT-B vs ViT-H	DINOv3 ViT-B vs ViT-L (ViT-B weights are already downloaded)	Missing
Decisions needed before computing
Target network, for comparability with the paper's tables. The paper uses R34 for OD/OC and prostate and Segformer-B0 for polyp. There are two problems:

Our prostate R34 source-only scores 26.49, against 51.02 in the paper, so our prostate R34 is broken and needs retraining. Our nnU scores 61.04 but isn't the paper's network.
Polyp would need a Segformer-B0 source model. Ours are R34 (71.06) and nnU (76.07). The paper's source-only is 74.22.
I suggest matching the paper's networks in the main tables and putting nnU in a separate row.

Baselines. Re-implementing TENT, SAR, VPTTA and the other baselines is a lot of work. I suggest citing the paper's numbers, and only where our source-only matches theirs (OD/OC Do. A: 53.67 against 53.59 matches; prostate doesn't yet). Also keep our ReDi-FM + SAM reproduction (RF1) as a sanity check.

Stream order for sources other than A. The paper streams the remaining domains "in a fixed order (A→B→…)". With source C on OD/OC that's A→B→D→E. I'll assume that.

The plan
Phase 0: code (no GPU)

Add --source-domain to all training scripts and to evaluate.py. When the source isn't A, the stream drops that domain and keeps A.
Add the Eφmax and Sα metrics for polyp.
Add a Segformer-B0 network (source-only training and Crop-Teach target).
Add a latency measurement (ms per image) to evaluate.py.
Add ablation switches to the paper loss: drop the crop hard term, the DINOv3 hard term, Φ, the soft term, or the gate.
Phase 1: train models for every source domain. That's 15 source domains (5 + 6 + 4), each needing:

a target network: R34, or Segformer-B0 for polyp
a native-res crop model of the same kind (prostate's native-res = its resized images)
a DINOv3 full-image head to place crops
That's 45 training jobs. This replaces GPU-queue item 2, which only covers domains A and B.

Phase 2: main tables (T1 and T2). Crop-Teach and source-only for each source domain with seeds 0, 1 and 2: 3 × 15 = 45 Crop-Teach runs plus 15 source-only runs.

Phase 3: long-term (T3). The best setting, 3 rounds, for each source domain and seed on OD/OC and prostate. That's 33 runs, each 3× the length of a normal run.

Phase 4: ablations and analysis. OD/OC over all 5 source domains with 3 seeds, unless noted:

T4, loss components, each step adding one piece:
crop teacher only
DINOv3 teacher only
both
plus Φ
plus the soft term
plus the gate (the full method)
Fig. 3 equivalent, crop placement: largest vs all regions, native-res vs low-res, crop scale, and the detector as placer.
T7, Crop-Infer vs Crop-Teach: all 3 problems, every source domain.
Foundation-model size: DINOv3 ViT-B vs ViT-L for placing crops.
T5, latency: polyp, measured once.
Rough cost, assuming 10–20 minutes per run on the RTX 5060:

Phase	Approximate GPU time
Phase 1 (training)	well over a day; each polyp DINOv3 head alone takes about 80 minutes
Phase 2	about 12–15 hours
Phase 3	about 25 hours
T4 alone	about 6 rows × 15 runs ≈ 25 hours
Phases 1–4 are mostly for the rented GPU. Phase 0 and a source-A dry run of every table can run here first.

Order: do Phase 0, then a source-A dry run of T1 to T7 with seed 0 locally. That checks every code path. Then send Phases 1–4 to the rented GPU.

Should I write this into GPU_QUEUE.md, replacing items 1 and 2, and start on the Phase 0 code? Before any GPU work I need your call on decisions 1 and 2: whether to use the paper's networks (retrain prostate R34, add Segformer-B0), and whether to cite the paper's baseline numbers.