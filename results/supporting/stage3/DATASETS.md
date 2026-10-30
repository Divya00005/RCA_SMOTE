# Stage 3 datasets and selection rule

Written by `python stage3_datasets.py` **before any stage-3 result was looked at**. The final list (51 datasets) was fixed on 2026-10-23 while the run on the first 26 datasets was still going; no result had been opened.

## Group A: known minority sub-groups (Bellinger protocol, majority capped at 2000)

- Source: PMLB (same source as the 15 stage-1/2 datasets), classification datasets, not marked deprecated.
- At least 4 classes (so the Bellinger construction gives a majority of >= 2 classes and >= 2 minority sub-groups).
- Numeric features: at least 80% of features continuous according to PMLB's metadata (all_summary_stats.tsv).
- Enough samples per class: after the Bellinger construction (classes < 10 samples dropped, sub-groups halved, each >= 10, majority <= 2000) there are >= 2 minority sub-groups and the imbalance ratio is >= 2.5 (the smallest IR among the 15 existing datasets is 2.88).
- No near-copies: not made from the same raw data as an existing Group A or Group B dataset, and at most one dataset from a family of views of the same raw data (first in alphabetical order).
- Applying this rule to all PMLB datasets gives exactly 5 new datasets; PMLB has no more.
- To reach about 30, the same rule was applied to the OpenML-CC18 benchmark suite (72 curated datasets; additionally: no missing values, <= 300 features). This adds 4: GesturePhaseSegmentationProcessed (id 4538), first-order-theorem-proving (1475), steel-plates-fault (40982), wall-robot-navigation (1497).

## Group B: standard imbalanced benchmarks (sub-groups unknown, standard metrics only)

- AW-SMOTE's 13 KEEL datasets, exactly as named in their paper (downloaded manually from the KEEL imbalanced repository, full-dataset zip, not the pre-split fold files). Nominal features (flare-F only) are one-hot encoded.
- Plus 8 more KEEL binary imbalanced datasets from families not used anywhere else in this study. Families with numeric or convertible features (at most one nominal attribute, one-hot encoded) and < 10,000 samples: glass, abalone, cleveland, iris (zoo, lymphography, car, kr-vs-k, poker, kddcup are mostly nominal; winequality-white, shuttle, page-blocks, segment, vowel, led7digit, dermatology, ecoli, yeast, vehicle, new-thyroid, wisconsin, pima, haberman, flare, winequality-red are already used). cleveland and iris have one binary version each (cleveland-0_vs_4, iris0). glass (13 versions) and abalone (7 versions) are many re-labellings of the same raw data, so from each we take the versions with the lowest, median and highest imbalance ratio in KEEL's table: glass1 (1.82), glass-0-1-6_vs_2 (10.29), glass5 (22.78); abalone9-18 (16.4), abalone-21_vs_8 (40.5), abalone19 (129.44).
- Plus every dataset of imblearn.datasets.fetch_datasets (27 datasets) EXCEPT: (a) made from the same raw data as a Group A dataset; (b) categorical features (nominal attributes in the source description, or > 10% of features binary); (c) very large: > 10,000 samples or > 300 features.

## Primary metrics (fixed before running)

- Group A: wBA (each minority sub-group counts equally); secondary: recall of the smallest sub-group
- Group B: G-mean (as in AW-SMOTE's paper); F1, MCC, ROC-AUC, PR-AUC also reported

## Reporting subsets

- **A-dev** (5): used in stage 2 to choose the RCA-SMOTE design. Reported separately.
- **A-stage2-heldout** (10): used once in stage 2 to test the frozen method.
- **A-new** (9: 5 PMLB + 4 OpenML-CC18): never used before.
- **Group A main result** = A-stage2-heldout + A-new.
- **B-keel-awsmote** (13), **B-keel-extra** (8) and **B-imblearn** (6). Six KEEL datasets share raw data with a Group A dataset (column `shares_raw_data_with`); Group B is also reported without them.

## Included datasets

| name | group | subset | source | samples | features | minority | IR | minority sub-group sizes | shares raw data with |
|---|---|---|---|---|---|---|---|---|---|
| ecoli | A | A-dev | PMLB | 290 | 7 | 70 | 3.14 | 40 / 20 / 10 |  |
| led7 | A | A-dev | PMLB | 1983 | 7 | 311 | 5.38 | 167 / 83 / 41 / 20 |  |
| shuttle | A | A-dev | PMLB | 2244 | 9 | 244 | 8.2 | 171 / 50 / 13 / 10 |  |
| texture | A | A-dev | PMLB | 2375 | 40 | 375 | 5.33 | 200 / 100 / 50 / 25 |  |
| wine_quality_white | A | A-dev | PMLB | 2282 | 11 | 282 | 7.09 | 175 / 87 / 20 |  |
| yeast | A | A-stage2-heldout | PMLB | 1397 | 8 | 98 | 13.26 | 51 / 25 / 12 / 10 |  |
| vowel | A | A-stage2-heldout | PMLB | 538 | 13 | 88 | 5.11 | 45 / 22 / 11 / 10 |  |
| segmentation | A | A-stage2-heldout | PMLB | 1174 | 19 | 184 | 5.38 | 99 / 49 / 24 / 12 |  |
| satimage | A | A-stage2-heldout | PMLB | 2350 | 36 | 350 | 5.71 | 200 / 100 / 50 |  |
| pendigits | A | A-stage2-heldout | PMLB | 2375 | 16 | 375 | 5.33 | 200 / 100 / 50 / 25 |  |
| dermatology | A | A-stage2-heldout | PMLB | 315 | 34 | 70 | 3.5 | 40 / 20 / 10 |  |
| page_blocks | A | A-stage2-heldout | PMLB | 2200 | 10 | 200 | 10 | 115 / 57 / 28 |  |
| optdigits | A | A-stage2-heldout | PMLB | 2375 | 64 | 375 | 5.33 | 200 / 100 / 50 / 25 |  |
| letter | A | A-stage2-heldout | PMLB | 2375 | 16 | 375 | 5.33 | 200 / 100 / 50 / 25 |  |
| new_thyroid | A | A-stage2-heldout | PMLB | 202 | 5 | 52 | 2.88 | 35 / 17 |  |
| analcatdata_authorship | A | A-new | PMLB | 704 | 70 | 91 | 6.74 | 61 / 30 |  |
| collins | A | A-new | PMLB | 376 | 23 | 63 | 4.97 | 29 / 14 / 10 / 10 |  |
| mfeat_factors | A | A-new | PMLB | 1187 | 216 | 187 | 5.35 | 100 / 50 / 25 / 12 |  |
| movement_libras | A | A-new | PMLB | 224 | 90 | 56 | 3 | 24 / 12 / 10 / 10 |  |
| sleep | A | A-new | PMLB | 2350 | 13 | 350 | 5.71 | 200 / 100 / 50 |  |
| gesture_phase | A | A-new | OpenML-CC18 | 2350 | 32 | 350 | 5.71 | 200 / 100 / 50 |  |
| first_order_theorem_proving | A | A-new | OpenML-CC18 | 2350 | 51 | 350 | 5.71 | 200 / 100 / 50 |  |
| steel_plates_fault | A | A-new | OpenML-CC18 | 1739 | 27 | 273 | 5.37 | 146 / 73 / 36 / 18 |  |
| wall_robot_navigation | A | A-new | OpenML-CC18 | 2300 | 24 | 300 | 6.67 | 200 / 100 |  |
| wisconsin | B | B-keel-awsmote | KEEL | 683 | 9 | 239 | 1.86 | 239 |  |
| pima | B | B-keel-awsmote | KEEL | 768 | 8 | 268 | 1.87 | 268 |  |
| haberman | B | B-keel-awsmote | KEEL | 306 | 3 | 81 | 2.78 | 81 |  |
| vehicle0 | B | B-keel-awsmote | KEEL | 846 | 18 | 199 | 3.25 | 199 |  |
| ecoli1 | B | B-keel-awsmote | KEEL | 336 | 7 | 77 | 3.36 | 77 | ecoli (dev) |
| new-thyroid2 | B | B-keel-awsmote | KEEL | 215 | 5 | 35 | 5.14 | 35 | new_thyroid |
| ecoli3 | B | B-keel-awsmote | KEEL | 336 | 7 | 35 | 8.6 | 35 | ecoli (dev) |
| yeast-0-2-5-6_vs_3-7-8-9 | B | B-keel-awsmote | KEEL | 1004 | 8 | 99 | 9.14 | 99 | yeast |
| yeast-0-2-5-7-9_vs_3-6-8 | B | B-keel-awsmote | KEEL | 1004 | 8 | 99 | 9.14 | 99 | yeast |
| yeast-2_vs_8 | B | B-keel-awsmote | KEEL | 482 | 8 | 20 | 23.1 | 20 | yeast |
| flare-F | B | B-keel-awsmote | KEEL | 1066 | 41 | 43 | 23.79 | 43 |  |
| winequality-red-3_vs_5 | B | B-keel-awsmote | KEEL | 691 | 11 | 10 | 68.1 | 10 |  |
| winequality-red-8_vs_6-7 | B | B-keel-awsmote | KEEL | 855 | 11 | 18 | 46.5 | 18 |  |
| glass1 | B | B-keel-extra | KEEL | 214 | 9 | 76 | 1.82 | 76 |  |
| glass-0-1-6_vs_2 | B | B-keel-extra | KEEL | 192 | 9 | 17 | 10.29 | 17 |  |
| glass5 | B | B-keel-extra | KEEL | 214 | 9 | 9 | 22.78 | 9 |  |
| abalone9-18 | B | B-keel-extra | KEEL | 731 | 10 | 42 | 16.4 | 42 |  |
| abalone-21_vs_8 | B | B-keel-extra | KEEL | 581 | 10 | 14 | 40.5 | 14 |  |
| abalone19 | B | B-keel-extra | KEEL | 4174 | 10 | 32 | 129.44 | 32 |  |
| cleveland-0_vs_4 | B | B-keel-extra | KEEL | 173 | 13 | 13 | 12.31 | 13 |  |
| iris0 | B | B-keel-extra | KEEL | 150 | 4 | 50 | 2 | 50 |  |
| spectrometer | B | B-imblearn | imblearn | 531 | 93 | 45 | 10.8 | 45 |  |
| us_crime | B | B-imblearn | imblearn | 1994 | 100 | 150 | 12.29 | 150 |  |
| yeast_ml8 | B | B-imblearn | imblearn | 2417 | 103 | 178 | 12.58 | 178 |  |
| scene | B | B-imblearn | imblearn | 2407 | 294 | 177 | 12.6 | 177 |  |
| oil | B | B-imblearn | imblearn | 937 | 49 | 41 | 21.85 | 41 |  |
| ozone_level | B | B-imblearn | imblearn | 2536 | 72 | 73 | 33.74 | 73 |  |

## Excluded candidates

| source | dataset(s) | reason |
|---|---|---|
| PMLB | cloud | imbalance ratio after construction 1.67 < 2.5 (108 samples, 4 similar-size classes) |
| PMLB | confidence | imbalance ratio after construction 1.12 < 2.5 (72 samples) |
| PMLB | mfeat_fourier / mfeat_karhunen / mfeat_zernike | other views of the same raw data as mfeat_factors |
| PMLB | mfeat_morphological / mfeat_pixel | same raw data as mfeat_factors; also < 80% continuous |
| PMLB | mnist | near-copy of optdigits/pendigits (NIST handwritten digits) |
| PMLB | led24 | near-copy of led7 |
| PMLB | vehicle | same raw data as KEEL vehicle0 (Group B) |
| PMLB | wine_quality_red | same raw data as KEEL winequality-red-* (Group B) |
| PMLB | kddcup, fars, poker, nursery, krkopt, car_evaluation, soybean, lymphography, flags, calendarDOW, allhyper, allrep, analcatdata_dmft, analcatdata_germangss, auto_insurance_symboling, horse_colic_lesion_type | < 80% continuous features (nominal/coded attributes) |
| OpenML-CC18 | mfeat-*, optdigits, pendigits, satimage, segment, texture, letter, vowel, vehicle, analcatdata_authorship | already used (Group A or B) or same raw data |
| OpenML-CC18 | semeion, mnist_784 | near-copies of the digit datasets (semeion is also binary pixels) |
| OpenML-CC18 | har, isolet, cnae-9, Fashion-MNIST, CIFAR_10, Devnagari-Script | > 300 features |
| OpenML-CC18 | MiceProtein, eucalyptus | missing values / nominal features |
| OpenML-CC18 | car, analcatdata_dmft, and the CC18 datasets with < 4 classes | nominal features or < 4 classes |
| KEEL | zoo-3, lymphography-normal-fibrosis, car-*, kr-vs-k-*, poker-*, kddcup-* | mostly nominal features |
| KEEL | other glass and abalone versions | same raw data as the 3 versions taken from each family |
| KEEL | winequality-white-*, shuttle-*, page-blocks*, segment0, vowel0, led7digit-*, dermatology-6, ecoli-*, yeast*, vehicle1-3, new-thyroid1 | family already used in Group A or B |
| PMLB | _deprecated_* (glass, cleveland, auto, car, solar_flare, ...) | marked deprecated by PMLB |
| imblearn | ecoli, optical_digits, satimage, pen_digits, letter_img, yeast_me2, wine_quality | same raw data as a Group A dataset |
| imblearn | libras_move | same raw data as movement_libras (Group A) |
| imblearn | abalone, abalone_19, sick_euthyroid, thyroid_sick, car_eval_34, car_eval_4, solar_flare_m0, coil_2000 | categorical features |
| imblearn | arrhythmia | categorical: 83 of 278 features are binary |
| imblearn | isolet (617 features), webpage (34,780 samples), mammography (11,183), protein_homo (145,751) | very large |

## Notes

- Some KEEL zips also contain macOS helper files (__MACOSX/._*.dat); they are ignored.
- cleveland-0_vs_4: the KEEL file has 173 rows (KEEL's web table says 177, probably counting rows with missing values that are not in the file). The file is used as it is. KEEL stores this dataset's coded attributes (sex, chest-pain type, ECG result, ...) as numbers; they are used as given, like every other KEEL dataset.
- abalone*: the one nominal attribute (Sex) is one-hot encoded, as for flare-F.

## KEEL download links (full dataset zip; save as `data/keel/<name>.zip`)

- wisconsin: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/wisconsin.zip
- pima: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/pima.zip
- haberman: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/haberman.zip
- vehicle0: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/vehicle0.zip
- ecoli1: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/ecoli1.zip
- new-thyroid2: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/new-thyroid2.zip
- ecoli3: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/ecoli3.zip
- yeast-0-2-5-6_vs_3-7-8-9: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/yeast-0-2-5-6_vs_3-7-8-9.zip
- yeast-0-2-5-7-9_vs_3-6-8: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/yeast-0-2-5-7-9_vs_3-6-8.zip
- yeast-2_vs_8: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/yeast-2_vs_8.zip
- flare-F: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/flare-F.zip
- winequality-red-3_vs_5: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/winequality-red-3_vs_5.zip
- winequality-red-8_vs_6-7: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/winequality-red-8_vs_6-7.zip
- glass1: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/glass1.zip
- glass-0-1-6_vs_2: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/glass-0-1-6_vs_2.zip
- glass5: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/glass5.zip
- abalone9-18: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/abalone9-18.zip
- abalone-21_vs_8: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/abalone-21_vs_8.zip
- abalone19: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/abalone19.zip
- cleveland-0_vs_4: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/cleveland-0_vs_4.zip
- iris0: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/iris0.zip
