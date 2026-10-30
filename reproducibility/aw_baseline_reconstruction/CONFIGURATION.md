# Diagnostic reconstruction configuration and ambiguity register

This study does not claim an exact historical reconstruction: the paper does not publish the numeric CV seed, exact preprocessing/encoding, all sampler RNG policies, or the complete experiment runner. Stated historical library versions are matched in an isolated Python 3.10.19 environment; the patch level and imbalanced-learn/pandas auxiliary versions are audit choices.

## Legitimate evaluation

- Full original KEEL rows and original minority-positive targets; no RCA Bellinger transformation or sample cap.
- StratifiedKFold(n_splits=5,shuffle=True,random_state=42). All methods share original train/test indices. Seed 42 comes from the published code's constant, not a numeric seed stated in the paper. Samplers and seeded classifiers re-created with random_state=42 for each fold.
- Primary input: numeric columns unchanged, nominal header domains encoded as 0..L-1 in header order. This keeps Table 1/Table 5 original attribute counts, but imposes an arbitrary Euclidean ordering on nominal attributes and is not authenticated as the historical preprocessing.
- StandardScaler fit only on original training rows; sampler applied only to scaled training rows. The original test set remains untouched. Sensitivities use no scaling, training-only MinMaxScaler, and header-domain one-hot encoding. Prespecified sensitivity datasets: pima, flare-F, winequality-red-3_vs_5, abalone19, kr-vs-k-three_vs_eleven and segment0; no favorable setting is selected as a replacement primary result.
- DT: gini, unlimited depth, max_features=None; kNN 5, Euclidean, uniform; SVM C=1, RBF gamma='scale', probability=True; GBoost 100 estimators, remaining sklearn 1.3.2 defaults. Explicit random_state=42 where applicable. Full parameters in ENVIRONMENT.json.
- F1 is minority-positive binary F1, G-mean uses test confusion sensitivity/specificity, AUC uses predict_proba for positive class. SVC predict remains its native decision-rule class, not a tuned probability threshold. Majority/macro/weighted F1 are diagnostics from those same predictions.
- Macro-average five fold scores within dataset/classifier/method, then equal-weight datasets, then four classifiers. Also retain pooled out-of-fold prediction metrics as a different aggregation, never silently swap.
- A sampler error is recorded for all four classifiers and no replacement sampler is substituted. Report missingness/counts before means. No completed-cohort score may silently omit failed dataset/fold evaluations.

## Untouched baseline

sources/adw.py is an exact pinned upstream download, SHA-256 72c3d438093538d0c5c3d19bc5dc456be9cdc743a7b907bfec5abd2591d15124. The primary AW row instantiates ADWSMOTE directly, not the repository's safe-epsilon wrapper. Diagnostic inspection after sampling makes no random draws and does not change samples. No correction is applied to its mask, rounding, retained noise, lambda interval or fallback behavior.

## Separate equation-oriented implementation

`equation_aw_smote.EquationAWSMOTE` implements the nondegenerate equations (1–12) and the cleaned-D interpretation of Algorithm 1. It is explicitly assumption-dependent, not a unique certified implementation of the entire internally ambiguous specification.

| Item | Audit interpretation | Reason/limit |
|---|---|---|
| DBSCAN | MinPts=2, 65th percentile of first non-self minority distance, as pinned code | Final PDF says 'k-distance' without fixing index/self convention. Retain author convention initially so clear corrections are not confounded with an invented radius rule. |
| Boundary kNN | Five other observations, own row identity excluded even with duplicate coordinates | Conventional mathematical interpretation; paper does not explicitly state self handling. |
| Neighbour universe | Cleaned D with correctly aligned labels, reused in both weighting stages | Algorithm 1 lines 4–5 explicitly rebuild D. |
| Eq 4 rounding | floor(q+.5), residual to largest normalized weight | Matches prose on PDF p8; reject negative corrected quotas rather than invent extra constrained apportionment. |
| Density | Mean Euclidean distance to other rows, delta=1e-8 | Formula matches; delta numeric value inherited from code because article does not specify it. |
| Pi allocation | Multinomial draws grouped by seed, as code | Exact seed-allocation mechanism is not specified by article. |
| Partner | Uniform distinct member of same cluster | 'RandomNeighbor' subset/k is not defined; no claim of guaranteed nearest-partner sampling. |
| Interpolation | Uniform [0,1), practical PRNG convention for continuous (0,1) | Confirmed change from released [.2,.8]. |
| Final training set | Cleaned original X plus synthetic points | Algorithm 1 updates D. |
| Budget N | Pre-denoising majority-minus-minority supplied before clustering | Algorithm 1 Require defines supplied N; final 1:1 claim then conflicts with physical noise removal. Class counts explicitly recorded. |
| Undefined cases | Raise UnspecifiedPaperCase for zero epsilon, no valid clusters, zero denominator or negative quota | No paper-defined output exists in those branches. Do not silently add a uniform/no-noise fallback. |

`MaskAlignedAWSMOTE` is an additional isolated Stage-3 mask diagnostic. It retains the original radius, Stage-2 universe, rounding, random draw sequence and [.2,.8] interpolation. It corrects only the label array associated with the reordered Stage-3 D. It does not purport to satisfy Algorithm 1's full cleaned-D behavior.

## Clearly invalid control

`INVALID_preCV_resampling`: fit the scaler on the full dataset; run the sampler on the full dataset; then split the resampled dataset into five stratified folds. This knowingly exposes evaluation to synthetic relatives and original test information. Synthetic test-row counts/prevalence are recorded. It is never included in legitimate reproduced tables or used as a scientific performance estimate. Any similarity to published numbers is evidence of compatibility with this failure mode, not proof of the authors' undisclosed runner.

## Coverage

Primary: six methods (None, pinned AW, SMOTE, frozen RCA, Equation-AW, mask-only) × 16 datasets × 5 folds × 4 classifiers. Supplementary classic rows: smote-variants 1.0.1 SYMPROD, SMOTEWB and AMSCO on 13 datasets, defaults with seed 42. GSMOTE and RSMOTE sources/revisions are unspecified by the paper/pinned repository and will remain unverified unless authoritative implementation identity is recovered. Partial-cohort ranks cannot reproduce the six-method Table 4 rankings.

No model selection, RCA changes, manuscript writing, original-evidence overwrite or GitHub push is authorized or performed here.
