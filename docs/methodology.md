# Methodology

The frozen RCA-SMOTE configuration is V7: minority DBSCAN with MinPts=2 and radius percentile 90; noise-point rescue using five training neighbours and a 95th-percentile attachment threshold; square-root size quotas with existing rounding; evenly cycled seeds and a partner among five globally retained minority neighbours; uniform interpolation. No algorithm tuning or quota repair was performed for this release.

Dependency chain: rca_smote -> improved -> rc_aw_smote -> pinned fetched AW implementation. The corrected documentation does not imply a minimum quota guarantee. See paper/main.tex for equations and Algorithm 1.
