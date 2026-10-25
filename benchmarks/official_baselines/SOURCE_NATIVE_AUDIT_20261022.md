# Source-native validation audit

## Current evidence

The completed extension uses the 25 Exathlon and 23 GHL files selected by
the fixed TSB-AD-M manifest. GutenTAG is generated independently with the
official package. These results must retain their preparation identifiers.

The [original Exathlon documentation](https://github.com/exathlonbenchmark/exathlon/wiki/Dataset)
describes 93 application traces (59 undisturbed and 34 disturbed), 2,283
features sampled every second, application/trace identities, and event-time
ground truth. Its normal traces support training independently of disturbed
test traces. The original release needs explicit feature preparation,
missing-value treatment, application matching, and event-to-row labeling.
The [repository](https://github.com/exathlonbenchmark/exathlon) publishes the
raw archives and a CC BY-NC-SA 4.0 data license.

The [TimeEval dataset catalog](https://timeeval.github.io/evaluation-paper/notebooks/Datasets.html)
lists preprocessed Exathlon (39 datasets, variable feature counts) and GHL
(48 datasets, 16 channels) with train/test learning types. It links archives
of approximately 106 MB and 153 MB. These are TimeEval preparations and
should have a separate preparation identifier from the original releases.
The GHL source link redirects to a Kaspersky-hosted release. The web reader
could not inspect that redirected archive, and no native archive was
downloaded or evaluated during this stage.

## Remaining protocol work

1. Preserve the released training/test partition and verify normal-only
   training, channel order, timestamp order, and row labels for each pair.
2. Define application-matched Exathlon pairs before evaluating methods.
   Keep trace boundaries separate, including during derivatives and scaling.
3. Record archive and extracted-file hashes, release/version, license, and
   source-to-prepared feature mapping. Fix any exclusions by data validity.
4. Reject multiple entity IDs within a nominal single-trace pair until an
   entity-aware loader is implemented and tested.
5. Compare every retained pair with fixed detector configurations and the
   same audited baselines. Report the preparation sensitivity separately.

No new native accuracy result is claimed. The three completed extension
sources should not be described as three independent real deployments:
GHL and GutenTAG are synthetic, and the current Exathlon files are TSB-AD
preparations of Spark traces.
