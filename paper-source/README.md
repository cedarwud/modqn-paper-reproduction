# Paper Source Snapshot

This directory vendors the minimum paper source chain needed to keep
`modqn-paper-reproduction` portable outside the larger `papers/` workspace.

Included snapshot:

1. `ref/`
   Source PDF for `PAP-2024-MORL-MULTIBEAM`
2. `txt_layout/`
   Regenerated layout-preserving OCR text used for formula/table checks
3. `catalog/`
   Machine-readable catalog snapshot for the same paper

These files are copied from the original workspace authority chain and should be
treated as repo-local authority inputs for this project.

If an external `paper-catalog/` exists elsewhere, use it only as a cross-check.
It must not silently override this repo-local snapshot during standalone work.
