# Alignment v2

Alignment v2 is a deterministic, programmatically validated pilot dataset for MiniMind-Align. The generator uses seed 42, Python standard-library arithmetic/JSON/CSV rendering, maintained safety templates, and a fail-closed audit.

The new pilot has 1000 train and 160 validation examples in eight categories. The final SFT pilot appends the chosen responses from Alignment v1 without modifying any Alignment v1 file. Validation contains only the independent v2 validation set; the frozen v1 test prompts are never used as training material.

Run from the repository root:

    .venv/bin/python dataset/alignment_v2/build_alignment_v2.py --mode smoke
    .venv/bin/python dataset/alignment_v2/audit_alignment_v2.py --mode smoke
    .venv/bin/python dataset/alignment_v2/build_alignment_v2.py --mode pilot
    .venv/bin/python dataset/alignment_v2/audit_alignment_v2.py --mode pilot

Every generated record has a manifest entry with category, family, method, validator, seed and metadata. The audit writes JSON/Markdown reports and exits non-zero on any critical failure.
