# DiT Experiments: A Self-Contained Explanation

[20-slide presentation](dit_results_update.pdf) ·
[Detailed speaker notes, 20 pages](speaker_notes.pdf)

The talk introduces DiT generation and codebooks, then explains each experiment
through its starting point, question, intervention, control and measured outcome.
It covers temporal/structural diagnostics, local refits, temporal codebooks,
training objectives, code reassignment and fitting completed VQDM codebooks.
Negative results and unfinished quality comparisons remain explicit.

The newer 5,000-image FID is not compared to the old 10,000-image baseline as a gain.
All numeric assets required to rebuild this PDF are included. Private raw evidence
and live infrastructure exports are intentionally not included.
The public scientific tables are in [run_summaries](../../run_summaries/README.md).

## Build

From this folder, with XeLaTeX, Arial, Poppler, NumPy and Pillow:

```powershell
New-Item -ItemType Directory -Force _build
xelatex -interaction=nonstopmode -halt-on-error -output-directory=_build dit_results_update.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=_build dit_results_update.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=_build speaker_notes.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=_build speaker_notes.tex
python -B verify_pdf.py
```

The verifier checks 20 slides, 20 note pages, text bounds and figure inclusion.
