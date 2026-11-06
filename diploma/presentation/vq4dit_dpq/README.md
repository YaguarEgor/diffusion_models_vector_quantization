# VQ4DiT and DPQ

[28-slide presentation](review.pdf) · [Detailed speaker notes, 28 pages](speaker_notes.pdf)

Eight introductory slides explain DiT, conditioning, DDPM/DDIM and ODE/SDE.
Ten slides then cover each article: representation, calibration, assignments,
training objectives, storage and published results. The notes include concrete
numeric examples, explanations of formulas and answers to likely questions.

Tables in this talk are article results, not newly measured reproduction results.
Nominal index precision is distinguished from full storage cost. A common input
to two complete networks is not assumed to give identical inputs to every block.
The DPQ E-step and M-step use different objectives; no FID monotonicity is claimed.

## Sources

- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- [VQ4DiT: Efficient Post-Training Vector Quantization for Diffusion Transformers](https://ojs.aaai.org/index.php/AAAI/article/view/33782/35937)
- [Memory-Efficient Generative Models via Product Quantization](https://openaccess.thecvf.com/content/ICCV2025/papers/Shao_Memory-Efficient_Generative_Models_via_Product_Quantization_ICCV_2025_paper.pdf)
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)

## Build

From this folder, with XeLaTeX, Arial, Poppler, NumPy and Pillow:

```powershell
New-Item -ItemType Directory -Force _build
xelatex -interaction=nonstopmode -halt-on-error -output-directory=_build review.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=_build review.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=_build speaker_notes.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=_build speaker_notes.tex
python -B verify_pdf.py
```
