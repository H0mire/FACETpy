# Model → Source-Paper References

Each deep-learning model under `src/facet/models/` adapts an architecture from
the literature (or is an internal prototype). This file maps every model to its
source paper(s).

The actual paper PDFs are kept **locally** under `output/papers/` (that folder
is gitignored, so the binaries are not committed). The canonical arXiv/DOI links
below are the source of truth; re-download instructions are at the bottom.

The in-repo survey `docs/research/dl_eeg_gradient_artifacts.pdf` is the umbrella
literature review several models point into (section numbers noted below).

## Models derived from a published paper

| Model | Architecture | Source paper | arXiv / DOI | Local PDF (`output/papers/`) |
|---|---|---|---|---|
| `conv_tasnet` | Conv-TasNet | Luo & Mesgarani, *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation*, IEEE/ACM TASLP 27(8), 2019 | [arXiv:1809.07454](https://arxiv.org/abs/1809.07454) | `conv-tasnet_luo-mesgarani_2019_arxiv-1809.07454.pdf` |
| `demucs` | Demucs (time-domain) | Défossez et al., *Music Source Separation in the Waveform Domain*, 2019 | [arXiv:1911.13254](https://arxiv.org/abs/1911.13254) | `demucs_defossez_2019_arxiv-1911.13254.pdf` |
| `sepformer` | SepFormer | Subakan et al., *Attention is All You Need in Speech Separation*, ICASSP 2021 | [arXiv:2010.13154](https://arxiv.org/abs/2010.13154) | `sepformer_subakan_2021_arxiv-2010.13154.pdf` |
| `d4pm` | D4PM (dual-branch diffusion) | *D4PM: A Dual-branch Driven Denoising Diffusion Probabilistic Model … for EEG Artifacts Removal*, 2025 | [arXiv:2509.14302](https://arxiv.org/abs/2509.14302) | `d4pm_2025_arxiv-2509.14302.pdf` |
| `dpae` | Dual-Pathway Autoencoder | Xiong, Ma & Li, *A general dual-pathway network for EEG denoising*, Front. Neurosci. 17, 2023 | [10.3389/fnins.2023.1258024](https://doi.org/10.3389/fnins.2023.1258024) | `dpae_xiong_2023_frontiers_fnins.2023.1258024.pdf` |
| `ic_unet` | IC-U-Net | Chuang et al., *IC-U-Net: A U-Net-based Denoising Autoencoder … for Automatic EEG Artifact Removal*, NeuroImage 263, 2022 | [arXiv:2111.10026](https://arxiv.org/abs/2111.10026) · [10.1016/j.neuroimage.2022.119586](https://doi.org/10.1016/j.neuroimage.2022.119586) | `ic-u-net_chuang_2022_arxiv-2111.10026.pdf` |
| `st_gnn` | Spatio-temporal GCN | Yu, Yin & Zhu, *Spatio-Temporal Graph Convolutional Networks*, IJCAI 2018 | [arXiv:1709.04875](https://arxiv.org/abs/1709.04875) | `stgcn_yu-yin-zhu_2018_arxiv-1709.04875.pdf` |
| `st_gnn` | (electrode graph) EEG-GCNN | Wagh & Varatharajah, *EEG-GCNN*, ML4H 2020 | [arXiv:2011.12107](https://arxiv.org/abs/2011.12107) | `eeg-gcnn_wagh-varatharajah_2020_arxiv-2011.12107.pdf` |
| `st_gnn` | (spectral filters) ChebNet | Defferrard, Bresson & Vandergheynst, *Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering*, NeurIPS 2016 | [arXiv:1606.09375](https://arxiv.org/abs/1606.09375) | `chebnet_defferrard_2016_arxiv-1606.09375.pdf` |
| `vit_spectrogram` | Vision Transformer | Dosovitskiy et al., *An Image is Worth 16×16 Words*, ICLR 2021 | [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) | `vit_dosovitskiy_2021_arxiv-2010.11929.pdf` |
| `vit_spectrogram` | (inpainting) MAE | He et al., *Masked Autoencoders Are Scalable Vision Learners*, CVPR 2022 | [arXiv:2111.06377](https://arxiv.org/abs/2111.06377) | `mae_he_2022_arxiv-2111.06377.pdf` |
| `dhct_gan`, `dhct_gan_v2` | DHCT-GAN | Cai et al., *DHCT-GAN: Improving EEG Signal Quality with a Dual-Branch Hybrid CNN-Transformer Network*, MDPI Sensors 25(1):231, 2025 | [10.3390/s25010231](https://doi.org/10.3390/s25010231) | — *(open access, but MDPI blocks automated download — fetch manually)* |
| `denoise_mamba` | DenoiseMamba (ConvSSD) | Liu et al., *DenoiseMamba: An Innovative Approach for EEG Artifact Removal Leveraging Mamba and CNN*, IEEE JBHI, 2025 | [IEEE Xplore 11012652](https://ieeexplore.ieee.org/document/11012652) · PMID 40408214 | — *(paywalled)* |
| `nested_gan` | Nested-GAN | *End-to-End EEG Artifact Removal Method via Nested Generative Adversarial Network*, Biomed. Phys. Eng. Express, 2025 | [10.1088/2057-1976/ae1a8c](https://doi.org/10.1088/2057-1976/ae1a8c) · PMID 41183389 | — *(paywalled)* |

`nested_gan` additionally borrows Transformer blocks (MDTA / GDFN) from Restormer:
Zamir et al., *Restormer: Efficient Transformer for High-Resolution Image
Restoration*, CVPR 2022 — [arXiv:2111.09881](https://arxiv.org/abs/2111.09881).
`sepformer` uses the sinusoidal positional encoding from Vaswani et al.,
*Attention Is All You Need*, NeurIPS 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).

## Internal prototypes (no external source paper)

| Model | Note |
|---|---|
| `cascaded_dae` | Channel-wise cascaded denoising autoencoder; ported from the older FACETpy `feature/deeplearning` prototype. |
| `cascaded_context_dae` | 7-epoch-context variant of `cascaded_dae`. |
| `demo01` | Frozen proof-of-concept (7-epoch context CNN); not based on a publication. |

## Survey cross-references

`docs/research/dl_eeg_gradient_artifacts.pdf` is the project's literature review;
several models cite a section of it:

- `denoise_mamba` → §6.2
- `conv_tasnet` → §7.1.1
- `demucs` → §7.1.2
- `vit_spectrogram` → §7.2.1
- `st_gnn` → §7.3

## (Re-)downloading the PDFs

Open-access PDFs live in the gitignored `output/papers/`. To repopulate them:

```bash
mkdir -p output/papers && cd output/papers
for spec in \
  "1809.07454:conv-tasnet_luo-mesgarani_2019" \
  "1911.13254:demucs_defossez_2019" \
  "2010.13154:sepformer_subakan_2021" \
  "2509.14302:d4pm_2025" \
  "2111.10026:ic-u-net_chuang_2022" \
  "1709.04875:stgcn_yu-yin-zhu_2018" \
  "2011.12107:eeg-gcnn_wagh-varatharajah_2020" \
  "1606.09375:chebnet_defferrard_2016" \
  "2010.11929:vit_dosovitskiy_2021" \
  "2111.06377:mae_he_2022" ; do
    id="${spec%%:*}"; name="${spec##*:}"; curl -sSL -o "${name}_arxiv-${id}.pdf" "https://arxiv.org/pdf/${id}"
done
# Frontiers (open access):
curl -sSL -A "Mozilla/5.0" -o "dpae_xiong_2023_frontiers_fnins.2023.1258024.pdf" \
  "https://www.frontiersin.org/articles/10.3389/fnins.2023.1258024/pdf"
# DHCT-GAN (MDPI, open access but Cloudflare-protected): download manually from
#   https://www.mdpi.com/1424-8220/25/1/231
```
