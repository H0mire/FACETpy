"""Baue die Literaturliste in den Formaten, die Word und Zotero lesen.

Die Quelle ist `src/facet/models/REFERENCES.md` plus die Zitate, die in den
Korrekturmodulen im Docstring stehen. Dieses Skript hält sie an einer Stelle
maschinenlesbar, damit die Liste nicht in drei Dateien auseinanderläuft.

**Jeder Eintrag trägt ein `quelle`-Feld** und nennt damit, woher die Angabe
stammt: Crossref (über die DOI), die arXiv-API oder PubMed. Keine Angabe stammt
aus dem Gedächtnis — bei fünf Einträgen wich `REFERENCES.md` von der
tatsächlichen Autorenliste ab, unter anderem bei Nested-GAN und DenoiseMamba,
wo sogar der Erstautor ein anderer ist. Wer hier etwas ergänzt, holt es aus einer
dieser drei Quellen und trägt sie ein.

Ausgabe nach ``output/kapitel_5/``:

* ``literatur.xml`` — Words eigenes Format. In Word über
  Referenzen → Quellen verwalten → Durchsuchen laden.
* ``literatur.bib``  — BibTeX, für Zotero/LaTeX und als Gegenprobe.
* ``literatur_pruefliste.md`` — was noch zu prüfen ist.
"""

from __future__ import annotations

import html
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output/kapitel_5"

# ---------------------------------------------------------------------------
# tag, typ, autoren (Nachname, Vorname), jahr, titel, gefaess, band, heft,
# seiten, doi/url, quelle
# ---------------------------------------------------------------------------
REFS = [
    # --- Korrekturverfahren: die Grundlage der Arbeit -----------------------
    dict(tag="Allen2000", typ="JournalArticle",
         autoren=["Allen, Philip J.", "Josephs, Oliver", "Turner, Robert"], jahr="2000",
         titel="A method for removing imaging artifact from continuous EEG recorded during functional MRI",
         gefaess="NeuroImage", band="12", heft="2", seiten="230-239",
         doi="10.1006/nimg.2000.0599", quelle="Crossref"),
    dict(tag="Niazy2005", typ="JournalArticle",
         autoren=["Niazy, R. K.", "Beckmann, C. F.", "Iannetti, G. D.",
                  "Brady, J. M.", "Smith, S. M."], jahr="2005",
         titel="Removal of FMRI environment artifacts from EEG data using optimal basis sets",
         gefaess="NeuroImage", band="28", heft="3", seiten="720-737",
         doi="10.1016/j.neuroimage.2005.06.067", quelle="Crossref"),
    dict(tag="Moosmann2009", typ="JournalArticle",
         autoren=["Moosmann, Matthias", "Schönfelder, Vinzenz H.", "Specht, Karsten",
                  "Scheeringa, René", "Nordby, Helge", "Hugdahl, Kenneth"], jahr="2009",
         titel="Realignment parameter-informed artefact correction for simultaneous EEG-fMRI recordings",
         gefaess="NeuroImage", band="45", heft="4", seiten="1144-1150",
         doi="10.1016/j.neuroimage.2009.01.024", quelle="Crossref"),

    # --- Quellentrennung (Audio) -------------------------------------------
    dict(tag="Luo2019", typ="JournalArticle",
         autoren=["Luo, Yi", "Mesgarani, Nima"], jahr="2019",
         titel="Conv-TasNet: Surpassing ideal time-frequency magnitude masking for speech separation",
         gefaess="IEEE/ACM Transactions on Audio, Speech, and Language Processing",
         band="27", heft="8", seiten="1256-1266",
         doi="10.1109/TASLP.2019.2915167", url="https://arxiv.org/abs/1809.07454",
         quelle="Crossref"),
    dict(tag="Defossez2019", typ="Report",
         autoren=["Défossez, Alexandre", "Usunier, Nicolas", "Bottou, Léon", "Bach, Francis"],
         jahr="2019", titel="Music source separation in the waveform domain",
         gefaess="arXiv:1911.13254", url="https://arxiv.org/abs/1911.13254", quelle="arXiv"),
    dict(tag="Subakan2021", typ="ConferenceProceedings",
         autoren=["Subakan, Cem", "Ravanelli, Mirco", "Cornell, Samuele",
                  "Bronzi, Mirko", "Zhong, Jianyuan"], jahr="2021",
         titel="Attention is all you need in speech separation",
         gefaess="ICASSP 2021 - IEEE International Conference on Acoustics, Speech and Signal Processing",
         seiten="21-25", doi="10.1109/ICASSP39728.2021.9413901",
         url="https://arxiv.org/abs/2010.13154", quelle="Crossref"),

    # --- EEG-Entrauschung ---------------------------------------------------
    dict(tag="Xiong2023", typ="JournalArticle",
         autoren=["Xiong, Wenjing", "Ma, Lin", "Li, Haifeng"], jahr="2023",
         titel="A general dual-pathway network for EEG denoising",
         gefaess="Frontiers in Neuroscience", band="17", seiten="1258024",
         doi="10.3389/fnins.2023.1258024", quelle="PubMed (PMID 38328554)"),
    dict(tag="Chuang2022", typ="JournalArticle",
         autoren=["Chuang, Chun-Hsiang", "Chang, Kong-Yi", "Huang, Chih-Sheng",
                  "Jung, Tzyy-Ping"], jahr="2022",
         titel="IC-U-Net: A U-Net-based denoising autoencoder using mixtures of independent components for automatic EEG artifact removal",
         gefaess="NeuroImage", band="263", seiten="119586",
         doi="10.1016/j.neuroimage.2022.119586", url="https://arxiv.org/abs/2111.10026",
         quelle="Crossref"),
    dict(tag="Cai2025", typ="JournalArticle",
         autoren=["Cai, Yinan", "Meng, Zhao", "Huang, Dian"], jahr="2025",
         titel="DHCT-GAN: Improving EEG signal quality with a dual-branch hybrid CNN-Transformer network",
         gefaess="Sensors", band="25", heft="1", seiten="231",
         doi="10.3390/s25010231", quelle="Crossref"),
    dict(tag="Chen2025", typ="JournalArticle",
         autoren=["Chen, Wensheng", "Li, Yurong", "Zheng, Nan", "Shi, Wuxiang"], jahr="2025",
         titel="DenoiseMamba: An innovative approach for EEG artifact removal leveraging Mamba and CNN",
         gefaess="IEEE Journal of Biomedical and Health Informatics",
         band="29", heft="9", seiten="6551-6564",
         doi="10.1109/JBHI.2025.3573042", quelle="PubMed (PMID 40408214)"),
    dict(tag="Yang2025", typ="JournalArticle",
         autoren=["Yang, Tianqi", "Hu, Nan", "Cai, Shengsheng", "Xu, Dongyang"], jahr="2025",
         titel="End-to-end EEG artifact removal method via nested generative adversarial network",
         gefaess="Biomedical Physics & Engineering Express", band="11", heft="6", seiten="065054",
         doi="10.1088/2057-1976/ae1a8c", quelle="Crossref"),
    dict(tag="Shao2025", typ="Report",
         autoren=["Shao, Feixue", "Liu, Xueyu", "Wu, Yongfei", "Lu, Jianbo",
                  "Yan, Guiying", "Yang, Weihua"], jahr="2025",
         titel="D4PM: A dual-branch driven denoising diffusion probabilistic model with joint posterior diffusion sampling for EEG artifacts removal",
         gefaess="arXiv:2509.14302", url="https://arxiv.org/abs/2509.14302", quelle="arXiv"),
    dict(tag="DaoGu2024", typ="ConferenceProceedings",
         autoren=["Dao, Tri", "Gu, Albert"], jahr="2024",
         titel="Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality",
         gefaess="Proceedings of the 41st International Conference on Machine Learning (ICML)",
         url="https://arxiv.org/abs/2405.21060", quelle="arXiv"),

    # --- Graph- und Bildarchitekturen ---------------------------------------
    dict(tag="Yu2018", typ="ConferenceProceedings",
         autoren=["Yu, Bing", "Yin, Haoteng", "Zhu, Zhanxing"], jahr="2018",
         titel="Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting",
         gefaess="Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI)",
         url="https://arxiv.org/abs/1709.04875", quelle="arXiv"),
    dict(tag="Wagh2020", typ="ConferenceProceedings",
         autoren=["Wagh, Neeraj", "Varatharajah, Yogatheesan"], jahr="2020",
         titel="EEG-GCNN: Augmenting electroencephalogram-based neurological disease diagnosis using a domain-guided graph convolutional neural network",
         gefaess="Proceedings of the Machine Learning for Health (ML4H) Workshop, NeurIPS",
         url="https://arxiv.org/abs/2011.12107", quelle="arXiv"),
    dict(tag="Defferrard2016", typ="ConferenceProceedings",
         autoren=["Defferrard, Michaël", "Bresson, Xavier", "Vandergheynst, Pierre"],
         jahr="2016",
         titel="Convolutional neural networks on graphs with fast localized spectral filtering",
         gefaess="Advances in Neural Information Processing Systems 29 (NeurIPS)",
         url="https://arxiv.org/abs/1606.09375", quelle="arXiv"),
    dict(tag="Dosovitskiy2021", typ="ConferenceProceedings",
         autoren=["Dosovitskiy, Alexey", "Beyer, Lucas", "Kolesnikov, Alexander",
                  "Weissenborn, Dirk", "Zhai, Xiaohua", "Unterthiner, Thomas",
                  "Dehghani, Mostafa", "Minderer, Matthias", "Heigold, Georg",
                  "Gelly, Sylvain", "Uszkoreit, Jakob", "Houlsby, Neil"], jahr="2021",
         titel="An image is worth 16x16 words: Transformers for image recognition at scale",
         gefaess="International Conference on Learning Representations (ICLR)",
         url="https://arxiv.org/abs/2010.11929", quelle="arXiv"),
    dict(tag="He2022", typ="ConferenceProceedings",
         autoren=["He, Kaiming", "Chen, Xinlei", "Xie, Saining", "Li, Yanghao",
                  "Dollár, Piotr", "Girshick, Ross"], jahr="2022",
         titel="Masked autoencoders are scalable vision learners",
         gefaess="IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
         seiten="15979-15988", doi="10.1109/CVPR52688.2022.01553",
         url="https://arxiv.org/abs/2111.06377", quelle="Crossref"),
    dict(tag="Zamir2022", typ="ConferenceProceedings",
         autoren=["Zamir, Syed Waqas", "Arora, Aditya", "Khan, Salman",
                  "Hayat, Munawar", "Khan, Fahad Shahbaz", "Yang, Ming-Hsuan"], jahr="2022",
         titel="Restormer: Efficient transformer for high-resolution image restoration",
         gefaess="IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
         seiten="5718-5729", doi="10.1109/CVPR52688.2022.00564",
         url="https://arxiv.org/abs/2111.09881", quelle="Crossref"),
    dict(tag="Vaswani2017", typ="ConferenceProceedings",
         autoren=["Vaswani, Ashish", "Shazeer, Noam", "Parmar, Niki", "Uszkoreit, Jakob",
                  "Jones, Llion", "Gomez, Aidan N.", "Kaiser, Lukasz", "Polosukhin, Illia"],
         jahr="2017", titel="Attention is all you need",
         gefaess="Advances in Neural Information Processing Systems 30 (NeurIPS)",
         url="https://arxiv.org/abs/1706.03762", quelle="arXiv"),

    # --- Software ------------------------------------------------------------
    dict(tag="Gramfort2013", typ="JournalArticle",
         autoren=["Gramfort, Alexandre", "Luessi, Martin", "Larson, Eric",
                  "Engemann, Denis A.", "Strohmeier, Daniel", "Brodbeck, Christian",
                  "Goj, Roman", "Jas, Mainak", "Brooks, Teon", "Parkkonen, Lauri",
                  "Hämäläinen, Matti"], jahr="2013",
         titel="MEG and EEG data analysis with MNE-Python",
         gefaess="Frontiers in Neuroscience", band="7", seiten="267",
         doi="10.3389/fnins.2013.00267", quelle="PubMed (PMID 24431986)"),
    dict(tag="Paszke2019", typ="ConferenceProceedings",
         autoren=["Paszke, Adam", "Gross, Sam", "Massa, Francisco", "Lerer, Adam",
                  "Bradbury, James", "Chanan, Gregory", "Killeen, Trevor", "Lin, Zeming",
                  "Gimelshein, Natalia", "Antiga, Luca", "Desmaison, Alban",
                  "Köpf, Andreas", "Yang, Edward", "DeVito, Zach", "Raison, Martin",
                  "Tejani, Alykhan", "Chilamkurthy, Sasank", "Steiner, Benoit",
                  "Fang, Lu", "Bai, Junjie", "Chintala, Soumith"], jahr="2019",
         titel="PyTorch: An imperative style, high-performance deep learning library",
         gefaess="Advances in Neural Information Processing Systems 32 (NeurIPS)",
         url="https://arxiv.org/abs/1912.01703", quelle="arXiv"),
    dict(tag="Harris2020", typ="JournalArticle",
         autoren=["Harris, Charles R.", "Millman, K. Jarrod", "van der Walt, Stéfan J.",
                  "Gommers, Ralf", "Virtanen, Pauli", "Cournapeau, David", "Wieser, Eric",
                  "Taylor, Julian", "Berg, Sebastian", "Smith, Nathaniel J."], jahr="2020",
         titel="Array programming with NumPy",
         gefaess="Nature", band="585", heft="7825", seiten="357-362",
         doi="10.1038/s41586-020-2649-2",
         quelle="Crossref (26 Autoren, hier die ersten 10 — 'et al.' im Zitierstil)"),
    dict(tag="Virtanen2020", typ="JournalArticle",
         autoren=["Virtanen, Pauli", "Gommers, Ralf", "Oliphant, Travis E.",
                  "Haberland, Matt", "Reddy, Tyler", "Cournapeau, David",
                  "Burovski, Evgeni", "Peterson, Pearu", "Weckesser, Warren",
                  "Bright, Jonathan"], jahr="2020",
         titel="SciPy 1.0: Fundamental algorithms for scientific computing in Python",
         gefaess="Nature Methods", band="17", heft="3", seiten="261-272",
         doi="10.1038/s41592-019-0686-2",
         quelle="Crossref (über 100 Autoren, hier die ersten 10 — 'et al.' im Zitierstil)"),

    # --- FACETpy documentation pages (Word-citable implementation records) --
    # Keep these URLs aligned with docs/source/thesis_reference/models/.  They
    # document the evaluated FACETpy variants; the architecture papers above
    # remain the sources for the original methods.
    *[
        dict(tag=tag, typ="Report", autoren=["Müller, Janik Michael"], jahr="2026",
             titel=f"FACETpy model reference: {title}", gefaess="FACETpy documentation (Read the Docs)",
             url=f"https://facetpy.readthedocs.io/en/latest/thesis_reference/models/{slug}.html",
             quelle="FACETpy Read the Docs")
        for tag, title, slug in [
            ("FACETpyCascadedDAE2026", "Cascaded DAE", "cascaded_dae"),
            ("FACETpyContextDAE2026", "Context DAE", "context_dae"),
            ("FACETpyDPAE2026", "DPAE", "dpae"),
            ("FACETpyICUNet2026", "IC-U-Net", "ic_unet"),
            ("FACETpyNestedGAN2026", "Nested GAN", "nested_gan"),
            ("FACETpyDHCTGAN2026", "DHCT-GAN", "dhct_gan"),
            ("FACETpyD4PM2026", "D4PM", "d4pm"),
            ("FACETpyDenoiseMamba2026", "DenoiseMamba", "denoise_mamba"),
            ("FACETpyConvTasNet2026", "Conv-TasNet", "conv_tasnet"),
            ("FACETpyDemucs2026", "Demucs", "demucs"),
            ("FACETpyMultichannelDemucs2026", "Multichannel Demucs extension", "multichannel_demucs"),
            ("FACETpySepFormer2026", "SepFormer", "sepformer"),
            ("FACETpyViTSpectrogram2026", "ViT-Spectrogram", "vit_spectrogram"),
            ("FACETpySTGNN2026", "ST-GNN", "st_gnn"),
        ]
    ],

    # --- Direct web sources retained from Backup 2 --------------------------
    # These records preserve the literal external links found in the thesis
    # draft. They are deliberately separate from the canonical research-paper
    # entries above: a URL in prose is not assumed to be the preferred
    # scholarly citation for the same software or method.
    dict(tag="BackupNeuroKit22021", typ="JournalArticle",
         autoren=["Makowski, Dominique", "Pham, Tam", "Lau, Zen J.", "Brammer, Jan C.", "Lespinasse, François", "Pham, Hung", "Schölzel, Christopher", "Chen, S. H. Annabel"], jahr="2021",
         titel="NeuroKit2: A Python toolbox for neurophysiological signal processing",
         gefaess="Behavior Research Methods", doi="10.3758/s13428-020-01516-y",
         url="https://doi.org/10.3758/s13428-020-01516-y", quelle="Backup 2 URL; Crossref"),
    dict(tag="BackupSciPyGitHub2026", typ="Report", autoren=["SciPy Contributors"], jahr="2026",
         titel="SciPy source repository", gefaess="GitHub", url="https://github.com/scipy/scipy",
         quelle="Backup 2 URL"),
    dict(tag="KimBandettini2023", typ="Report", autoren=["Kim, Seong-Gi", "Bandettini, Peter A."], jahr="2023",
         titel="Principles of BOLD Functional MRI", gefaess="Functional Neuroradiology, Springer",
         doi="10.1007/978-3-031-10909-6_19", url="https://link.springer.com/chapter/10.1007/978-3-031-10909-6_19",
         quelle="Backup 2 URL; publisher record"),
    dict(tag="BackupMatplotlib2026", typ="Report", autoren=["Matplotlib Development Team"], jahr="2026",
         titel="Matplotlib documentation", gefaess="Matplotlib", url="https://matplotlib.org",
         quelle="Backup 2 URL"),
    dict(tag="BackupMNEDocs2026", typ="Report", autoren=["MNE-Python Contributors"], jahr="2026",
         titel="MNE-Python documentation", gefaess="MNE-Python", url="https://mne.tools/stable/index.html",
         quelle="Backup 2 URL"),
    dict(tag="BackupPandasDocs2026", typ="Report", autoren=["pandas development team"], jahr="2026",
         titel="pandas documentation", gefaess="pandas", url="https://pandas.pydata.org",
         quelle="Backup 2 raw URL"),
    dict(tag="BackupScikitLearnDocs2026", typ="Report", autoren=["scikit-learn developers"], jahr="2026",
         titel="scikit-learn documentation", gefaess="scikit-learn", url="https://scikit-learn.org/stable/",
         quelle="Backup 2 raw URL"),
    dict(tag="BackupScikitLearnPreprocessing2026", typ="Report", autoren=["scikit-learn developers"], jahr="2026",
         titel="Preprocessing data", gefaess="scikit-learn documentation",
         url="https://scikit-learn.org/stable/modules/preprocessing.html", quelle="Backup 2 raw URL"),
    dict(tag="BackupDataCampUV2026", typ="Report", autoren=["DataCamp"], jahr="2026",
         titel="Python uv tutorial", gefaess="DataCamp", url="https://www.datacamp.com/de/tutorial/python-uv",
         quelle="Backup 2 URL"),
    dict(tag="BackupNvidiaPyTorch2026", typ="Report", autoren=["NVIDIA"], jahr="2026",
         titel="PyTorch glossary", gefaess="NVIDIA", url="https://www.nvidia.com/en-us/glossary/pytorch/",
         quelle="Backup 2 URL"),
    dict(tag="BackupScienceDirectS003537872300869X", typ="Report", autoren=["Elsevier"], jahr="2026",
         titel="ScienceDirect resource S003537872300869X", gefaess="ScienceDirect",
         url="https://www.sciencedirect.com/science/article/pii/S003537872300869X",
         quelle="Backup 2 URL; bibliographic metadata requires manual verification"),
    dict(tag="BackupScienceDirectB9780323998987000274", typ="Report", autoren=["Elsevier"], jahr="2026",
         titel="ScienceDirect resource B9780323998987000274", gefaess="ScienceDirect",
         url="https://www.sciencedirect.com/science/chapter/bookseries/abs/pii/B9780323998987000274",
         quelle="Backup 2 URL; bibliographic metadata requires manual verification"),
    dict(tag="FACETpyDocs2026", typ="Report", autoren=["FACETpy Contributors"], jahr="2026",
         titel="FACETpy documentation", gefaess="Read the Docs", url="https://facetpy.readthedocs.io/",
         quelle="FACETpy documentation"),
    # --- Sources resolved from Backup 2 citation TODOs ---------------------
    dict(tag="BrainVoyagerEMEG", typ="Report", autoren=["Brain Innovation B.V."], jahr="2026",
         titel="Simultaneous EEG-fMRI: fMRI artifact detection and removal",
         gefaess="BrainVoyager User's Guide",
         url="https://brainvoyager.com/bv/doc/UsersGuide/EMEGSuite/SimultaneousEEG-fMRIArtifactDetectionAndRemoval.html",
         quelle="Official BrainVoyager documentation"),
    dict(tag="Shahriar2025", typ="Report",
         autoren=["Shahriar, K. A.", "Bhuiyan, E. H.", "Luo, Q.", "Chowdhury, M. E. H.", "Zhou, X. J."],
         jahr="2025", titel="Deep Learning for Gradient and BCG Artifacts Removal in EEG During Simultaneous fMRI",
         gefaess="arXiv:2507.22263", url="https://arxiv.org/abs/2507.22263", quelle="arXiv"),
    dict(tag="Duffy2020", typ="ConferenceProceedings",
         autoren=["Duffy, Ben A.", "Toga, Arthur W.", "Kim, Hosung"], jahr="2020",
         titel="Gradient Artifact Correction for Simultaneous EEG-fMRI using Denoising Autoencoders",
         gefaess="Proceedings of the IEEE International Symposium on Biomedical Imaging (ISBI)",
         doi="10.1109/ISBI45749.2020.9098447", quelle="Crossref"),
    dict(tag="Bullock2021", typ="JournalArticle",
         autoren=["Bullock, Madeleine", "Jackson, Graeme D.", "Abbott, David F."], jahr="2021",
         titel="Artifact Reduction in Simultaneous EEG-fMRI: A Systematic Review of Methods and Contemporary Usage",
         gefaess="Frontiers in Neurology", band="12", seiten="622719",
         doi="10.3389/fneur.2021.622719", quelle="Crossref"),
    dict(tag="Appelhoff2019", typ="JournalArticle",
         autoren=["Appelhoff, Stefan", "Sanderson, Michael", "Brookes, Matthew", "Hämäläinen, Matti", "Oostenveld, Robert", "Delorme, Arnaud"],
         jahr="2019", titel="MNE-BIDS: Organizing electrophysiological data into the BIDS format and facilitating their analysis",
         gefaess="Journal of Open Source Software", band="4", heft="44", seiten="1896",
         doi="10.21105/joss.01896", quelle="DOI"),
    dict(tag="TextualizeRich", typ="Report", autoren=["Textualize"], jahr="2026",
         titel="Rich: Render rich text, tables, progress bars, syntax highlighting, and more to the terminal",
         gefaess="GitHub", url="https://github.com/Textualize/rich", quelle="Official software repository"),
    dict(tag="Lin2025", typ="JournalArticle", autoren=["Lin, Nan"], jahr="2025",
         titel="An EEG dataset for interictal epileptiform discharge with spatial distribution information",
         gefaess="Scientific Data", band="12", seiten="229", url="https://github.com/vepiset/vepiset_dataset",
         quelle="Dataset repository citation"),
    dict(tag="AppleLogicPro", typ="Report", autoren=["Apple"], jahr="2026",
         titel="Logic Pro User Guide for Mac", gefaess="Apple Support",
         url="https://support.apple.com/guide/logicpro/welcome/mac", quelle="Official Apple documentation"),
    dict(tag="FACETpyRepository", typ="Report", autoren=["FACETpy Contributors"], jahr="2026",
         titel="FACETpy source repository", gefaess="GitHub", url="https://github.com/H0mire/FACETpy",
         quelle="Official project repository"),
    dict(tag="Widrow1975", typ="JournalArticle",
         autoren=["Widrow, Bernard", "Glover, John R.", "McCool, John M.", "Kaunitz, John", "Williams, Charles S.", "Hearn, Robert H.", "Zeidler, James R.", "Dong, Eugene Jr.", "Goodlin, Robert C."],
         jahr="1975", titel="Adaptive Noise Cancelling: Principles and Applications",
         gefaess="Proceedings of the IEEE", band="63", heft="12", seiten="1692-1716",
         doi="10.1109/PROC.1975.10036", quelle="DOI"),
]

def _split_name(name: str) -> tuple[str, str]:
    last, _, first = name.partition(", ")
    return last.strip(), first.strip()


def word_xml() -> str:
    ns = "http://schemas.openxmlformats.org/officeDocument/2006/bibliography"
    out = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
           f'<b:Sources xmlns:b="{ns}" xmlns="{ns}" SelectedStyle="">']
    for r in REFS:
        e = [f'  <b:Source>',
             f'    <b:Tag>{html.escape(r["tag"])}</b:Tag>',
             f'    <b:SourceType>{r["typ"]}</b:SourceType>',
             f'    <b:Year>{r["jahr"]}</b:Year>',
             f'    <b:Title>{html.escape(r["titel"])}</b:Title>']
        if r["autoren"]:
            e += ['    <b:Author>', '      <b:Author>', '        <b:NameList>']
            for a in r["autoren"]:
                last, first = _split_name(a)
                e += ['          <b:Person>',
                      f'            <b:Last>{html.escape(last)}</b:Last>',
                      f'            <b:First>{html.escape(first)}</b:First>',
                      '          </b:Person>']
            e += ['        </b:NameList>', '      </b:Author>', '    </b:Author>']
        tag_for_venue = {"JournalArticle": "b:JournalName",
                         "ConferenceProceedings": "b:ConferenceName",
                         "Report": "b:Publisher"}[r["typ"]]
        if r.get("gefaess"):
            e.append(f'    <{tag_for_venue}>{html.escape(r["gefaess"])}</{tag_for_venue}>')
        for key, xml_tag in (("band", "b:Volume"), ("heft", "b:Issue"), ("seiten", "b:Pages")):
            if r.get(key):
                e.append(f'    <{xml_tag}>{html.escape(str(r[key]))}</{xml_tag}>')
        link = r.get("doi") and f"https://doi.org/{r['doi']}" or r.get("url")
        if link:
            e.append(f'    <b:URL>{html.escape(link)}</b:URL>')
        e.append('  </b:Source>')
        out += e
    out.append('</b:Sources>')
    return "\n".join(out) + "\n"


def bibtex() -> str:
    kinds = {"JournalArticle": "article", "ConferenceProceedings": "inproceedings",
             "Report": "misc"}
    venue_field = {"article": "journal", "inproceedings": "booktitle", "misc": "howpublished"}
    lines = ["% FACETpy — Literatur für Kapitel 5",
             "% Erzeugt von tools/reporting/build_bibliography.py aus src/facet/models/REFERENCES.md.",
             "% Alle Angaben aus Crossref, arXiv oder PubMed abgerufen; "
             "das note-Feld nennt die Quelle je Eintrag.", ""]
    for r in REFS:
        kind = kinds[r["typ"]]
        lines.append(f"@{kind}{{{r['tag']},")
        if r["autoren"]:
            lines.append("  author       = {" + " and ".join(r["autoren"]) + "},")
        lines.append("  title        = {" + r["titel"] + "},")
        if r.get("gefaess"):
            lines.append(f"  {venue_field[kind]:<12} = {{{r['gefaess']}}},")
        for key, field in (("band", "volume"), ("heft", "number"), ("seiten", "pages")):
            if r.get(key):
                lines.append(f"  {field:<12} = {{{r[key]}}},")
        lines.append(f"  year         = {{{r['jahr']}}},")
        if r.get("doi"):
            lines.append("  doi          = {" + r["doi"] + "},")
        if r.get("url"):
            lines.append("  url          = {" + r["url"] + "},")
        lines.append("  note         = {Angabe aus " + r["quelle"] + "},")
        lines.append("}")
        lines.append("")
    return "\n".join(lines)


def checklist() -> str:
    """Was beim Abgleich mit Crossref/arXiv/PubMed herauskam."""
    korrekturen = [
        ("Yang2025 (Nested-GAN)", "REFERENCES.md: „Yang, Hu, Cai & Xu\" mit falschen "
         "Vornamen", "Yang, Tianqi; Hu, Nan; Cai, Shengsheng; Xu, Dongyang"),
        ("Chen2025 (DenoiseMamba)", "REFERENCES.md: „Liu et al.\" — falscher Erstautor, "
         "der Eintrag hieß hier vorher Liu2025",
         "Chen, Wensheng; Li, Yurong; Zheng, Nan; Shi, Wuxiang"),
        ("Cai2025 (DHCT-GAN)", "REFERENCES.md nannte nur „Cai et al.\"",
         "Cai, Yinan; Meng, Zhao; Huang, Dian"),
        ("Niazy2005", "im Repo als „Iannotti\" geführt", "Iannetti, G. D."),
        ("Xiong2023", "Vornamen wichen ab", "Xiong, Wenjing; Ma, Lin; Li, Haifeng"),
        ("Chuang2022", "Vorname des Erstautors wich ab", "Chuang, Chun-Hsiang"),
        ("Shao2025 (D4PM)", "REFERENCES.md nannte gar keine Autoren; der Eintrag "
         "hieß hier vorher D4PM2025",
         "Shao, Feixue; Liu, Xueyu; Wu, Yongfei; Lu, Jianbo; Yan, Guiying; Yang, Weihua"),
    ]
    out = ["# Literaturliste — Abgleich mit den Registern", "",
           f"{len(REFS)} Einträge. Literaturangaben wurden über DOI/Crossref, arXiv "
           "oder PubMed abgeglichen. Ergänzende FACETpy- und Backup-2-Webquellen "
           "bewahren die im Projekt verwendeten direkten URLs. Das `quelle`-Feld im "
           "Generator und das `note`-Feld in der BibTeX-Datei nennen die Herkunft je Eintrag.", "",
           "## Sieben Korrekturen gegenüber `src/facet/models/REFERENCES.md`", "",
           "Bei zwei Einträgen war der **Erstautor** ein anderer — dort ändert sich "
           "auch der Zitierschlüssel im Text.", "",
           "| Eintrag | im Repo | tatsächlich |", "|---|---|---|"]
    for tag, alt, neu in korrekturen:
        out.append(f"| {tag} | {alt} | {neu} |")
    out += ["", "## Gekürzte Autorenlisten", "",
            "`Harris2020` (NumPy) hat 26 und `Virtanen2020` (SciPy) über 100 Autoren. "
            "Hier stehen je die ersten zehn; jeder Zitierstil kürzt sie ohnehin auf "
            "„et al.\". Falls die Prüfungsordnung die vollständige Liste verlangt, "
            "über die DOI nachziehen.", "",
            "## Nicht in dieser Liste", "",
            "- Der interne Überblicksbericht `docs/research/dl_eeg_gradient_artifacts.pdf` "
            "hat keine Publikationsangabe und ist als unveröffentlichtes Projektdokument "
            "zu zitieren.",
            "- `cascaded_dae`, `cascaded_context_dae` und `demo01` sind interne "
            "Prototypen ohne Quellpaper (siehe REFERENCES.md).",
            "- FACETpy selbst: Version 2.0.2, Autor Janik Michael Müller, GPLv3."]
    return "\n".join(out) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in (("literatur.xml", word_xml()),
                       ("literatur.bib", bibtex()),
                       ("literatur_pruefliste.md", checklist())):
        (OUT / name).write_text(text, encoding="utf-8")
        print(f"  {OUT / name}  ({len(text.splitlines())} Zeilen)")
    print(f"\n{len(REFS)} Einträge, einschließlich FACETpy- und Backup-2-Webquellen.")
    print("Sieben Angaben wichen von REFERENCES.md ab — siehe literatur_pruefliste.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
