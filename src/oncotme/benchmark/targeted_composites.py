"""Pre-specified TME composites для классов таргетной терапии при РМЖ.

Каждый composite pre-specified на основе published biology конкретного
drug class. Направление каждого компонента указано явно (+/-) по mechanism,
а не выбрано по данным → one test per class, no post-hoc picking.

Формат: name → list[(signature_name, weight, rationale)].

Scope:
- Anti-HER2 (trastuzumab, T-DM1+pertuzumab): ADCC biology
- PARPi + chemo (ABT-888/veliparib + carboplatin): STING/cGAS DDR
- Anti-angiopoietin (AMG-386): vascular + macrophage
- Anti-IGF-1R (Ganitumab): IGF stromal crosstalk
- pan-HER TKI (Neratinib): HER-family signaling
- AKT inhibitor (MK-2206): PI3K/AKT + immune suppression

ICI omitted — уже покрыто separate pembrolizumab composite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Component:
    name: str
    sign: int  # +1 benefit marker (high → more response); -1 resistance marker
    rationale: str


@dataclass
class TargetedComposite:
    class_name: str
    arm_labels: List[str]
    biology: str
    citations: List[str]
    components: List[Component]

    def formula(self) -> str:
        parts = []
        for c in self.components:
            prefix = "+" if c.sign > 0 else "-"
            parts.append(f"{prefix} z({c.name})")
        return " ".join(parts).lstrip("+ ").strip()


# -------- Anti-HER2 (trastuzumab family) ----------------------------------
# ADCC biology: NK-cells recognize antibody-coated HER2+ cells via FcγR,
# mediated through HLA/APM context; stroma (CAF) and TGFβ suppress NK function
# and Trastuzumab penetration.
#
# Key references:
# - Stagg 2011 PNAS — trastuzumab + anti-PD1 synergy, ADCC-dependent
# - Triulzi 2015 Oncotarget — stromal immune escape in HER2+ breast cancer
# - Shi 2015 JCI — NK cell depletion abolishes trastuzumab efficacy
# - Kurozumi 2019 BCR — TILs predict anti-HER2 response
ANTI_HER2 = TargetedComposite(
    class_name="anti-HER2",
    arm_labels=[
        "Paclitaxel + Trastuzumab",
        "Paclitaxel + Pertuzumab + Trastuzumab",
        "Paclitaxel + AMG 386 + Trastuzumab",
        "Paclitaxel + MK-2206 + Trastuzumab",
        "T-DM1 + Pertuzumab",
    ],
    biology="ADCC — NK-cell effector + antigen presentation, limited by stroma/TGFβ",
    citations=[
        "Stagg PNAS 2011",
        "Triulzi Oncotarget 2015",
        "Shi JCI 2015",
        "Kurozumi BCR 2019",
    ],
    components=[
        Component("NK_cell", +1, "ADCC effector cells"),
        Component("HLA_I_score", +1, "antigen presentation context"),
        Component("B_cell", +1, "immune-permissive microenvironment"),
        Component("TLS_signature", +1, "mature immune infiltrate"),
        Component("CAF_proxy", -1, "stromal exclusion of ADCC effectors"),
        Component("TGFb_activity", -1, "NK cell suppression + drug penetration block"),
    ],
)


# -------- PARPi + chemo (ABT-888/veliparib + carboplatin) -----------------
# PARP inhibition + carboplatin create DNA damage → cytosolic dsDNA → cGAS/STING
# → type I IFN. Benefit predicted by tumours ALREADY primed for IFN-response,
# high proliferation (HRD-driven), and low stromal barrier to drug penetration.
#
# Key references:
# - Pantelidou 2019 Cancer Discov — olaparib STING activation in BRCA-null TNBC
# - Sen 2019 Cancer Cell — cGAS-STING PARP inhibitor immune response
# - Post 2022 Clin Cancer Res — proliferative index predicts PARPi+platinum in TNBC
# - Loibl 2018 — GeparOLA PARPi neoadjuvant proliferation link
PARPI = TargetedComposite(
    class_name="PARPi+platinum",
    arm_labels=["Paclitaxel + ABT 888 + Carboplatin"],
    biology="DNA damage → STING/IFN-I, proliferation-dependent",
    citations=[
        "Pantelidou Cancer Discov 2019",
        "Sen Cancer Cell 2019",
        "Post CCR 2022",
        "Loibl GeparOLA 2018",
    ],
    components=[
        Component("IFN_gamma", +1, "STING-driven IFN axis already primed"),
        Component("cytolytic_score", +1, "immune-active milieu"),
        Component("CXCL1_MDSC_axis", +1, "DNA damage induces inflammation"),
        Component("CAF_proxy", -1, "stromal barrier to drug access"),
    ],
)


# -------- Anti-angiopoietin (AMG-386) -------------------------------------
# AMG-386 blocks Ang1/Ang2 → vascular normalization + reduced Tie2+ macrophage
# support. Benefit predicted in tumours dependent on angiogenesis, rich in
# Tie2+/M2 macrophages, hypoxic. Mature TLS-associated vessels less affected.
#
# Key references:
# - De Palma 2012 Cell — Tie2 macrophages and angiopoietin
# - Rigamonti 2014 Nat Rev Clin Oncol — vascular normalization + immune
# - Murdoch 2004 Blood — M2 macrophages and angiogenesis
ANTI_ANGIO = TargetedComposite(
    class_name="anti-Angiopoietin",
    arm_labels=["Paclitaxel + AMG 386", "Paclitaxel + AMG-386"],
    biology="vascular normalization + Tie2+/M2 macrophage dependency",
    citations=[
        "De Palma Cell 2012",
        "Rigamonti Nat Rev Clin Oncol 2014",
        "Murdoch Blood 2004",
    ],
    components=[
        Component("angiogenesis", +1, "drug target pathway activity"),
        Component("M2_macrophage", +1, "Tie2+ macrophage population"),
        Component("hypoxia_score", +1, "angiogenesis-driven hypoxia"),
        Component("TLS_signature", -1, "mature vasculature less angiopoietin-dep"),
    ],
)


# -------- Anti-IGF-1R (Ganitumab) -----------------------------------------
# Ganitumab blocks IGF-1R, which is expressed in both tumour and stromal
# fibroblasts. Stromal IGF-1 paracrine loop from CAFs + TGFβ signaling
# reduces IGF-1R blockade efficacy. Benefit predicted in low-stroma tumours
# where proliferation is tumour-autonomous.
#
# Key references:
# - Cao 2013 Cancer Res — IGF-1 stroma crosstalk
# - Wang 2015 Cancer Discov — CAF IGF-1 in chemotherapy resistance
# - Fu 2019 Cell Rep — IGF-1R and TGFβ pathway crosstalk
ANTI_IGF = TargetedComposite(
    class_name="anti-IGF-1R",
    arm_labels=["Paclitaxel + Ganitumab"],
    biology="stromal IGF-1 paracrine loop blunts IGF-1R blockade",
    citations=[
        "Cao Cancer Res 2013",
        "Wang Cancer Discov 2015",
        "Fu Cell Rep 2019",
    ],
    components=[
        Component("CAF_proxy", -1, "CAF-derived IGF-1 paracrine rescue"),
        Component("TGFb_activity", -1, "IGF-1R / TGFβ pathway crosstalk"),
        Component("immune_exclusion", -1, "stromal barriers"),
    ],
)


# -------- pan-HER TKI (Neratinib) -----------------------------------------
# Irreversible pan-HER TKI; effective in low-stromal, angiogenically-active
# tumours with low TGFβ (which drives resistance via EMT).
#
# Key references:
# - Canonici 2013 Oncotarget — neratinib + trastuzumab synergy
# - Chandarlapaty 2015 Cancer Discov — PI3K/AKT escape from HER TKIs
# - Shibue 2019 NEJM — TGFβ-driven TKI resistance in HER2+
PAN_HER = TargetedComposite(
    class_name="pan-HER TKI",
    arm_labels=["Paclitaxel + Neratinib"],
    biology="pan-HER inhibition; TGFβ-EMT drives resistance",
    citations=[
        "Canonici Oncotarget 2013",
        "Chandarlapaty Cancer Discov 2015",
        "Shibue NEJM 2019",
    ],
    components=[
        Component("angiogenesis", +1, "tumour-autonomous growth axis"),
        Component("TGFb_activity", -1, "TGFβ-EMT drug resistance"),
        Component("CAF_proxy", -1, "stromal barrier"),
    ],
)


# -------- AKT inhibitor (MK-2206) -----------------------------------------
# AKT blockade effective in PI3K/AKT-dependent tumours. Tumours with high
# immune exclusion / TGFβ have M2-driven pro-survival signaling that
# bypasses AKT inhibition.
#
# Key references:
# - Ma 2010 Mol Cancer Ther — MK-2206 single-agent activity
# - Kechagioglou 2014 Anticancer Res — PTEN loss vs MK-2206 response
AKT = TargetedComposite(
    class_name="AKT inhibitor",
    arm_labels=["Paclitaxel + MK-2206"],
    biology="PI3K/AKT axis; bypassed via stromal/immune-suppressive signaling",
    citations=[
        "Ma Mol Cancer Ther 2010",
        "Kechagioglou Anticancer Res 2014",
    ],
    components=[
        Component("immune_exclusion", -1, "alternative pro-survival pathways"),
        Component("TGFb_activity", -1, "TGFβ-driven resistance"),
        Component("M2_macrophage", -1, "M2 pro-survival signaling"),
    ],
)


ALL_COMPOSITES: List[TargetedComposite] = [
    ANTI_HER2,
    PARPI,
    ANTI_ANGIO,
    ANTI_IGF,
    PAN_HER,
    AKT,
]
