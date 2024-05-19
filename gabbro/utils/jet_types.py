"""Utility functions for jet types."""

jet_types_dict = {
    "QCD": {"label": 0, "tex_label": "$q/g$", "file_prefix": "ZJetsToNuNu_", "color": "C0"},
    "Hbb": {
        "label": 1,
        "tex_label": "$H\\rightarrow b\\bar{b}$",
        "file_prefix": "HToBB_",
        "color": "C1",
    },
    "Hcc": {
        "label": 2,
        "tex_label": "$H\\rightarrow c\\bar{c}$",
        "file_prefix": "HToCC_",
        "color": "C2",
    },
    "Hgg": {
        "label": 3,
        "tex_label": "$H\\rightarrow gg$",
        "file_prefix": "HToGG_",
        "color": "C3",
    },
    "H4q": {
        "label": 4,
        "tex_label": "$H\\rightarrow 4q$",
        "file_prefix": "HToWW4Q_",
        "color": "C4",
    },
    "Hqql": {
        "label": 5,
        "tex_label": "$H\\rightarrow \\ell\\nu qq'$",
        "file_prefix": "HToWW2Q1L_",
        "color": "C5",
    },
    "Zqq": {
        "label": 6,
        "tex_label": "$Z\\rightarrow q\\bar{q}$",
        "file_prefix": "ZToQQ_",
        "color": "C6",
    },
    "Wqq": {
        "label": 7,
        "tex_label": "$W\\rightarrow qq'$",
        "file_prefix": "WToQQ_",
        "color": "C7",
    },
    "Tbqq": {
        "label": 8,
        "tex_label": "$t\\rightarrow bqq'$",
        "file_prefix": "TTBar_",
        "color": "C8",
    },
    "Tbl": {
        "label": 9,
        "tex_label": "$t\\rightarrow b\\ell\\nu$",
        "file_prefix": "TTBarLep_",
        "color": "C9",
    },
}


def get_tex_label_from_numerical_label(label: int) -> str:
    """Return the TeX label of a jet type given its numerical label."""
    for jet_type, jet_type_dict in jet_types_dict.items():
        if jet_type_dict["label"] == label:
            return jet_type_dict["tex_label"]
    raise ValueError(f"Invalid label: {label}")


def get_numerical_label_from_file_prefix(file_prefix: str) -> int:
    """Return the numerical label given the (JetClass) file_prefix (e.g. "ZJetsToNuNu_" -> 0)"""
    for jet_type, jet_type_dict in jet_types_dict.items():
        if jet_type_dict["file_prefix"] == file_prefix:
            return jet_type_dict["label"]
    raise ValueError(f"Invalid file_prefix: {file_prefix}")


def get_jet_type_from_file_prefix(file_prefix: str) -> str:
    """Get the jet type from the file prefix.

    E.g. "ZJetsToNuNu_" -> "QCD"
    """
    for jet_type, jet_type_dict in jet_types_dict.items():
        if jet_type_dict["file_prefix"] == file_prefix:
            return jet_type
    raise ValueError(f"Invalid file_prefix: {file_prefix}")
