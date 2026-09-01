"""
Maps raw RDD2022 class codes (as produced by the YOLO model) to
human-friendly labels for display to the end user.

Source: RDD2022 (Road Damage Dataset 2022 / CRDDC'2022) class legend,
cross-checked against the official data article and sekilab/RoadDamageDetector.
D0w0 is not defined in the official RDD2022 papers; it appears only as a
rare / often-dropped class in derivative work, so it is labeled generically.
"""

DAMAGE_LABELS = {
    "D00": "Longitudinal Crack",
    "D01": "Longitudinal Crack (Joint)",
    "D0w0": "Miscellaneous / Unclassified",
    "D10": "Transverse Crack",
    "D11": "Transverse Crack (Joint)",
    "D20": "Alligator Crack",
    "D40": "Pothole",
    "D43": "Crosswalk Blur",
    "D44": "White Line Blur",
    "D50": "Utility Hole Cover",
}


def get_label(class_code: str) -> str:
    """Return the friendly label for a class code, falling back to the
    raw code itself if it's ever unrecognized (e.g. model retrained later)."""
    return DAMAGE_LABELS.get(class_code, class_code)