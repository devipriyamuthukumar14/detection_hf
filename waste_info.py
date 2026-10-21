"""
waste_info.py
Maps each detected CLASS to a PARENT CATEGORY and gives general disposal guidance.
Edit the text here; no other file needs to change.
"""

# ---- The three parent categories required by the project ----
HAZARDOUS = "Hazardous"
BIODEGRADABLE = "Biodegradable"
NON_BIODEGRADABLE = "Non-biodegradable / Recyclable"

ALL_PARENTS = [HAZARDOUS, BIODEGRADABLE, NON_BIODEGRADABLE]

# ---- Class -> parent category (the "clearly defined class mapping") ----
PARENT_CATEGORY = {
    # Hazardous
    "battery": HAZARDOUS,
    "aerosol_can": HAZARDOUS,
    "ewaste_device": HAZARDOUS,             # optional extra dataset
    "medical_sanitary": HAZARDOUS,          # optional extra dataset
    "paint_chemical_container": HAZARDOUS,  # optional extra dataset
    # Biodegradable
    "food_waste": BIODEGRADABLE,
    # Non-biodegradable / recyclable
    "plastic_bottle": NON_BIODEGRADABLE,
    "plastic_bag_wrapper": NON_BIODEGRADABLE,
    "plastic_cup_straw": NON_BIODEGRADABLE,
    "plastic_container": NON_BIODEGRADABLE,
    "paper_cardboard": NON_BIODEGRADABLE,
    "glass_bottle": NON_BIODEGRADABLE,
    "metal_can": NON_BIODEGRADABLE,
    "bottle_cap_lid": NON_BIODEGRADABLE,
    "cigarette_butt": NON_BIODEGRADABLE,
    "plastic_other": NON_BIODEGRADABLE,
    "metal_scrap_foil": NON_BIODEGRADABLE,
    "other_litter": NON_BIODEGRADABLE,      # mixed bucket (shoes, rope, unlabeled litter)
}

# ---- Box colours for drawing (OpenCV uses BGR order: Blue, Green, Red) ----
PARENT_COLOR_BGR = {
    HAZARDOUS: (0, 0, 255),            # red
    BIODEGRADABLE: (0, 170, 0),        # green
    NON_BIODEGRADABLE: (255, 128, 0),  # blue
}

# ---- Shown whenever at least one hazardous item is detected ----
HAZARD_DISCLAIMER = (
    "These are model predictions made from a photo only. The system cannot tell what a "
    "container holds, whether an item is damaged or contaminated, or whether it is truly "
    "hazardous. Please contact your local authorised waste-disposal facility or follow your "
    "municipality's waste-management guidance."
)

# ---- General, class-specific guidance (rules differ from place to place) ----
GUIDANCE = {
    "battery": "Batteries can leak or overheat if damaged. Keep them separate from ordinary "
               "household waste and take them to an authorised battery or e-waste collection point.",
    "aerosol_can": "Spray cans may still hold pressurised or flammable contents, even if they seem "
                   "empty. Do not puncture or burn them. Ask your local waste authority about "
                   "hazardous-waste drop-off.",
    "ewaste_device": "Electronic devices contain materials that should not go in normal bins. "
                     "Use an authorised e-waste collection or recycling centre.",
    "medical_sanitary": "Medical or sanitary items need special handling. Do not mix them with "
                        "household recyclables; ask a local authorised facility how to dispose of them.",
    "paint_chemical_container": "Containers that held paint or chemicals may contain residue. "
                                "Do not reuse or mix them with normal waste; consult a local authorised "
                                "hazardous-waste facility.",
    "food_waste": "Usually suitable for composting or the wet/organic waste bin. Keep it away from "
                  "dry recyclables so they stay clean.",
    "plastic_bottle": "Empty it and put it with dry recyclable plastics if your local system accepts "
                      "plastic bottles.",
    "plastic_bag_wrapper": "Thin plastic is often not accepted in normal recycling. Reduce use, reuse "
                           "bags, and check your local rules for plastic-film collection.",
    "plastic_cup_straw": "Single-use plastic items are usually hard to recycle. Prefer reusable cups, "
                         "and dispose of these in the dry-waste bin.",
    "plastic_container": "Empty and rinse the container, then check whether your local recycling accepts "
                         "this plastic type.",
    "paper_cardboard": "Keep it dry and clean and put it with paper/cardboard recycling. Greasy or wet "
                       "paper may not be recyclable.",
    "glass_bottle": "Put glass in a glass-recycling collection if available. Handle with care to avoid cuts.",
    "metal_can": "Empty, and place with metal recyclables. Cans are widely recyclable.",
    "bottle_cap_lid": "Check local rules, as some places accept caps and lids with the bottle and others "
                      "do not. Small items can get lost in sorting.",
    "cigarette_butt": "Cigarette butts contain plastic filters and are not biodegradable. Put them in a "
                      "bin and never on the ground or in drains.",
    "plastic_other": "Mixed or unidentified plastic, including polystyrene pieces and six-pack rings. "
                     "Usually dry-waste bin; cut six-pack rings before disposal so wildlife is not trapped.",
    "metal_scrap_foil": "Foil, pull-tabs and scrap metal: collect separately with metal recyclables if your "
                        "local system accepts them. Sharp pieces should be handled with care.",
    "other_litter": "Mixed items (for example shoes, rope, or unidentified litter). Put them in the dry-waste "
                    "bin; reusable or textile items may be donated or taken to textile recycling.",
}

DEFAULT_GUIDANCE = "No specific guidance available. Please follow your local waste-management rules."


def get_parent(class_name):
    """Return the parent category for a class (or 'Unknown' if not in the table)."""
    return PARENT_CATEGORY.get(class_name, "Unknown")


def get_guidance(class_name):
    """Return the general disposal guidance text for a class."""
    return GUIDANCE.get(class_name, DEFAULT_GUIDANCE)
