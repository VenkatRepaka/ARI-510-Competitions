"""Hand-drawn boxes (fractions x0, y0, x1, y1 of the EXIF-corrected image) around the object that decides the label
in each original dishes training photo. dirty = the food/residue/waiting dishes in Incomplete photos,
clean = the clean / racked / put-away dishes in Complete photos. Drawn by eye from gridded contact sheets."""
DIRTY = {
    2: (0.15, 0.60, 0.45, 0.78),   # food plate on counter left of sink
    8: (0.08, 0.50, 0.82, 0.85),   # cups with coffee residue on tray
    15: (0.10, 0.05, 0.95, 0.95),  # plate with food scraps
    17: (0.18, 0.38, 0.72, 0.66),  # used plate + cutlery in sink
    20: (0.05, 0.45, 0.75, 0.95),  # plate with leftover food
    25: (0.15, 0.30, 1.00, 1.00),  # stained cups
    28: (0.00, 0.35, 0.60, 0.75),  # cluttered pots (dark)
    31: (0.18, 0.30, 0.70, 0.75),  # bowl heaped with dishes
    43: (0.20, 0.30, 0.85, 0.85),  # used cups and saucer
    46: (0.55, 0.30, 0.92, 0.62),  # heap of dishes by sink
    47: (0.40, 0.72, 1.00, 1.00),  # ketchup plates
    48: (0.05, 0.75, 0.75, 1.00),  # ketchup plate
    58: (0.00, 0.40, 1.00, 1.00),  # greasy pan / spoons
    60: (0.40, 0.74, 0.90, 0.95),  # plate in sink + pan left out
    66: (0.00, 0.15, 1.00, 0.75),  # dirty pan and colander in sink
    88: (0.10, 0.20, 0.85, 0.95),  # stained cup and saucers
}
CLEAN = {
    1: (0.00, 0.15, 1.00, 1.00),   # plates in rack
    7: (0.05, 0.35, 1.00, 0.90),   # mugs / plates on rack
    12: (0.10, 0.30, 0.90, 0.75),  # empty clean sink
    22: (0.15, 0.05, 0.80, 0.85),  # knives in block
    38: (0.00, 0.35, 1.00, 1.00),  # loaded dishwasher
    40: (0.10, 0.20, 1.00, 0.60),  # glasses drying upside down
    44: (0.20, 0.45, 0.90, 0.80),  # dishes on cupboard racks
    49: (0.00, 0.20, 1.00, 1.00),  # rack + cutlery drying
    50: (0.00, 0.30, 1.00, 1.00),  # rack + clean plate
    52: (0.00, 0.30, 0.80, 1.00),  # plates into dishwasher
    64: (0.10, 0.20, 1.00, 0.85),  # rack with gloves
    67: (0.10, 0.45, 1.00, 0.85),  # plates on shelf rack
    68: (0.00, 0.35, 1.00, 0.95),  # wall drying rack
    81: (0.05, 0.25, 0.90, 0.90),  # drying rack
    96: (0.05, 0.05, 0.95, 1.00),  # drying basket
    99: (0.40, 0.42, 1.00, 0.65),  # drying rack by window
    101: (0.30, 0.40, 0.60, 0.55), # bowls on counter (labelled Complete)
}
