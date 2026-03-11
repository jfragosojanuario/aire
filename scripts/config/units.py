import re

# Raw unit alternatives (linear and weight)
LINEAR_UNIT_ALTERNATIVES_RAW = (
    'mm|cm|dm|m|km|'
    'metro|metros|meters|'
    'centimetro|centimetros|centímetro|centímetros|'
    'milimetro|milimetros|milímetro|milímetros|'
    'inch|in|foot|ft|yard|yd'
)
WEIGHT_UNIT_ALTERNATIVES_RAW = 'kg|g|mg|ton|lb|lbs'

# Combined raw units for convenience
ALL_UNITS_RAW = f"{LINEAR_UNIT_ALTERNATIVES_RAW}|{WEIGHT_UNIT_ALTERNATIVES_RAW}"

# Regex patterns for values and units
VALUE_UNIT_PATTERN = re.compile(
    rf'(\d+\.?\d*)\s*({ALL_UNITS_RAW})\b',
    re.IGNORECASE
)

COMPOUND_PATTERN = re.compile(
    rf'(\d+\.?\d*(?:\s*(?:x|X|\*|por|by)\s*\d+\.?\d*)+)\s*({LINEAR_UNIT_ALTERNATIVES_RAW})\b',
    re.IGNORECASE
)

SUM_PATTERN = re.compile(
    rf'(\d+\.?\d*)\s*({LINEAR_UNIT_ALTERNATIVES_RAW})'
    rf'(?:\s+[^\+\n\r]*)?'      # optional material text (no +)
    rf'\s*\+\s*'
    rf'(\d+\.?\d*)\s*({LINEAR_UNIT_ALTERNATIVES_RAW})',
    re.IGNORECASE
)

# Normalization and conversions
UNIT_NORMALIZATION = {
    # metros
    'metro': 'm',
    'metros': 'm',
    'meters': 'm',

    # centimetros
    'centimetro': 'cm',
    'centimetros': 'cm',
    'centímetro': 'cm',
    'centímetros': 'cm',

    # milimetros
    'milimetro': 'mm',
    'milimetros': 'mm',
    'milímetro': 'mm',
    'milímetros': 'mm',
}

UNIT_CONVERSIONS = {
    # Linear → meters
    'mm': (0.001, 'm'),
    'cm': (0.01, 'm'),
    'dm': (0.1, 'm'),
    'm': (1, 'm'),
    'km': (1000, 'm'),
    'inch': (0.0254, 'm'),
    'in': (0.0254, 'm'),
    'foot': (0.3048, 'm'),
    'ft': (0.3048, 'm'),
    'yard': (0.9144, 'm'),
    'yd': (0.9144, 'm'),

    # Weight → kg
    'kg': (1, 'kg'),
    'g': (0.001, 'kg'),
    'mg': (1e-6, 'kg'),
    'ton': (1000, 'kg'),
    'lb': (0.453592, 'kg'),
    'lbs': (0.453592, 'kg'),
}

