import re

MATERIAL_SPEC_REGEX = {

    # STEEL
    "steel": re.compile(
        r'\b('
        r'S\d{3,4}'                                       # S235, S355
        r'|grade\s*S\d{3,4}'                               # grade S355
        r'|S\d{3,4}\s*grade'                               # S355 grade
        r'|A\d{3}'                                         # A400, A500
        r'|B\d{3}[A-Z]?'                                   # B500, B500S
        r'|EN\s*10025(?:-\d{1,2})?'                         # EN 10025-2
        r'|EN\s*10210(?:-\d{1,2})?'                         # EN 10210
        r'|EN\s*10219(?:-\d{1,2})?'                         # EN 10219
        r'|ASTM\s*A\d{3,4}'                                 # ASTM A36
        r'|(?:HEA|HEB|HEM|IPE|IPN|UPN|UB|UC|HL|HD)\s*\d{2,4}'  # HEA200
        r'|(?:RHS|SHS|CHS|HSS)\s*\d+(?:x\d+)*(?:x\d+(\.\d+)?)?' # RHS 200x100x6
        r')\b',
        re.IGNORECASE
    ),

    # CONCRETE
    "concrete": re.compile(
        r'\b('
        r'C\d{2,3}\/\d{2,3}'                                # C30/37
        r'|LC\d{2,3}\/\d{2,3}'                               # LC30/33 (lightweight)
        r'|B\d{2,3}'                                         # B25 (old PT class)
        r'|fck\s*\d{2,3}'                                    # fck 30
        r'|EN\s*206'                                         # EN 206
        r'|XC\d'                                             # Exposure class XC2
        r'|XS\d'
        r'|XD\d'
        r'|XF\d'
        r'|XA\d'
        r')\b',
        re.IGNORECASE
    ),

    # BRICK / MASONRY
    "brick": re.compile(
        r'\b('
        r'M\d{2}'                                            # M10 mortar class
        r'|EN\s*771(?:-\d)?'                                  # EN 771-1
        r'|clay\s*brick'
        r'|face\s*brick'
        r')\b',
        re.IGNORECASE
    ),

    # WOOD / TIMBER
    "wood": re.compile(
        r'\b('
        r'C\d{2}'                                             # C24 timber class
        r'|GL\d{2}[a-z]?'                                     # GL24h
        r'|D\d{2}'                                            # D40 hardwood
        r'|EN\s*338'                                          # Timber strength class
        r'|EN\s*14080'                                        # Glulam
        r')\b',
        re.IGNORECASE
    ),

    # ALUMINUM
    "aluminum": re.compile(
        r'\b('
        r'EN\s*573'
        r'|EN\s*755'
        r'|EN\s*AW-\d{4}'                                     # EN AW-6060
        r'|6060'
        r'|6061'
        r'|6082'
        r')\b',
        re.IGNORECASE
    ),

    # INSULATION
    "insulation": re.compile(
        r'\b('
        r'EPS\s*\d*'                                          # EPS 100
        r'|XPS\s*\d*'
        r'|MW\s*\d*'                                           # Mineral wool
        r'|λ\s*=\s*0\.\d+'                                     # lambda value
        r')\b',
        re.IGNORECASE
    )
}

