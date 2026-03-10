# -*- coding: utf-8 -*-

import os
import json
import re
import math
from rapidfuzz import fuzz, process
from dotenv import load_dotenv
import time


# Load environment variables and retrieve the API key
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Prefer new OpenAI SDK (v1) if available; otherwise fall back to legacy (v0.28.x)
try:
  from openai import OpenAI  # SDK v1.x
  _HAS_OPENAI_V1 = True
  
except Exception as e:
  OpenAI = None
  _HAS_OPENAI_V1 = False

if _HAS_OPENAI_V1:
  # v1 client reads API key from env by default
  client = OpenAI()
else:
  # Legacy SDK
  import openai as openai_legacy  # type: ignore
  openai_legacy.api_key = OPENAI_API_KEY




from tests.test_inputs import TEST_INPUTS as test_inputs

def llm_process_json(prompt):
  """
  This function sends a prompt to the LLM requesting a JSON-formatted response.
  Then, parses and returns the output as a JSON object.
  """
  max_attempts = 2
  for attempt in range(1, max_attempts + 1):
    try:
      if _HAS_OPENAI_V1:
        res = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[
            {"role": "user", "content": prompt}
          ],
          timeout=20
        )
        content = res.choices[0].message.content if res and res.choices else ""

      else:
        res = openai_legacy.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "user", "content": prompt}
          ],
          request_timeout=20
        )
        content = res.choices[0].message.content if res and res.choices else ""

      # Process the LLM output
      res_match = re.search(r"```(?:json)?\s*(.*?)(?:```|$)", content, re.DOTALL)
      clean_res = res_match.group(1).strip() if res_match else (content.strip() if content else "")
      return json.loads(clean_res)
    except Exception as e:
        # Swallow and retry up to max_attempts
      if attempt >= max_attempts:
        print(f"[llm] call failed: {type(e).__name__}: {e!s}")
        return None

def extract_area_volume(dimensions):
    """
    Calculates area (m^2) and volume (m^3) based on standardized dimensions.
    """
    area_m2 = None
    volume_m3 = None

    # Extract standardized values, defaulting to None if not present or invalid
    length = dimensions['length']['standardized_value'] if dimensions['length'] else None
    width = dimensions['width']['standardized_value'] if dimensions['width'] else None
    height = dimensions['height']['standardized_value'] if dimensions['height'] else None
    thickness = dimensions['thickness']['standardized_value'] if dimensions['thickness'] else None

    # Resolve primary dimensions for calculation by prioritizing available ones.
    # Prioritize L, W, H. If H is missing, use T.
    calc_l = length
    calc_w = width
    calc_h = height # Start with explicit height

    # Replace the height, width or length calculation dimensions (if doesn't exist) by thickness - ONLY ONE
    if calc_h is None:
        calc_h = thickness
    elif calc_w is None:
        calc_w = thickness
    elif calc_l is None:
        calc_l = thickness

    # Volume Calculation: requires three dimensions (L, W, H_resolved)
    if calc_l is not None and calc_w is not None and calc_h is not None:
        volume_m3 = calc_l * calc_w * calc_h

    # Area Calculation: requires two dimensions (L, W, H_resolved)
    if calc_l is not None and calc_w is not None:
        area_m2 = calc_l * calc_w
    elif calc_l is not None and calc_h is not None: # Fallback for L x H_resolved if W is missing
        area_m2 = calc_l * calc_h
    elif calc_w is not None and calc_h is not None: # Fallback for W x H_resolved if L is missing
        area_m2 = calc_w * calc_h

    # Prepare the output dictionary structure
    return {
        'area': {
            'value': area_m2,
            'unit': 'm^2' if area_m2 is not None else None,
            'standardized_value': area_m2,
            'standardized_unit': 'm^2' if area_m2 is not None else None
        },
        'volume': {
            'value': volume_m3,
            'unit': 'm^3' if volume_m3 is not None else None,
            'standardized_value': volume_m3,
            'standardized_unit': 'm^3' if volume_m3 is not None else None
        }
    }


from scripts.config.material_keywords import MATERIAL_TYPE_KEYWORDS
from scripts.config.element_keywords import ELEMENT_TYPE_KEYWORDS
from scripts.config.material_specs import MATERIAL_SPEC_REGEX

"""### Functionality

Modify the Materials category using Regex for the specifications and also RapidFuzz to identify the material type, keywords and matched DB entry.
"""

# --- Material Specification Extractor (unchanged) ---
def extract_material_specification(material, text):
    if not text: return {'value': None, 'confidence': 0.0}
    # Ensure 'material' is a valid key before accessing MATERIAL_SPEC_REGEX
    if material not in MATERIAL_SPEC_REGEX:
        return {'value': None, 'confidence': 0.0}
    match = MATERIAL_SPEC_REGEX[material].search(text)
    if match:
        spec = match.group(0).upper()
        spec = re.sub(r'\bGRADE\b', '', spec, flags=re.IGNORECASE).replace(' ', '').strip()
        return {'value': spec, 'confidence': 0.9}
    else: return {'value': None, 'confidence': 0.0}

# --- Updated Fuzzy Material Type Extractor (unchanged) ---
def extract_material_type_fuzzy(text, material_keywords_dict):
    if text is None:
        return {'value': None, 'matched_db_entry': None, 'confidence': 0.0}

    text_lower = text.lower()
    text_lower = text_lower.replace(',', '') # Remove commas from the input text
    best_match = {'value': None, 'matched_db_entry': None, 'confidence': 0.0}
    all_keywords_for_fuzz = []

    for material_type, keywords in material_keywords_dict.items():
        for keyword in keywords:
            all_keywords_for_fuzz.append((keyword.lower(), material_type)) # Store (keyword_lower, material_type)

    if all_keywords_for_fuzz:
        # Create a list of just the keyword strings for process.extractOne
        keyword_strings_only = [item[0] for item in all_keywords_for_fuzz]

        # Use process.extractOne for fuzzy matching on the keyword strings
        # Use token_set_ratio as it handles word order and extra words well for keywords
        match_result = process.extractOne(text_lower, keyword_strings_only, scorer=fuzz.token_set_ratio, score_cutoff=75)

        if match_result:
            matched_keyword_str_from_list, score, index_in_keyword_strings_only = match_result

            # Retrieve the original material_type based on the index
            original_material_type = all_keywords_for_fuzz[index_in_keyword_strings_only][1]

            best_match['value'] = original_material_type
            best_match['matched_db_entry'] = original_material_type # Populate with the material_type itself
            best_match['confidence'] = score / 100.0 # Normalize score to 0-1 range

    return best_match

# --- Simplified Main Extraction Function (only for materials.primary) ---
def extract_materials_info(raw_material_text, input_text):
    result = {
        'material_type': None,
        'specification': None,
        'keywords': [],
        'matched_db_entry': None,
        'match_confidence': 0.0
    }

    # Extract material_type using fuzzy matching from the raw_material_text
    material_type_fuzzy_info = extract_material_type_fuzzy(raw_material_text, MATERIAL_TYPE_KEYWORDS)
    matched_material_type = material_type_fuzzy_info['value'] # Get the standardized material type

    result['material_type'] = matched_material_type

    # Update matched_db_entry and capitalize it
    if material_type_fuzzy_info['matched_db_entry']:
        result['matched_db_entry'] = material_type_fuzzy_info['matched_db_entry'].capitalize()
    else:
        result['matched_db_entry'] = None

    result['match_confidence'] = material_type_fuzzy_info['confidence']

    # Extract material_specification using the fuzzy matched material type as the key for regex
    if matched_material_type and matched_material_type in MATERIAL_SPEC_REGEX: # Added check for valid key
        material_spec_info = extract_material_specification(matched_material_type, input_text)
        if material_spec_info['value']:
            result['specification'] = material_spec_info['value']
            # If a spec is found, ensure its confidence is reflected, potentially overriding lower fuzzy material confidence
            result['match_confidence'] = max(
                result['match_confidence'],
                material_spec_info['confidence']
            )

    # Populate keywords list
    if result['material_type'] and result['material_type'] not in result['keywords']:
        result['keywords'].append(result['material_type'])
    if result['specification'] and result['specification'] not in result['keywords']:
        result['keywords'].append(result['specification'])

    return result

def extract_materials(text):
    # Extract materials hierarchy
    prompt = """
    You are given a sentence describing a construction element.
    Your task is to extract only the material words that appear explicitly in the sentence, preserving the original wording exactly as written.
    Follow these rules strictly:
        1. Identify how many distinct materials exist in the sentence.
        2. If a material appears more than once, only include it **once**.
        3. If there is only one material, return only "primary".
        4. If there are two materials (e.g., composite system), return "primary" and "secondary".
        5. If there are more, continue hierarchically ("tertiary", etc.).
        6. Each value must be exactly the word(s) as written in the sentence.
        7. If a value does not match any material, set it as null.
    Example:
        "reinforced concrete" -> "primary": "concrete"; "secondary": "reinforced";
        "betão armado" -> "primary": "betão"; "secondary": "armado";
        "Viga pré-esforçada"  -> "primary": "pré-esforçada";
    The output should have the following format:
    {{
        "primary": "material1",
        "secondary": "material2",
        ...
    }}
    Return only JSON (or null).

    Input Sentence: {input_text}
    """
     
    formatted_prompt = prompt.format(
      input_text=text
    )


    materials_hierarchy = llm_process_json(formatted_prompt)

    if not materials_hierarchy:
        return None

    cleaned_hierarchy = {}
    seen_material_types = set()

    for key, value in materials_hierarchy.items():

        processed = extract_materials_info(value, text)

        # Skip invalid materials
        if processed['material_type'] is None:
            continue

        # Skip duplicates based on normalized material_type
        if processed['material_type'] in seen_material_types:
            continue

        seen_material_types.add(processed['material_type'])
        cleaned_hierarchy[key] = processed

    return cleaned_hierarchy if cleaned_hierarchy else None



"""## Elements

For this field, we use Regex and RapidFuzz.

### Variables (constants)
"""



"""### Functionality

Modify the Element type category so that it's extraction is based on RapidFuzz.
"""

# --- New Fuzzy Keyword Extractor (simple, returns value and confidence) ---
def _extract_fuzzy_keyword_simple(text, keywords_dict):
    text_lower = text.lower()
    best_match = {'value': None, 'confidence': 0.0}
    all_keywords_for_fuzz = []

    for standard_value, keywords_list in keywords_dict.items():
        for keyword in keywords_list:
            all_keywords_for_fuzz.append((keyword.lower(), standard_value))

    if all_keywords_for_fuzz:
        keyword_strings_only = [item[0] for item in all_keywords_for_fuzz]
        words = re.findall(r'\b\w+\b', text_lower)

        max_score = 0
        current_best_value = None

        for word in words:
            match_result = process.extractOne(word, keyword_strings_only, scorer=fuzz.ratio, score_cutoff=75)
            if match_result:
                _, score, index_in_keyword_strings_only = match_result
                original_standard_value = all_keywords_for_fuzz[index_in_keyword_strings_only][1]

                if score > max_score:
                    max_score = score
                    current_best_value = original_standard_value

        if current_best_value:
            best_match['value'] = current_best_value
            best_match['confidence'] = max_score / 100.0

    return best_match

# --- Modified Main Extraction Function to only extract element_type ---
def extract_element_type(text):
    results = {
        'value': None,
        'confidence': 0.0
    }

    # Extract element_type using the fuzzy matching
    element_type_info = _extract_fuzzy_keyword_simple(text, ELEMENT_TYPE_KEYWORDS)
    results.update(element_type_info)

    return results


"""## Dimensions

This field's result is obtained with:
  - Regex
  - LLM

### Functionality

First, we use Regex to extract the dimensions from the input string:
"""

from scripts.config.units import (
    LINEAR_UNIT_ALTERNATIVES_RAW,
    WEIGHT_UNIT_ALTERNATIVES_RAW,
    ALL_UNITS_RAW,
    VALUE_UNIT_PATTERN,
    COMPOUND_PATTERN,
    SUM_PATTERN,
    UNIT_NORMALIZATION,
    UNIT_CONVERSIONS,
)


def convert_to_si(value, unit):
    unit = unit.lower()

    # Normalizar se necessário
    if unit in UNIT_NORMALIZATION:
        unit = UNIT_NORMALIZATION[unit]

    factor, standardized_unit = UNIT_CONVERSIONS[unit]

    return value * factor, standardized_unit


def extract_dimensions_values(text):
    """
    Extract all dimensions (linear + weight).
    Handles:
        - 10m
        - 5 x 2 x 0.3 m
        - 300kg
        - mixed units
    """

    if not text:
        return []

    results = []
    seen = set()  # prevents duplicates

    # ---------- Sum dimensions ----------
    for match in SUM_PATTERN.finditer(text):
        # Fetch unit
        unit = match.group(2).lower()
        if unit in UNIT_NORMALIZATION:
          unit = UNIT_NORMALIZATION[unit]

        # Fetch values
        numbers = re.findall(r'\d+\.?\d*', match.group(0))
        sum = 0
        for number in numbers:
            sum += float(number)

        # Append to result structure
        standardized_value, standardized_unit = convert_to_si(sum, unit)
        results.append({
            "value": sum,
            "unit": unit,
            "standardized_value": standardized_value,
            "standardized_unit": standardized_unit
        })

    # ---------- Compound dimensions ----------
    # Only linear units make sense here
    for match in COMPOUND_PATTERN.finditer(text):
        # Fetch unit
        unit = match.group(2).lower()
        if unit in UNIT_NORMALIZATION:
          unit = UNIT_NORMALIZATION[unit]

        # Fetch values
        numbers = re.findall(r'\d+\.?\d*', match.group(1))
        for num in numbers:
            key = (num, unit)
            seen.add(key)

            value = float(num)
            standardized_value, standardized_unit = convert_to_si(value, unit)

            results.append({
                "value": value,
                "unit": unit,
                "standardized_value": standardized_value,
                "standardized_unit": standardized_unit
            })

    # ---------- Explicit matches ----------
    for value_str, unit in VALUE_UNIT_PATTERN.findall(text):
        # Fetch unit
        unit = unit.lower()
        if unit in UNIT_NORMALIZATION:
          unit = UNIT_NORMALIZATION[unit]

        key = (value_str, unit)
        if (key in seen):
          continue
        # seen.add(key)

        value = float(value_str)
        standardized_value, standardized_unit = convert_to_si(value, unit)

        results.append({
            "value": value,
            "unit": unit,
            "standardized_value": standardized_value,
            "standardized_unit": standardized_unit
        })

    return results

def rule_based_dimension_assignment(text, extracted_values):
    if not extracted_values:
        return None

    text_lower = text.lower()

    result = {
        "length": None,
        "width": None,
        "height": None,
        "diameter": None,
        "thickness": None,
        "weight": None
    }

    # -----------------------------
    # 1. Detect element type
    # -----------------------------
    element_type = None

    if any(word in text_lower for word in ["pilar", "column"]):
        element_type = "column"
    elif any(word in text_lower for word in ["viga", "beam"]):
        element_type = "beam"
    elif any(word in text_lower for word in ["parede", "muro", "wall"]):
        element_type = "wall"
    elif any(word in text_lower for word in ["laje", "slab"]):
        element_type = "slab"
    elif any(word in text_lower for word in ["fundação", "foundation", "footing"]):
        element_type = "foundation"

    # Separate linear and weight values
    linear_values = [v for v in extracted_values if v["standardized_unit"] == "m"]
    weight_values = [v for v in extracted_values if v["standardized_unit"] == "kg"]

    # -----------------------------
    # 2. Assign by explicit keyword
    # -----------------------------
    for dim in linear_values:
        value = dim["standardized_value"]
        if "altura" in text_lower or "height" in text_lower or "depth" in text_lower or "profundidade" in text_lower :
            result["height"] = value
        elif "espessura" in text_lower or "thick" in text_lower:
            result["thickness"] = value
        elif "largura" in text_lower or "width" in text_lower:
            result["width"] = value
        elif "comprimento" in text_lower or "length" in text_lower:
            result["length"] = value

    # Assign weight if present
    if weight_values:
        result["weight"] = weight_values[0]["standardized_value"]

    # -----------------------------
    # 3. Compound dimensions (x / by / por / e)
    # -----------------------------
    COMPOUND_PATTERN = re.compile(r'([\d\.]+(?:\s*(?:x|by|por|e)\s*[\d\.]+)+)\s*(m|cm)?')
    for match in COMPOUND_PATTERN.finditer(text_lower):
        numbers = [float(n) for n in re.findall(r'\d+\.?\d*', match.group(1))]
        unit = match.group(2)
        if unit:
            numbers = [convert_to_si(n, unit)[0] for n in numbers]

        # Only assign if keywords have not already filled the dimension
        if len(numbers) == 2:
            if result["length"] is None:
                result["length"] = numbers[0]
            if result["width"] is None:
                result["width"] = numbers[1]
        elif len(numbers) == 3:
            if result["length"] is None:
                result["length"] = numbers[0]
            if result["width"] is None:
                result["width"] = numbers[1]
            if result["height"] is None:
                result["height"] = numbers[2]

    # -----------------------------
    # 4. Wall layered thickness (20 + 5)
    # -----------------------------
    if element_type == "wall" and "+" in text_lower:
      matches = re.findall(r'(\d+\.?\d*)\s*(m|cm)', text_lower)

      values = []
      for value, unit in matches:
          si_value = convert_to_si(float(value), unit)[0]
          values.append(si_value)

      if values:
          result["thickness"] = sum(values)

    return result

"""Now that we have extracted all the dimensions' values, we need to assign each one to the respective dimension (e.g. weight, height, etc.):"""

def call_llm(input, regex_values):
  # Define the generic prompt
  prompt = """
  You are given a sentence describing a construction element and a list of numeric dimension values extracted from it.
  Assign all values to the corresponding correct dimension category (width, length, height, diameter, thickness, weight).
  If three values are given for a column or pillar, assume the largest value is height.
  If two values are given for a beam, assume they are width and height.
  If only one large value is present, assume it is length.
  If a value does not match any category, set it as null.
  The output should have the following format:
  {{
    length: x1,
    width: x2,
    height: x3,
    diameter: x4,
    thickness: x5,
    weight: x6
  }}
  Return only JSON.

  Input Sentence: {input_text}
  Extracted dimension values:
  {dimension_values}
  """

  # Format the generic prompt with the input from arguments
  formatted_prompt = prompt.format(
    input_text=input,
    dimension_values=regex_values
  )

  return llm_process_json(formatted_prompt)

def extract_dimensions_categories(input_text, regex_values):
    rule_result = rule_based_dimension_assignment(input_text, regex_values)

    # Caso 1 & 2: rule-based falhou completamente
    if rule_result is None or all(v is None for v in rule_result.values()):
        return call_llm(input_text, regex_values)

    # Caso 3: complementar com LLM por dimensão faltante (fallback híbrido)
    prompt = """
    For this input text: {input_text}
    I obtained the following structure (mapping dimension categories to values): {rule_result}

    If necessary, change it. If you think it's correct, just return the same structure.
    Return only JSON.
    """
    formatted_prompt = prompt.format(input_text=input_text, rule_result=rule_result)
    return llm_process_json(formatted_prompt)

"""Also, use the LLM to classify the section type in one of the following:
*   Square
*   Rectangular
*   Circular


"""

def infer_section_type(dimensions):
  """
  Infers the section type (circular, square, rectangular) based on dimensions.
  Prioritizes diameter, then explicit linear dimensions (length, width, height),
  using thickness as a last resort to form a 2D section if only one explicit
  linear dimension is available.
  """
  if not dimensions:
        return None

  # Check for circular section from explicit diameter
  diameter = dimensions['diameter']['standardized_value'] if dimensions['diameter'] else None
  if diameter is not None:
      return "circular"

  # Collect explicit linear dimensions (length, width, height)
  explicit_linear_dims = []
  for key in ['length', 'width', 'height']:
      value = dimensions[key]['standardized_value'] if dimensions[key] else None
      if value is not None:
          explicit_linear_dims.append(value)

  thickness = dimensions['thickness']['standardized_value'] if dimensions['thickness'] else None

  # Case 1: Two or more explicit linear dimensions (primary dimensions like L, W, H)
  if len(explicit_linear_dims) >= 2:
      # Take the first two available explicit linear dimensions to define the section
      dim1, dim2 = explicit_linear_dims[0], explicit_linear_dims[1]
      if math.isclose(dim1, dim2, rel_tol=1e-9):
          return "square"
      else:
          return "rectangular"

  # Case 2: One explicit linear dimension and thickness (thickness as a secondary dimension)
  elif len(explicit_linear_dims) == 1 and thickness is not None:
      dim1 = explicit_linear_dims[0]
      dim2 = thickness
      if math.isclose(dim1, dim2, rel_tol=1e-9):
          return "square"
      else:
          return "rectangular"

  # Case 3: Only thickness is available, or less than two effective dimensions for a 2D section
  # If this point is reached, it means we cannot reliably determine a 2D section type.
  return None

"""Merge the two results in a single dimensions structure:"""

def extract_dimensions(input):
  regex_values = extract_dimensions_values(input)
  llm_category_assignments = extract_dimensions_categories(input, regex_values)
  if not isinstance(llm_category_assignments, dict):
    llm_category_assignments = {}

  # Initialize dimensions result structure using output_schema to ensure all fields are present
  dimensions_result = {
      'height': {
          'value': None,
          'unit': None,
          'standardized_value': None,
          'standardized_unit': None,
      },
      'width': {
          'value': None,
          'unit': None,
          'standardized_value': None,
          'standardized_unit': None,
      },
      'thickness': {
          'value': None,
          'unit': None,
          'standardized_value': None,
          'standardized_unit': None,
      },
      'length': {
          'value': None,
          'unit': None,
          'standardized_value': None,
          'standardized_unit': None,
      },
      'weight': {
          'value': None,
          'unit': None,
          'standardized_value': None,
          'standardized_unit': None,
      },
      'diameter': {
          'value': None,
          'unit': None,
          'standardized_value': None,
          'standardized_unit': None,
      },
      'area': {
          'value': None,
          'unit': None,
          'standardized_value': None,
          'standardized_unit': None
      },
      'volume': {
          'value': None,
          'unit': None,
          'standardized_value': None,
          'standardized_unit': None
      },
      'section_type': None
  }

  for dimension_category in dimensions_result.keys():
    # Skip area, volume, section_type as they are calculated later or come from a different LLM call
    if dimension_category in ['area', 'volume', 'section_type']:
        continue

    assigned_value_from_llm = (llm_category_assignments or {}).get(dimension_category)

    if assigned_value_from_llm is not None: # Only proceed if LLM assigned a value to this category
      for i, dimension_info in enumerate(regex_values):
        if dimension_info is not None: # Ensure the entry hasn't been marked as used
          # Compare standardized_value from regex with the value assigned by LLM
          # Using math.isclose for float comparison robustness
          if math.isclose(dimension_info['standardized_value'], assigned_value_from_llm, rel_tol=1e-9):
            # Update the dimension_result with the full dimension_info dict (excluding span for output)
            updated_dim_info = {k: v for k, v in dimension_info.items() if k != 'span'}
            dimensions_result[dimension_category].update(updated_dim_info)
            regex_values[i] = None # Mark this dimension_info as used
            break # Move to the next dimension_category

  # Calculate section_type
  dimensions_result['section_type'] = infer_section_type(dimensions_result) # Note: extract_section_type also calls infer_section_type

  # Calculate Area/Volume
  calculated_properties = extract_area_volume(dimensions_result) # This function expects `dimensions_result` structure
  dimensions_result['area'].update(calculated_properties['area'])
  dimensions_result['volume'].update(calculated_properties['volume'])

  return dimensions_result


def extract_count(text):
    """
    Extracts numerical quantities and associated units from the input text.
    Returns a dictionary with the value, unit, and confidence score.
    """
    if not text:
        return {'value': None, 'unit': None, 'confidence': 0.0}

    # Build a dynamic quantity regex that also matches "<value> <elementType>(s|es)"
    # Determine element type for plural-sensitive matching; fall back to a generic word if missing
    element_info = extract_element_type(text) or {'value': None}
    element_value = element_info.get('value')
    element_core = re.escape(element_value) if element_value else r'[a-zA-Z]+'
    element_pattern = rf'{element_core}(?:es|s)?'

    QUANTITY_REGEX = re.compile(
        rf'(?:'
            rf'\b(?:quantity|qty|quantidade|numero de itens|number of items)\s*[:=\s]*(\d+\.?\d*)\s*(units|unit|pieces|pecas|unidades|items|item|un)?\b'
            rf'|'
            rf'\b(\d+\.?\d*)\s*(units|unit|pieces|pecas|unidades|items|item|un)\b'
            rf'|'
            rf'\b(\d+\.?\d*)\s+{element_pattern}\b'
        rf')',
        re.IGNORECASE
    )

    match = QUANTITY_REGEX.search(text)
    if match:
        try:
            value = None
            unit = None

            # Check which part of the OR condition matched based on group presence
            if match.group(1) is not None: # Pattern 1 matched: keyword leads
                value = float(match.group(1))
                unit = match.group(2).lower() if match.group(2) else None
            elif match.group(3) is not None: # Pattern 2 matched: value with mandatory unit
                value = float(match.group(3))
                unit = match.group(4).lower() if match.group(4) else None

            if value is not None:
                return {'value': value, 'unit': unit, 'confidence': 0.9}
        except ValueError:
            pass
    return {'value': None, 'unit': None, 'confidence': 0.0}


def extract_quantities(input, dimensions):
    area_volume = extract_area_volume(dimensions)
    count_info = extract_count(input)

    calculate_flag = False
    if count_info['value'] is not None and count_info['value'] > 0:
        if area_volume['volume']['standardized_value'] is not None or area_volume['area']['standardized_value'] is not None:
            calculate_flag = True

    return {
        'calculate': calculate_flag,
        'volume': area_volume['volume'],
        'area': area_volume['area'],
        'count': count_info
    }


from scripts.config.construction_categories import CONSTRUCTION_CATEGORY_KEYWORDS

"""### Functionality"""

def extract_construction_category(text):
    """
    Performs case-insensitive keyword matching against the provided dictionary CONSTRUCTION_CATEGORY_KEYWORDS
    using RapidFuzz for fuzzy matching. Returns the standard value and a high confidence score if found,
    or {'value': None, 'confidence': 0.0} otherwise.
    """
    if not text:
        return {'value': None, 'confidence': 0.0}

    text_lower = text.lower()
    best_match_value = None
    max_score = 0.0

    # Prepare all keywords for fuzzy matching
    all_keywords_for_fuzz = []
    for standard_value, keywords_list in CONSTRUCTION_CATEGORY_KEYWORDS.items():
        for keyword in keywords_list:
            all_keywords_for_fuzz.append((keyword.lower(), standard_value)) # Store (keyword, original_category)

    if not all_keywords_for_fuzz:
        return {'value': None, 'confidence': 0.0}

    # Extract only the keyword strings for process.extractOne
    keyword_strings_only = [item[0] for item in all_keywords_for_fuzz]

    # Check for each word in the input text
    words = re.findall(r'\b\w+\b', text_lower)
    for word in words:
        # Use token_set_ratio as it handles word order and extra words well for keywords
        match_result = process.extractOne(word, keyword_strings_only, scorer=fuzz.token_set_ratio, score_cutoff=75)

        if match_result:
            matched_keyword_str, score, index_in_keyword_strings_only = match_result
            original_standard_value = all_keywords_for_fuzz[index_in_keyword_strings_only][1]

            if score > max_score:
                max_score = score
                best_match_value = original_standard_value

    if best_match_value:
        return {'value': best_match_value, 'confidence': max_score / 100.0}
    else:
        return {'value': None, 'confidence': 0.0}



def calculate_overall_confidence(element_type_info, materials_info, quantities, construction_category_info):
  confidence_scores = []

  if element_type_info['confidence'] > 0.0:
      confidence_scores.append(element_type_info['confidence'])
  #if materials_info['primary']['match_confidence'] > 0.0:
     # confidence_scores.append(materials_info['primary']['match_confidence'])
  if materials_info is not None and materials_info.get('primary') is not None:
    if materials_info['primary'].get('match_confidence', 0.0) > 0.0:
        confidence_scores.append(materials_info['primary']['match_confidence'])
  if quantities['count']['confidence'] > 0.0:
      confidence_scores.append(quantities['count']['confidence'])
  if construction_category_info['confidence'] > 0.0:
      confidence_scores.append(construction_category_info['confidence'])

  return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

"""## Generating the final output"""

def generate_final_output(input):
  dimensions = extract_dimensions(input)
  quantities = extract_quantities(input, dimensions)

  result = {
    'element_type': extract_element_type(input),
    'materials': extract_materials(input),
    'dimensions': dimensions,
    'quantities': quantities,
    'construction_category': extract_construction_category(input),
    'overall_confidence': calculate_overall_confidence(extract_element_type(input), extract_materials(input), quantities, extract_construction_category(input))
  }
  return result

