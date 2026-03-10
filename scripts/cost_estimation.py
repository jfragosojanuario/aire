# -*- coding: utf-8 -*-
import json
import re # 
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from scripts.config.material_keywords import MATERIAL_TYPE_KEYWORDS as materials_keywords_map
from scripts.config.element_keywords import ELEMENT_TYPE_KEYWORDS as elements_keywords_map

# Combine the dictionaries into a global map
global_keywords_map = {**materials_keywords_map, **elements_keywords_map}


def preprocess_text_with_keywords(text, keyword_map):
    if not isinstance(text, str): # Handle non-string inputs gracefully
        return ''

    processed_text = text.lower() # Work with lowercased text for matching
    added_keywords = set() # Use a set to avoid duplicate keywords

    for main_term, keywords_list in keyword_map.items():
        if main_term.lower() in processed_text:
            for keyword in keywords_list:
                added_keywords.add(keyword.lower()) # Add keywords in lowercase

    # Append unique keywords to the original text
    # Ensure keywords are not already present in the original text (case-insensitive check)
    final_text_parts = [text]
    for kw in added_keywords:
        if kw not in processed_text:
            final_text_parts.append(kw)

    return ' '.join(final_text_parts)

"""## Get User JSON Inputs

### Subtask:
Load pre-generated JSON inputs from tests/generated_inputs.py if present; otherwise allow interactive entry.
"""

try:
    from tests.generated_inputs import GENERATED_INPUTS as user_json_inputs  # list of JSON strings
    print(f"Loaded {len(user_json_inputs)} JSON inputs from tests/generated_inputs.py")
except Exception:
    user_json_inputs = []
    num_inputs = int(input("How many JSON inputs do you want to provide? "))
    for i in range(num_inputs):
        print(f"\n--- Entering JSON Input {i+1} ---")
        print("Please provide the JSON string. It can be pure JSON, a JSON array, or Python-formatted (e.g., `my_json = \"\"\"{...}\"\"\"` or `{'key': 'value'}`).")
        json_str = input(f"Enter JSON string {i+1}: ")
        user_json_inputs.append(json_str)
    print(f"Collected {len(user_json_inputs)} JSON inputs.")


# Load the Excel data, making sure it exists
try:
    df = pd.read_excel("../database/Data AIRE project.xlsx")
    print("Dataset 'Data AIRE project.xlsx' loaded successfully.")
except FileNotFoundError:
    print("Error: '../database/Data AIRE project.xlsx' not found.")
    raise # Re-raise to stop execution if file is missing

# Create the combined_text_df column, handling potential NaN values
df['combined_text_df'] = (df['Domain'].fillna('') + ' ' +
                        df['Subdomain'].fillna('') + ' ' +
                        df['product_material'].fillna('') + ' ' +
                        df['related_terms_pt'].fillna('') + ' ' +
                        df['related_terms_en'].fillna('') + ' ' +
                        df['technical_specs'].fillna(''))

# Apply keyword preprocessing to the DataFrame's combined text
df['combined_text_df'] = df['combined_text_df'].apply(lambda x: preprocess_text_with_keywords(x, global_keywords_map))


def extract_text_from_json(json_obj):
    text_parts = []

    # element_type
    if 'element_type' in json_obj and 'value' in json_obj['element_type'] and json_obj['element_type']['value'] is not None:
        text_parts.append(json_obj['element_type']['value'])

    # materials.primary
    if 'materials' in json_obj and 'primary' in json_obj['materials']:
        primary_material = json_obj['materials']['primary']
        if 'material_type' in primary_material and primary_material['material_type'] is not None:
            text_parts.append(primary_material['material_type'])
        if 'specification' in primary_material and primary_material['specification'] is not None:
            text_parts.append(primary_material['specification'])
        if 'keywords' in primary_material and primary_material['keywords'] is not None:
            text_parts.extend(primary_material['keywords'])

    # construction_category
    if 'construction_category' in json_obj and 'value' in json_obj['construction_category'] and json_obj['construction_category']['value'] is not None:
        text_parts.append(json_obj['construction_category']['value'])

    return ' '.join(filter(None, text_parts)).strip()


def extract_structured_data_from_json(json_obj):
    extracted_data = {
        'element_type': None,
        'material_type': None,
        'specification': None,
        'dimensions': {},
        'quantities': {}
    }

    if 'element_type' in json_obj and 'value' in json_obj['element_type']:
        extracted_data['element_type'] = json_obj['element_type']['value']

    if 'materials' in json_obj and 'primary' in json_obj['materials']:
        primary_material = json_obj['materials']['primary']
        if 'material_type' in primary_material:
            extracted_data['material_type'] = primary_material['material_type']
        if 'specification' in primary_material:
            extracted_data['specification'] = primary_material['specification']

    if 'dimensions' in json_obj:
        for dim_key, dim_val in json_obj['dimensions'].items():
            if isinstance(dim_val, dict) and ('value' in dim_val or 'standardized_value' in dim_val):
                extracted_data['dimensions'][dim_key] = {
                    'value': dim_val.get('value'),
                    'unit': dim_val.get('unit'),
                    'standardized_value': dim_val.get('standardized_value'),
                    'standardized_unit': dim_val.get('standardized_unit')
                }

    if 'quantities' in json_obj:
        for q_key, q_val in json_obj['quantities'].items():
            if isinstance(q_val, dict) and ('value' in q_val or 'standardized_value' in q_val):
                extracted_data['quantities'][q_key] = {
                    'value': q_val.get('value'),
                    'unit': q_val.get('unit'),
                    'standardized_value': q_val.get('standardized_value'),
                    'standardized_unit': q_val.get('standardized_unit')
                }
    return extracted_data


def calculate_material_quantity(extracted_json_data, df_measurement_unit, usage_rate_from_df, waste_rate_from_df, domain, product_material_df):
    quantity = None
    quantity_unit = None
    assumptions = []

    dims = extracted_json_data.get('dimensions', {})
    quants = extracted_json_data.get('quantities', {})
    element_type = extracted_json_data.get('element_type', '').lower()
    material_type_from_json = extracted_json_data.get('material_type', '').lower()

    is_wood_structure = (domain == 'Wood Structures')

    # --- Handle usage_rate and waste_rate, and their assumptions ---
    actual_waste_rate = waste_rate_from_df
    if actual_waste_rate is None:
        actual_waste_rate = 0.0
        assumptions.append("Waste rate defaulted to 0.0% (no waste) if not specified or unparseable.")

    actual_usage_rate_for_generic_dims = 1.0
    if usage_rate_from_df is None:
        assumptions.append("Usage rate defaulted to 1.0 (no adjustment) if not specified or unparseable.")

    def _get_best_value(item_dict):
        if isinstance(item_dict, dict):
            return item_dict.get('standardized_value') or item_dict.get('value')
        return None

    def _get_best_unit(item_dict):
        if isinstance(item_dict, dict):
            return item_dict.get('standardized_unit') or item_dict.get('unit')
        return None

    # Strip whitespace from df_measurement_unit for robust comparison
    if isinstance(df_measurement_unit, str):
        df_measurement_unit = df_measurement_unit.strip()

    # Try to use explicitly calculated quantities first from the JSON
    volume_data = quants.get('volume', {})
    if df_measurement_unit == 'm³' and volume_data:
        q_value = _get_best_value(volume_data)
        q_unit = _get_best_unit(volume_data)
        if q_value is not None and (q_unit and ('m3' in q_unit.lower() or 'm^3' in q_unit.lower())):
            quantity = q_value
            quantity_unit = 'm³'

    area_data = quants.get('area', {})
    if quantity is None and df_measurement_unit == 'm²' and area_data:
        q_value = _get_best_value(area_data)
        q_unit = _get_best_unit(area_data)
        if q_value is not None and (q_unit and q_unit.lower() == 'm2'):
            quantity = q_value
            quantity_unit = 'm²'

    length_data_quant = quants.get('length', {})
    if quantity is None and df_measurement_unit == 'm' and length_data_quant:
        q_value = _get_best_value(length_data_quant)
        q_unit = _get_best_unit(length_data_quant)
        if q_value is not None and (q_unit and q_unit.lower() == 'm'):
            quantity = q_value
            quantity_unit = 'm'

    weight_data = quants.get('weight', {})
    if quantity is None and df_measurement_unit == 'kg' and weight_data:
        q_value = _get_best_value(weight_data)
        q_unit = _get_best_unit(weight_data)
        if q_value is not None and (q_unit and q_unit.lower() == 'kg'):
            quantity = q_value
            quantity_unit = 'kg'

    count_data = quants.get('count', {})
    if quantity is None and df_measurement_unit.lower() in ['ud', 'un'] and count_data:
        quantity = count_data.get('value')
        quantity_unit = df_measurement_unit

    if quantity is not None:
        return quantity * (usage_rate_from_df if usage_rate_from_df is not None else actual_usage_rate_for_generic_dims) * (1 + actual_waste_rate), quantity_unit, assumptions

    # --- Specific calculations requiring usage_rate_from_df ---
    if df_measurement_unit == 'kg' and usage_rate_from_df is not None and usage_rate_from_df > 0:
        effective_length = _get_best_value(dims.get('length', {}))

        if element_type == 'column' or element_type in global_keywords_map.get('column', []):
            if effective_length is None:
                height_value = _get_best_value(dims.get('height', {}))
                if height_value is not None:
                    effective_length = height_value
                    assumptions.append(f"Length for '{element_type}' element was derived from 'height' ({effective_length}m) as length was missing (for kg/m calculation).")

        if effective_length is None:
            effective_length = 1.0
            assumptions.append("Length (for kg/m calculation) defaulted to 1.0m due to missing length in input.")

        quantity = effective_length * usage_rate_from_df
        quantity_unit = 'kg'
        return quantity * (1 + actual_waste_rate), quantity_unit, assumptions

    if df_measurement_unit.lower() in ['ud', 'un'] and usage_rate_from_df is not None and usage_rate_from_df > 0:
        area = _get_best_value(dims.get('area', {}))

        is_matched_ceramic_brick_wall = False
        if product_material_df and 'ceramic brick' in product_material_df.lower():
            if extracted_json_data.get('element_type', '').lower() == 'wall':
                is_matched_ceramic_brick_wall = True

        if area is None and is_matched_ceramic_brick_wall:
            length = _get_best_value(dims.get('length', {}))
            width = _get_best_value(dims.get('width', {}))
            height = _get_best_value(dims.get('height', {}))
            thickness = _get_best_value(dims.get('thickness', {}))

            calculated_area = None
            if length is not None and height is None:
                height = 1.0
                calculated_area = length * height
                assumptions.append(f"For 'Ceramic Brick Wall', area calculated as provided length ({length}m) multiplied by assumed 1.0m height (due to missing height for area calculation).")
            elif height is not None and length is None:
                length = 1.0
                calculated_area = length * height
                assumptions.append(f"For 'Ceramic Brick Wall', area calculated as provided height ({height}m) multiplied by assumed 1.0m length (due to missing length for area calculation).")
            elif thickness is not None and length is None and width is None and height is None:
                calculated_area = 1.0
                assumptions.append("Area defaulted to 1.0m² for 'Ceramic Brick Wall' due to missing dimensions for area calculation.")
            elif calculated_area is None:
                calculated_area = 1.0
                assumptions.append("Area defaulted to 1.0m² for 'Ceramic Brick Wall' due to missing dimensions for area calculation and no other specific rules applying.")

            area = calculated_area

        elif area is None and not is_matched_ceramic_brick_wall:
            return None, None, assumptions + ["Area missing for units/m^2 calculation; cannot use usage rate."]

        quantity = area * usage_rate_from_df
        quantity_unit = df_measurement_unit
        return quantity * (1 + actual_waste_rate), quantity_unit, assumptions

    # --- Generic calculations from dimensions ---
    if df_measurement_unit == 'm³':
        length_val = _get_best_value(dims.get('length', {}))
        width_val = _get_best_value(dims.get('width', {}))
        thickness_val = _get_best_value(dims.get('thickness', {}))
        height_val = _get_best_value(dims.get('height', {}))
        area_from_json = _get_best_value(dims.get('area', {})) or _get_best_value(quants.get('area', {}))

        is_column = (element_type == 'column' or element_type in global_keywords_map.get('column', []))
        is_wall = (element_type == 'wall' or element_type in global_keywords_map.get('wall', []))
        is_concrete_material = (material_type_from_json == 'concrete' or material_type_from_json in global_keywords_map.get('concrete', []))

        d1, d2, d3 = None, None, None
        calculated_volume = None

        if is_wall:
            d1 = length_val
            if d1 is None:
                d1 = 1.0
                assumptions.append("Length defaulted to 1.0m for wall volume calculation (no explicit length provided).")
            d2 = height_val
            if d2 is None:
                d2 = 1.0
                assumptions.append("Height defaulted to 1.0m for wall volume calculation (no explicit height provided).")
            d3 = thickness_val
            if d3 is None:
                if is_concrete_material and width_val is not None:
                    d3 = width_val
                    assumptions.append(f"Thickness for concrete wall derived from 'width' ({d3}m) for volume calculation (no explicit thickness provided).")
                else:
                    d3 = 1.0
                    assumptions.append("Thickness defaulted to 1.0m for wall volume calculation (no explicit thickness or suitable width provided).")
            calculated_volume = d1 * d2 * d3

        elif is_column:
            d1 = length_val
            if d1 is None and height_val is not None:
                d1 = height_val
                assumptions.append(f"Length for '{element_type}' element was derived from 'height' ({d1}m) as length was missing (for m³ calculation).")
            if d1 is None:
                d1 = 1.0
                assumptions.append("Length defaulted to 1.0m for column volume calculation.")
            d2 = width_val
            if d2 is None:
                d2 = 1.0
                assumptions.append("Width defaulted to 1.0m for column volume calculation.")
            d3 = thickness_val
            if d3 is None:
                d3 = 1.0
                assumptions.append("Thickness/Depth defaulted to 1.0m for column volume calculation.")
            if area_from_json is not None and (length_val is None and height_val is None):
                assumed_length_for_area = 1.0
                calculated_volume = area_from_json * assumed_length_for_area
                assumptions.append(f"For '{element_type}', volume calculated as provided area ({area_from_json}m²) multiplied by assumed {assumed_length_for_area}m length (as length/height was missing for m³ calculation).")
            else:
                calculated_volume = d1 * d2 * d3

        if calculated_volume is None:
            d1 = length_val
            if d1 is None:
                d1 = 1.0
                assumptions.append("Length defaulted to 1.0m for generic volume calculation.")
            d2 = width_val
            if d2 is None:
                d2 = 1.0
                assumptions.append("Width defaulted to 1.0m for generic volume calculation.")
            d3 = thickness_val
            if d3 is None:
                d3 = height_val
                if d3 is None:
                    d3 = 1.0
                    assumptions.append("Thickness/Height defaulted to 1.0m for generic volume calculation.")

            if is_wood_structure:
                if length_val is None:
                    return None, None, assumptions + ["Missing length for volume calculation in wood structure; no usage rate provided for length-to-volume conversion."]
                if width_val is None:
                    return None, None, assumptions + ["Missing width for volume calculation in wood structure."]
                if thickness_val is None and height_val is None:
                    return None, None, assumptions + ["Missing thickness/height for volume calculation in wood structure."]

            calculated_volume = d1 * d2 * d3

        quantity = calculated_volume
        quantity_unit = 'm³'

    elif df_measurement_unit == 'm²':
        length = _get_best_value(dims.get('length', {}))
        width = _get_best_value(dims.get('width', {}))
        area_from_json = _get_best_value(dims.get('area', {})) or _get_best_value(quants.get('area', {}))

        is_matched_ceramic_brick_wall = False
        if product_material_df and 'ceramic brick' in product_material_df.lower():
            if extracted_json_data.get('element_type', '').lower() == 'wall':
                is_matched_ceramic_brick_wall = True

        if area_from_json is not None:
            quantity = area_from_json
        elif length is not None and width is not None:
            quantity = length * width
        elif is_matched_ceramic_brick_wall:
            quantity = 1.0
            assumptions.append("Area defaulted to 1.0m² for 'Ceramic Brick Wall' due to missing dimensions for area calculation.")
        else:
            if length is None:
                length = 1.0
                assumptions.append("Length defaulted to 1.0m for area calculation.")
            if width is None:
                width = 1.0
                assumptions.append("Width defaulted to 1.0m for area calculation.")
            quantity = length * width
        quantity_unit = 'm²'

    elif df_measurement_unit == 'm':
        length = _get_best_value(dims.get('length', {}))

        if element_type == 'column' or element_type in global_keywords_map.get('column', []):
            if length is None:
                height_value = _get_best_value(dims.get('height', {}))
                if height_value is not None:
                    length = height_value
                    assumptions.append(f"Length for '{element_type}' element was derived from 'height' ({length}m) as length was missing (for m calculation).")

        if length is None:
            if is_wood_structure:
                return None, None, assumptions + ["Missing length for linear quantity calculation in wood structure."]
            length = 1.0
            assumptions.append("Length defaulted to 1.0m for linear quantity calculation.")
        quantity = length
        quantity_unit = 'm'

    if quantity is not None:
        return quantity * actual_usage_rate_for_generic_dims * (1 + actual_waste_rate), quantity_unit, assumptions

    return quantity, quantity_unit, assumptions


def parse_rate_string(rate_str):
    if pd.isna(rate_str):
        return None
    rate_str = str(rate_str).replace(',', '.').strip()
    match = re.search(r'([0-9]+\.?[0-9]*)', rate_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def normalize_text_for_match(text):
    if pd.isna(text):
        return ""
    return re.sub(r'[^a-z0-9]', '', str(text).lower())


parsed_user_jsons = []
combined_text_user_jsons = []

for i, raw_json_input_str in enumerate(user_json_inputs):
    cleaned_json_str = raw_json_input_str.strip()

    json_data = None

    # Attempt to remove common prefixes like "JSON " or "json "
    if cleaned_json_str.lower().startswith("json "):
        cleaned_json_str = cleaned_json_str[5:].strip()

    # Attempt 1: Direct parsing (pure JSON)
    try:
        json_data = json.loads(cleaned_json_str)
    except json.JSONDecodeError:
        pass # Try further cleaning

    # Attempt 2: Extract from Python assignment with triple quotes (e.g., variable = """{...}""")
    if json_data is None:
        # Updated regex to correctly capture content within triple quotes, handling both single and double triple quotes
        match = re.search(r"^\s*[\w_]+\s*=\s*(['\"]{3})(.*?)\1$", cleaned_json_str, re.DOTALL)
        if match:
            extracted_content = match.group(2)
            # Unescape common JSON escapes if present (e.g., \" to ")
            potential_json_str = extracted_content.replace('"', '"')
            try:
                json_data = json.loads(potential_json_str)
            except json.JSONDecodeError:
                pass

    # Attempt 3: Extract from simple quoted strings (e.g., '{"key": "value"}' or "{\"key\": \"value"}")
    if json_data is None and (
        (cleaned_json_str.startswith('"') and cleaned_json_str.endswith('"')) or
        (cleaned_json_str.startswith("'") and cleaned_json_str.endswith("'"))
    ):
        # Remove outermost quotes and unescape if necessary
        content_inside_quotes = cleaned_json_str[1:-1]
        potential_json_str = content_inside_quotes.replace('"', '"')
        try:
            json_data = json.loads(potential_json_str)
        except json.JSONDecodeError:
            pass

    if json_data:
        parsed_user_jsons.append(json_data)
        extracted_text = extract_text_from_json(json_data)
        combined_text_user_jsons.append(preprocess_text_with_keywords(extracted_text, global_keywords_map))
    else:
        print(f"Error parsing JSON input {i+1}: Failed to decode JSON after multiple attempts.")
        print(f"Problematic string: {raw_json_input_str[:200]}...")
        parsed_user_jsons.append(None) # Append None to maintain list consistency
        combined_text_user_jsons.append('') # Append empty string for embedding

print(f"Successfully processed {len(combined_text_user_jsons)} user JSON inputs.")
for i, text in enumerate(combined_text_user_jsons):
    if text:
        print(f"Combined text for user input {i+1}: {text}")
    else:
        print(f"Combined text for user input {i+1}: (parsing failed or no relevant text found)")

"""## Load SentenceTransformer Model and Generate Embeddings

### Subtask:
Load the 'all-MiniLM-L6-v2' pre-trained `SentenceTransformer` model. Generate vector embeddings for the `combined_text_df` column of the DataFrame and for each of the combined text strings extracted from the user-provided JSON inputs. This step ensures the model is loaded and embeddings are generated independently.
"""

from sentence_transformers import SentenceTransformer

# Load a pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the combined text in the DataFrame
df_embeddings = model.encode(df['combined_text_df'].tolist())

# Generate embeddings for the combined text from user JSON inputs
# Ensure to handle cases where combined_text_user_jsons might be empty or contain empty strings
user_json_embeddings = []
for text in combined_text_user_jsons:
    if text:
        user_json_embeddings.append(model.encode(text))
    else:
        # Append an empty array or handle appropriately if parsing failed for some inputs
        user_json_embeddings.append(np.array([]))

print("SentenceTransformer model loaded.")
print(f"Embeddings generated for {len(user_json_embeddings)} user JSON inputs and {len(df_embeddings)} DataFrame entries.")

"""## Calculate Cosine Similarity and Display Top Matches for All Inputs

### Subtask:
For each user-provided JSON input, calculate the cosine similarity between its embedding and all entries in the DataFrame. Identify and display the top 5 most similar entries for each JSON, including their similarity score and relevant details from the DataFrame.
"""



results_summary = []

for i, user_embedding in enumerate(user_json_embeddings):
    # Ensure the corresponding parsed JSON data is available for calculations
    if i >= len(parsed_user_jsons) or parsed_user_jsons[i] is None:
        print(f"\nSkipping User JSON Input {i+1} as original JSON parsing failed.")
        results_summary.append({"input_index": i+1, "combined_text": combined_text_user_jsons[i], "matches": [], "status": "failed_json_parsing"})
        continue

    # Extract structured data from the user's parsed JSON for quantity calculation
    extracted_json_data = extract_structured_data_from_json(parsed_user_jsons[i])

    # Skip if embedding generation failed for this input
    if not isinstance(user_embedding, np.ndarray) or user_embedding.size == 0:
        print(f"\nSkipping User JSON Input {i+1} due to invalid embedding (JSON parsing or text extraction likely failed).")
        results_summary.append({"input_index": i+1, "combined_text": combined_text_user_jsons[i], "matches": [], "status": "failed_embedding"})
        continue

    # Reshape user_embedding for cosine_similarity function
    user_embedding_reshaped = user_embedding.reshape(1, -1)

    # Calculate cosine similarity between the user input embedding and all DataFrame embeddings
    similarities = cosine_similarity(user_embedding_reshaped, df_embeddings)

    # --- Exact Match Prioritization Logic (for steel specifications) ---
    final_top_indices = []
    user_primary_spec = extracted_json_data.get('specification', '')
    user_material_type = extracted_json_data.get('material_type', '').lower()

    best_exact_match_found = False
    best_exact_candidate_index = -1

    if user_primary_spec and user_material_type == 'steel':
        user_primary_spec_norm = normalize_text_for_match(user_primary_spec)
        exact_match_candidate_indices = []

        for row_idx, df_row in df.iterrows():
            df_product_material_norm = normalize_text_for_match(df_row['product_material'])
            df_technical_specs_norm = normalize_text_for_match(df_row['technical_specs'])

            condition_domain = (df_row['Domain'] == 'Steel Structures')
            condition_spec_in_product = (user_primary_spec_norm in df_product_material_norm)
            condition_spec_in_tech_specs = (user_primary_spec_norm in df_technical_specs_norm)

            if condition_domain and (condition_spec_in_product or condition_spec_in_tech_specs):
                exact_match_candidate_indices.append(row_idx)

        if exact_match_candidate_indices:
            best_exact_candidate_index = -1
            highest_exact_candidate_similarity = -1
            for idx in exact_match_candidate_indices:
                if similarities[0][idx] > highest_exact_candidate_similarity:
                    highest_exact_candidate_similarity = similarities[0][idx]
                    best_exact_candidate_index = idx

            if best_exact_candidate_index != -1:
                final_top_indices.append(best_exact_candidate_index)
                all_similarities_sorted_indices = np.argsort(similarities[0])[-len(df):][::-1]
                for idx in all_similarities_sorted_indices:
                    if idx not in final_top_indices and len(final_top_indices) < 5:
                        final_top_indices.append(idx)
                best_exact_match_found = True

    if not best_exact_match_found:
        top_5_indices_semantic = np.argsort(similarities[0])[-5:][::-1]
        final_top_indices = top_5_indices_semantic.tolist()

    top_5_df_entries = df.iloc[final_top_indices]
    top_5_similarities = similarities[0][final_top_indices]

    print(f"\nTop 5 matches for User JSON Input {i+1} (Text: {combined_text_user_jsons[i]}):")
    user_input_summary = {"input_index": i+1, "combined_text": combined_text_user_jsons[i], "matches": [], "status": "success"}
    for j, (index, row) in enumerate(top_5_df_entries.iterrows()):
        unit_prices_val = float(row['unit_prices']) if pd.notna(row['unit_prices']) else None
        measurement_units_val = str(row['measurement_units']) if pd.notna(row['measurement_units']) else None
        domain_val = str(row['Domain']) if pd.notna(row['Domain']) else None
        product_material_df = str(row['product_material']) if pd.notna(row['product_material']) else None

        usage_rate_val = parse_rate_string(row['usage_rate']) if 'usage_rate' in row else None
        waste_rate_val = parse_rate_string(row['waste_rate']) if 'waste_rate' in row else None

        calculated_quantity = None
        calculated_quantity_unit = None
        total_price = None
        calculation_assumptions = []

        if unit_prices_val is not None and measurement_units_val is not None:
            calculated_quantity, calculated_quantity_unit, calculation_assumptions = calculate_material_quantity(
                extracted_json_data, measurement_units_val, usage_rate_from_df=usage_rate_val, waste_rate_from_df=waste_rate_val, domain=domain_val, product_material_df=product_material_df
            )
            if calculated_quantity is not None:
                total_price = calculated_quantity * unit_prices_val

        price_display = "N/A"
        if unit_prices_val is not None and measurement_units_val is not None:
            price_display = f"{unit_prices_val:.2f}€/{measurement_units_val}"
        elif unit_prices_val is not None:
            price_display = f"{unit_prices_val:.2f}€"
        elif measurement_units_val is not None:
            price_display = f"N/A / {measurement_units_val}"

        quantity_display = "N/A"
        if calculated_quantity is not None and calculated_quantity_unit is not None:
            quantity_display = f"{calculated_quantity:.2f} {calculated_quantity_unit}"

        total_price_display = "N/A"
        if total_price is not None:
            total_price_display = f"{total_price:.2f}€"

        match_label = ""
        if best_exact_match_found and best_exact_candidate_index == index:
            match_label = " (Prioritized Exact Match)"

        match_info = {
            "similarity": float(top_5_similarities[j]),
            "product_material": str(row['product_material']),
            "domain": str(row['Domain']),
            "subdomain": str(row['Subdomain']),
            "unit_prices": unit_prices_val,
            "measurement_units": measurement_units_val,
            "usage_rate": usage_rate_val,
            "waste_rate": waste_rate_val,
            "calculated_quantity": calculated_quantity,
            "calculated_quantity_unit": calculated_quantity_unit,
            "total_price": total_price,
            "assumptions": calculation_assumptions
        }
        user_input_summary["matches"].append(match_info)

        print(f"  {j+1}. Similarity: {top_5_similarities[j]:.4f} - {row['product_material']} ({row['Domain']} - {row['Subdomain']}){match_label}\n     - Unit Price: {price_display}, Quantity needed: {quantity_display}, Total Price: {total_price_display}")
        if calculation_assumptions:
            print(f"     - Assumptions: {'; '.join(calculation_assumptions)}")
        else:
            print("     - Assumptions: No assumptions needed.")

    results_summary.append(user_input_summary)

# --- Global Price Estimation Summary ---
if results_summary:
    first_user_input_summary = results_summary[0]
    if first_user_input_summary and first_user_input_summary['matches']:
        top_match = first_user_input_summary['matches'][0]
        total_price_estimation = top_match.get('total_price')
        assumptions = top_match.get('assumptions', [])
        product_material = top_match.get('product_material')

        print("\n--- Global Price Estimation ---")
        if total_price_estimation is not None:
            print(f"Price Estimation: {total_price_estimation:.2f}€")
            if product_material:
                print(f"Based on: {product_material}")
            if assumptions:
                print(f"Assumptions: {'; '.join(assumptions)}")
            else:
                print("Assumptions: No assumptions needed.")
        else:
            print("Price estimation not available for the top match.")
            if product_material:
                print(f"Based on: {product_material}")
            if assumptions:
                print(f"Assumptions: {'; '.join(assumptions)}")
            else:
                print("Assumptions: No assumptions needed.")
    else:
        print("No matches found for the user input.")
else:
    print("No results summary available.")