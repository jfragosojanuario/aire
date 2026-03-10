import os
from pathlib import Path
import sys
import json

import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[1]

# Prefer Streamlit secrets (production), fall back to .env (local dev)
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

if not os.environ.get("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Add it to Streamlit secrets (cloud) or .env (local).")
    st.stop()

# Ensure repo root is importable (so 'scripts' can be imported)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import project code
from scripts.text_input import generate_final_output
from scripts.config.material_keywords import MATERIAL_TYPE_KEYWORDS
from scripts.config.element_keywords import ELEMENT_TYPE_KEYWORDS


def preprocess_text_with_keywords(text: str, keyword_map: dict) -> str:
    if not isinstance(text, str):
        return ""
    processed_text = text.lower()
    added_keywords = set()
    for main_term, keywords_list in keyword_map.items():
        if main_term.lower() in processed_text:
            for keyword in keywords_list:
                added_keywords.add(keyword.lower())
    parts = [text]
    for kw in added_keywords:
        if kw not in processed_text:
            parts.append(kw)
    return " ".join(parts)


def extract_text_from_json(json_obj: dict) -> str:
    text_parts = []
    # element_type
    if (
        "element_type" in json_obj
        and "value" in json_obj["element_type"]
        and json_obj["element_type"]["value"] is not None
    ):
        text_parts.append(json_obj["element_type"]["value"])
    # materials.primary
    if "materials" in json_obj and json_obj["materials"] and "primary" in json_obj["materials"]:
        primary_material = json_obj["materials"]["primary"] or {}
        if "material_type" in primary_material and primary_material["material_type"] is not None:
            text_parts.append(primary_material["material_type"])
        if "specification" in primary_material and primary_material["specification"] is not None:
            text_parts.append(primary_material["specification"])
        if "keywords" in primary_material and primary_material["keywords"] is not None:
            text_parts.extend(primary_material["keywords"])
    # construction_category
    if (
        "construction_category" in json_obj
        and json_obj["construction_category"]
        and "value" in json_obj["construction_category"]
        and json_obj["construction_category"]["value"] is not None
    ):
        text_parts.append(json_obj["construction_category"]["value"])
    return " ".join(filter(None, text_parts)).strip()


@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def load_df_and_embeddings() -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_excel(REPO_ROOT / "database" / "Data AIRE project_V0.xlsx")
    df["combined_text_df"] = (
        df["Domain"].fillna("")
        + " "
        + df["Subdomain"].fillna("")
        + " "
        + df["product_material"].fillna("")
        + " "
        + df["related_terms_pt"].fillna("")
        + " "
        + df["related_terms_en"].fillna("")
        + " "
        + df["technical_specs"].fillna("")
    )
    global_keywords_map = {**MATERIAL_TYPE_KEYWORDS, **ELEMENT_TYPE_KEYWORDS}
    df["combined_text_df"] = df["combined_text_df"].apply(
        lambda x: preprocess_text_with_keywords(x, global_keywords_map)
    )
    model = load_model()
    df_embeddings = model.encode(df["combined_text_df"].tolist())
    return df, df_embeddings


def parse_rate_string(rate_str) -> Optional[float]:
    if pd.isna(rate_str):
        return None
    import re as _re
    rate_str = str(rate_str).replace(",", ".").strip()
    m = _re.search(r"([0-9]+\.?[0-9]*)", rate_str)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def normalize_text_for_match(text) -> str:
    import re as _re
    if pd.isna(text):
        return ""
    return _re.sub(r"[^a-z0-9]", "", str(text).lower())


def extract_structured_data_from_json(json_obj: dict) -> dict:
    extracted = {"element_type": None, "material_type": None, "specification": None, "dimensions": {}, "quantities": {}}
    if "element_type" in json_obj and "value" in json_obj["element_type"]:
        extracted["element_type"] = json_obj["element_type"]["value"]
    if "materials" in json_obj and json_obj["materials"] and "primary" in json_obj["materials"]:
        pm = json_obj["materials"]["primary"] or {}
        extracted["material_type"] = pm.get("material_type")
        extracted["specification"] = pm.get("specification")
    if "dimensions" in json_obj:
        for k, v in json_obj["dimensions"].items():
            if isinstance(v, dict) and ("value" in v or "standardized_value" in v):
                extracted["dimensions"][k] = {
                    "value": v.get("value"),
                    "unit": v.get("unit"),
                    "standardized_value": v.get("standardized_value"),
                    "standardized_unit": v.get("standardized_unit"),
                }
    if "quantities" in json_obj:
        for k, v in json_obj["quantities"].items():
            if isinstance(v, dict) and ("value" in v or "standardized_value" in v):
                extracted["quantities"][k] = {
                    "value": v.get("value"),
                    "unit": v.get("unit"),
                    "standardized_value": v.get("standardized_value"),
                    "standardized_unit": v.get("standardized_unit"),
                }
    return extracted


def calculate_material_quantity(extracted, df_measurement_unit, usage_rate_from_df, waste_rate_from_df, domain, product_material_df):
    from scripts.config.element_keywords import ELEMENT_TYPE_KEYWORDS
    from scripts.config.material_keywords import MATERIAL_TYPE_KEYWORDS
    gkm = {**MATERIAL_TYPE_KEYWORDS, **ELEMENT_TYPE_KEYWORDS}

    quantity = None
    quantity_unit = None
    assumptions = []

    dims = extracted.get("dimensions", {})
    quants = extracted.get("quantities", {})
    element_type = (extracted.get("element_type") or "").lower()
    material_type = (extracted.get("material_type") or "").lower()
    is_wood = (domain == "Wood Structures")

    actual_waste = waste_rate_from_df
    if actual_waste is None:
        actual_waste = 0.0
        assumptions.append("Waste rate defaulted to 0.0% (not specified).")
    actual_usage_generic = 1.0
    if usage_rate_from_df is None:
        assumptions.append("Usage rate defaulted to 1.0 (not specified).")

    def _val(d):
        return (d.get("standardized_value") or d.get("value")) if isinstance(d, dict) else None

    def _unit(d):
        return (d.get("standardized_unit") or d.get("unit")) if isinstance(d, dict) else None

    if isinstance(df_measurement_unit, str):
        df_measurement_unit = df_measurement_unit.strip()

    # Use pre-calculated quantities from JSON first
    vol = quants.get("volume", {})
    if df_measurement_unit == "m³" and vol:
        v, u = _val(vol), _unit(vol)
        if v is not None and u and ("m3" in u.lower() or "m^3" in u.lower()):
            quantity, quantity_unit = v, "m³"
    area_q = quants.get("area", {})
    if quantity is None and df_measurement_unit == "m²" and area_q:
        v, u = _val(area_q), _unit(area_q)
        if v is not None and u and u.lower() == "m2":
            quantity, quantity_unit = v, "m²"
    len_q = quants.get("length", {})
    if quantity is None and df_measurement_unit == "m" and len_q:
        v, u = _val(len_q), _unit(len_q)
        if v is not None and u and u.lower() == "m":
            quantity, quantity_unit = v, "m"
    wgt = quants.get("weight", {})
    if quantity is None and df_measurement_unit == "kg" and wgt:
        v, u = _val(wgt), _unit(wgt)
        if v is not None and u and u.lower() == "kg":
            quantity, quantity_unit = v, "kg"
    cnt = quants.get("count", {})
    if quantity is None and df_measurement_unit.lower() in ["ud", "un"] and cnt:
        quantity, quantity_unit = cnt.get("value"), df_measurement_unit

    if quantity is not None:
        rate = usage_rate_from_df if usage_rate_from_df is not None else actual_usage_generic
        return quantity * rate * (1 + actual_waste), quantity_unit, assumptions

    # kg/m using usage_rate
    if df_measurement_unit == "kg" and usage_rate_from_df:
        eff_len = _val(dims.get("length", {}))
        if eff_len is None:
            h = _val(dims.get("height", {}))
            if h is not None:
                eff_len = h
                assumptions.append(f"Length derived from height ({eff_len}m) for kg/m calc.")
        if eff_len is None:
            eff_len = 1.0
            assumptions.append("Length defaulted to 1.0m for kg/m calc.")
        return eff_len * usage_rate_from_df * (1 + actual_waste), "kg", assumptions

    # ud/un using usage_rate
    if df_measurement_unit.lower() in ["ud", "un"] and usage_rate_from_df:
        area = _val(dims.get("area", {}))
        is_cbw = product_material_df and "ceramic brick" in product_material_df.lower() and element_type == "wall"
        if area is None and is_cbw:
            l, h = _val(dims.get("length", {})), _val(dims.get("height", {}))
            t = _val(dims.get("thickness", {}))
            if l is not None and h is None:
                area = l * 1.0; assumptions.append(f"Area = length ({l}m) × assumed 1.0m height for Ceramic Brick Wall.")
            elif h is not None and l is None:
                area = 1.0 * h; assumptions.append(f"Area = assumed 1.0m length × height ({h}m) for Ceramic Brick Wall.")
            else:
                area = 1.0; assumptions.append("Area defaulted to 1.0m² for Ceramic Brick Wall.")
        elif area is None:
            return None, None, assumptions + ["Area missing for ud/un usage-rate calc."]
        return area * usage_rate_from_df * (1 + actual_waste), df_measurement_unit, assumptions

    # Generic dimension-based calculations
    if df_measurement_unit == "m³":
        lv = _val(dims.get("length", {})); wv = _val(dims.get("width", {}))
        tv = _val(dims.get("thickness", {})); hv = _val(dims.get("height", {}))
        av = _val(dims.get("area", {})) or _val(quants.get("area", {}))
        is_col = element_type == "column" or element_type in gkm.get("column", [])
        is_wall = element_type == "wall" or element_type in gkm.get("wall", [])
        is_conc = material_type == "concrete" or material_type in gkm.get("concrete", [])
        vol_calc = None

        if is_wall:
            d1 = lv or (assumptions.append("Length defaulted to 1.0m for wall.") or 1.0)
            d2 = hv or (assumptions.append("Height defaulted to 1.0m for wall.") or 1.0)
            if tv is None and is_conc and wv is not None:
                tv = wv; assumptions.append(f"Thickness derived from width ({tv}m) for concrete wall.")
            d3 = tv or (assumptions.append("Thickness defaulted to 1.0m for wall.") or 1.0)
            vol_calc = d1 * d2 * d3
        elif is_col:
            d1 = lv
            if d1 is None and hv is not None:
                d1 = hv; assumptions.append(f"Length derived from height ({d1}m) for column.")
            d1 = d1 or (assumptions.append("Length defaulted to 1.0m for column.") or 1.0)
            d2 = wv or (assumptions.append("Width defaulted to 1.0m for column.") or 1.0)
            d3 = tv or (assumptions.append("Thickness defaulted to 1.0m for column.") or 1.0)
            if av is not None and lv is None and hv is None:
                vol_calc = av * 1.0; assumptions.append(f"Volume = area ({av}m²) × assumed 1.0m length for column.")
            else:
                vol_calc = d1 * d2 * d3

        if vol_calc is None:
            d1 = lv or (assumptions.append("Length defaulted to 1.0m for volume.") or 1.0)
            d2 = wv or (assumptions.append("Width defaulted to 1.0m for volume.") or 1.0)
            d3 = tv or hv
            if d3 is None:
                assumptions.append("Thickness/Height defaulted to 1.0m for volume.")
                d3 = 1.0
            if is_wood:
                if lv is None: return None, None, assumptions + ["Missing length for wood structure."]
                if wv is None: return None, None, assumptions + ["Missing width for wood structure."]
                if tv is None and hv is None: return None, None, assumptions + ["Missing thickness/height for wood structure."]
            vol_calc = d1 * d2 * d3

        quantity, quantity_unit = vol_calc, "m³"

    elif df_measurement_unit == "m²":
        lv = _val(dims.get("length", {})); wv = _val(dims.get("width", {}))
        av = _val(dims.get("area", {})) or _val(quants.get("area", {}))
        is_cbw = product_material_df and "ceramic brick" in product_material_df.lower() and (extracted.get("element_type") or "").lower() == "wall"
        if av is not None:
            quantity = av
        elif lv is not None and wv is not None:
            quantity = lv * wv
        elif is_cbw:
            quantity = 1.0; assumptions.append("Area defaulted to 1.0m² for Ceramic Brick Wall.")
        else:
            lv = lv or (assumptions.append("Length defaulted to 1.0m for area.") or 1.0)
            wv = wv or (assumptions.append("Width defaulted to 1.0m for area.") or 1.0)
            quantity = lv * wv
        quantity_unit = "m²"

    elif df_measurement_unit == "m":
        lv = _val(dims.get("length", {}))
        if lv is None and (element_type == "column" or element_type in gkm.get("column", [])):
            hv = _val(dims.get("height", {}))
            if hv is not None:
                lv = hv; assumptions.append(f"Length derived from height ({lv}m) for column.")
        if lv is None:
            if is_wood: return None, None, assumptions + ["Missing length for wood structure."]
            lv = 1.0; assumptions.append("Length defaulted to 1.0m.")
        quantity, quantity_unit = lv, "m"

    if quantity is not None:
        return quantity * actual_usage_generic * (1 + actual_waste), quantity_unit, assumptions
    return None, None, assumptions


def render_results(user_input: str):
    if not user_input.strip():
        st.info("Enter a description (e.g., 'Steel beam S275 with length of 6 m') and press Analyze.")
        return

    with st.spinner("Analyzing input with the LLM and preparing matches..."):
        structured = generate_final_output(user_input)

    st.subheader("Structured output")
    st.json(structured, expanded=False)

    # Build combined text for retrieval
    global_keywords_map = {**MATERIAL_TYPE_KEYWORDS, **ELEMENT_TYPE_KEYWORDS}
    combined_text = preprocess_text_with_keywords(
        extract_text_from_json(structured), global_keywords_map
    )

    model = load_model()
    df, df_embeddings = load_df_and_embeddings()

    if not combined_text:
        st.warning("No relevant text extracted from the structured output for matching.")
        return

    extracted = extract_structured_data_from_json(structured)
    user_vec = model.encode(combined_text).reshape(1, -1)
    sims = cosine_similarity(user_vec, df_embeddings)[0]

    # Exact-match prioritisation for steel specifications
    user_spec = extracted.get("specification") or ""
    user_mat = (extracted.get("material_type") or "").lower()
    final_idxs = []
    best_exact_idx = -1

    if user_spec and user_mat == "steel":
        spec_norm = normalize_text_for_match(user_spec)
        candidates = []
        for row_idx, df_row in df.iterrows():
            if df_row["Domain"] == "Steel Structures":
                if spec_norm in normalize_text_for_match(df_row["product_material"]) or \
                   spec_norm in normalize_text_for_match(df_row["technical_specs"]):
                    candidates.append(row_idx)
        if candidates:
            best_exact_idx = max(candidates, key=lambda i: sims[i])
            final_idxs.append(best_exact_idx)
            for i in np.argsort(sims)[::-1]:
                if i not in final_idxs and len(final_idxs) < 5:
                    final_idxs.append(int(i))

    if not final_idxs:
        final_idxs = np.argsort(sims)[-5:][::-1].tolist()

    rows = []
    top_price_info = None

    for j, idx in enumerate(final_idxs):
        row = df.iloc[idx]
        sim = float(sims[idx])
        unit_prices_val = float(row["unit_prices"]) if pd.notna(row["unit_prices"]) else None
        measurement_units_val = str(row["measurement_units"]) if pd.notna(row["measurement_units"]) else None
        domain_val = str(row["Domain"]) if pd.notna(row["Domain"]) else None
        product_material_val = str(row["product_material"]) if pd.notna(row["product_material"]) else None
        usage_rate = parse_rate_string(row["usage_rate"]) if "usage_rate" in row else None
        waste_rate = parse_rate_string(row["waste_rate"]) if "waste_rate" in row else None

        calc_qty, calc_qty_unit, calc_assumptions = None, None, []
        total_price = None
        if unit_prices_val is not None and measurement_units_val is not None:
            calc_qty, calc_qty_unit, calc_assumptions = calculate_material_quantity(
                extracted, measurement_units_val,
                usage_rate_from_df=usage_rate, waste_rate_from_df=waste_rate,
                domain=domain_val, product_material_df=product_material_val,
            )
            if calc_qty is not None:
                total_price = calc_qty * unit_prices_val

        match_label = product_material_val or ""
        if best_exact_idx == idx:
            match_label += " ★ exact match"

        rows.append({
            "rank": j + 1,
            "similarity": round(sim, 4),
            "product": match_label,
            "domain": str(row["Domain"]),
            "subdomain": str(row["Subdomain"]),
            "unit_price": f"{unit_prices_val:.2f} €/{measurement_units_val}" if unit_prices_val and measurement_units_val else "N/A",
            "quantity": f"{calc_qty:.2f} {calc_qty_unit}" if calc_qty is not None else "N/A",
            "total_price": f"{total_price:.2f} €" if total_price is not None else "N/A",
        })

        if j == 0:
            top_price_info = {
                "product": product_material_val,
                "domain": domain_val,
                "subdomain": str(row["Subdomain"]) if pd.notna(row["Subdomain"]) else None,
                "unit_price": unit_prices_val,
                "measurement_units": measurement_units_val,
                "quantity": calc_qty,
                "quantity_unit": calc_qty_unit,
                "total_price": total_price,
                "assumptions": calc_assumptions,
            }

    st.subheader("Top matches")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Price Estimation Summary ---
    st.subheader("Price estimation")
    if top_price_info and top_price_info["total_price"] is not None:
        st.metric(
            label=f"Estimated total price",
            value=f"{top_price_info['total_price']:.2f} €",
        )
        with st.container(border=True):
            st.markdown("**Basis**")
            st.markdown(f"- **Product:** {top_price_info['product']}")
            st.markdown(f"- **Category:** {top_price_info['domain']} › {top_price_info['subdomain']}")
            st.markdown(f"- **Unit price:** {top_price_info['unit_price']:.2f} €/{top_price_info['measurement_units']}")
            st.markdown(f"- **Quantity used:** {top_price_info['quantity']:.2f} {top_price_info['quantity_unit']}")
        if top_price_info["assumptions"]:
            with st.expander("Assumptions"):
                for a in top_price_info["assumptions"]:
                    st.markdown(f"- {a}")
        else:
            st.caption("No assumptions needed for this estimation.")
    else:
        st.warning("Could not compute a price estimation for the top match (missing unit price or insufficient dimension data).")
        if top_price_info and top_price_info["assumptions"]:
            with st.expander("Assumptions / reasons"):
                for a in top_price_info["assumptions"]:
                    st.markdown(f"- {a}")


def main():
    st.set_page_config(page_title="AIRE - Cost Estimator", layout="centered")
    st.title("AIRE - Cost Estimator")
    st.caption("Type a construction element description to get structured data and top matches.")

    with st.container(border=True):
        user_input = st.text_input(
            "Describe the element",
            value="Steel beam S275 with length of 6 m",
            placeholder="e.g., Steel beam S275 with length of 6 m",
        )
        col1, col2 = st.columns([1, 4])
        with col1:
            run = st.button("Analyze", type="primary", use_container_width=True)
        with col2:
            st.markdown("")

    if run:
        render_results(user_input)

    with st.expander("Environment & Notes"):
        st.write(
            "- Requires OPENAI_API_KEY in .env at repo root.\n"
            "- Embeddings/model and dataset are cached for faster subsequent runs.\n"
            "- Dataset path: database/Data AIRE project_V0.xlsx"
        )


if __name__ == "__main__":
    main()

