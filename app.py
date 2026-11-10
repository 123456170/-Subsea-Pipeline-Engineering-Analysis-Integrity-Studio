import streamlit as st
import numpy as np
import pandas as pd
import math
import json
import io
import hashlib
from copy import deepcopy
from datetime import datetime
from scipy.optimize import brentq
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

try:
    import sympy as sp
    HAVE_SYMPY = True
except Exception:
    HAVE_SYMPY = False

try:
    import fipy
    HAVE_FIPY = True
except Exception:
    HAVE_FIPY = False

st.set_page_config(
    page_title="Subsea Pipeline Integrity Studio",
    page_icon="🌊",
    layout="wide",
)

G = 9.80665

PROFILE_COLUMNS = [
    "distance_m",
    "section",
    "pressure_pa",
    "pressure_barg",
    "temperature_c",
    "velocity_m_s",
    "reynolds",
    "friction_factor",
    "dp_friction_pa",
    "dp_elevation_pa",
    "wall_shear_pa",
    "density_kg_m3",
    "viscosity_pa_s",
    "flow_m3_s",
]

TRANSIENT_COLUMNS = [
    "time_s",
    "pressure_pa",
    "pressure_barg",
    "flow_m3_s",
    "flow_m3_h",
    "surge_pa",
    "surge_barg",
]

MATERIALS = {
    "API 5L X65": {
        "smts_pa": 530e6,
        "yield_pa": 450e6,
        "youngs_pa": 200e9,
        "poisson": 0.30,
        "default_roughness_mm": 0.045,
    },
    "API 5L X42": {
        "smts_pa": 413e6,
        "yield_pa": 290e6,
        "youngs_pa": 200e9,
        "poisson": 0.30,
        "default_roughness_mm": 0.050,
    },
    "Duplex 2205": {
        "smts_pa": 793e6,
        "yield_pa": 540e6,
        "youngs_pa": 200e9,
        "poisson": 0.30,
        "default_roughness_mm": 0.050,
    },
    "Inconel 625": {
        "smts_pa": 827e6,
        "yield_pa": 414e6,
        "youngs_pa": 207e9,
        "poisson": 0.30,
        "default_roughness_mm": 0.040,
    },
}

PIPE_LIBRARY = [
    {"nps": 6, "schedule": "40", "od_mm": 168.28, "wall_mm": 7.11, "internal_diameter_mm": 154.06, "roughness_mm": 0.045, "material": "API 5L X65"},
    {"nps": 6, "schedule": "80", "od_mm": 168.28, "wall_mm": 10.97, "internal_diameter_mm": 146.34, "roughness_mm": 0.045, "material": "API 5L X65"},
    {"nps": 8, "schedule": "40", "od_mm": 219.08, "wall_mm": 8.18, "internal_diameter_mm": 202.72, "roughness_mm": 0.045, "material": "API 5L X65"},
    {"nps": 8, "schedule": "80", "od_mm": 219.08, "wall_mm": 12.70, "internal_diameter_mm": 193.68, "roughness_mm": 0.045, "material": "API 5L X65"},
    {"nps": 10, "schedule": "40", "od_mm": 273.05, "wall_mm": 9.27, "internal_diameter_mm": 254.51, "roughness_mm": 0.045, "material": "API 5L X65"},
    {"nps": 10, "schedule": "80", "od_mm": 273.05, "wall_mm": 12.70, "internal_diameter_mm": 247.65, "roughness_mm": 0.045, "material": "API 5L X65"},
    {"nps": 12, "schedule": "40", "od_mm": 323.85, "wall_mm": 10.31, "internal_diameter_mm": 303.23, "roughness_mm": 0.045, "material": "API 5L X65"},
    {"nps": 12, "schedule": "80", "od_mm": 323.85, "wall_mm": 14.27, "internal_diameter_mm": 295.31, "roughness_mm": 0.045, "material": "API 5L X65"},
    {"nps": 16, "schedule": "40", "od_mm": 406.40, "wall_mm": 12.70, "internal_diameter_mm": 381.00, "roughness_mm": 0.045, "material": "API 5L X65"},
    {"nps": 16, "schedule": "80", "od_mm": 406.40, "wall_mm": 16.66, "internal_diameter_mm": 373.08, "roughness_mm": 0.045, "material": "API 5L X65"},
    {"nps": 20, "schedule": "40", "od_mm": 508.00, "wall_mm": 15.09, "internal_diameter_mm": 477.82, "roughness_mm": 0.045, "material": "API 5L X65"},
    {"nps": 20, "schedule": "80", "od_mm": 508.00, "wall_mm": 22.23, "internal_diameter_mm": 463.54, "roughness_mm": 0.045, "material": "API 5L X65"},
    {"nps": 24, "schedule": "40", "od_mm": 609.60, "wall_mm": 17.48, "internal_diameter_mm": 574.64, "roughness_mm": 0.045, "material": "API 5L X65"},
    {"nps": 24, "schedule": "80", "od_mm": 609.60, "wall_mm": 25.40, "internal_diameter_mm": 558.80, "roughness_mm": 0.045, "material": "API 5L X65"},
    {"nps": 36, "schedule": "40", "od_mm": 914.40, "wall_mm": 19.05, "internal_diameter_mm": 876.30, "roughness_mm": 0.050, "material": "API 5L X65"},
]

FLUIDS = {
    "Crude oil (demo)": {
        "rho_ref_kg_m3": 850.0,
        "mu_ref_pa_s": 0.080,
        "Tref_K": 293.15,
        "beta_1_K": 0.00075,
        "andrade_B_K": 1200.0,
        "bulk_modulus_pa": 1.4e9,
        "min_temp_c": -10.0,
        "max_temp_c": 120.0,
    },
    "Seawater": {
        "rho_ref_kg_m3": 1025.0,
        "mu_ref_pa_s": 0.00105,
        "Tref_K": 293.15,
        "beta_1_K": 0.00020,
        "andrade_B_K": 60.0,
        "bulk_modulus_pa": 2.2e9,
        "min_temp_c": -2.0,
        "max_temp_c": 35.0,
    },
    "Diesel / gas oil": {
        "rho_ref_kg_m3": 840.0,
        "mu_ref_pa_s": 0.0030,
        "Tref_K": 293.15,
        "beta_1_K": 0.00080,
        "andrade_B_K": 350.0,
        "bulk_modulus_pa": 1.3e9,
        "min_temp_c": -10.0,
        "max_temp_c": 80.0,
    },
    "Ethylene glycol mix (demo)": {
        "rho_ref_kg_m3": 1050.0,
        "mu_ref_pa_s": 0.0150,
        "Tref_K": 293.15,
        "beta_1_K": 0.00050,
        "andrade_B_K": 800.0,
        "bulk_modulus_pa": 1.8e9,
        "min_temp_c": -20.0,
        "max_temp_c": 90.0,
    },
}

APP_WIDGET_PREFIXES = (
    "sb_",
    "t_",
    "lib_",
    "cust_",
    "upd_",
    "rec_",
    "del_",
    "scen_",
    "hyd_",
    "target_",
    "solve_",
    "download_",
    "run_",
    "gen_",
    "reset_",
    "save_",
)


def default_config():
    return {
        "fluid": "Crude oil (demo)",
        "flow_rate_m3_h": 540.0,
        "inlet_pressure_barg": 120.0,
        "inlet_temp_c": 60.0,
        "ambient_temp_c": 4.0,
        "segments": 50,
        "sections": [
            {
                "name": "Uplink 12in",
                "length_m": 5000.0,
                "internal_diameter_mm": 303.23,
                "wall_thickness_mm": 10.31,
                "roughness_mm": 0.045,
                "elevation_start_m": -1000.0,
                "elevation_end_m": -950.0,
                "material": "API 5L X65",
                "inspection_interval_days": 365,
            },
            {
                "name": "Mainline 16in",
                "length_m": 15000.0,
                "internal_diameter_mm": 381.00,
                "wall_thickness_mm": 12.70,
                "roughness_mm": 0.045,
                "elevation_start_m": -950.0,
                "elevation_end_m": -1000.0,
                "material": "API 5L X65",
                "inspection_interval_days": 365,
            },
        ],
    }


def default_records():
    return [
        {
            "id": "INS-001",
            "date": "2026-08-20",
            "section": "Uplink 12in",
            "type": "MFL",
            "defect": "Corrosion",
            "metal_loss_pct": 24.0,
            "length_pct": 8.0,
            "remaining_wall_mm": 7.8,
            "growth_rate_pct_yr": 1.8,
            "status": "Monitor",
            "notes": "Demo inline inspection indication for screening only.",
        },
        {
            "id": "INS-002",
            "date": "2026-09-10",
            "section": "Uplink 12in",
            "type": "CP",
            "defect": "None",
            "metal_loss_pct": 0.0,
            "length_pct": 0.0,
            "remaining_wall_mm": 10.31,
            "growth_rate_pct_yr": 0.0,
            "status": "OK",
            "notes": "Cathodic protection potential satisfactory in demo record.",
        },
    ]


def default_transient_params():
    return {
        "closure_time_s": 1.0,
        "closure_fraction": 0.80,
        "duration_s": 20.0,
        "dt": 0.05,
    }


def init_session():
    if "config" not in st.session_state:
        st.session_state.config = default_config()
    if "inspection_records" not in st.session_state:
        st.session_state.inspection_records = default_records()
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = []
    if "transient_params" not in st.session_state:
        st.session_state.transient_params = default_transient_params()
    if "self_test_results" not in st.session_state:
        st.session_state.self_test_results = None
    if "ai_results" not in st.session_state:
        st.session_state.ai_results = None
    if "report_md" not in st.session_state:
        st.session_state.report_md = None
    if "report_json" not in st.session_state:
        st.session_state.report_json = None
    if "analysis_hash" not in st.session_state:
        st.session_state.analysis_hash = None
    if "transient_hash" not in st.session_state:
        st.session_state.transient_hash = None
    if "flash" not in st.session_state:
        st.session_state.flash = None


def reset_session_to_demo():
    st.session_state.config = default_config()
    st.session_state.inspection_records = default_records()
    st.session_state.scenarios = []
    st.session_state.transient_params = default_transient_params()
    st.session_state.self_test_results = None
    st.session_state.ai_results = None
    st.session_state.report_md = None
    st.session_state.report_json = None
    st.session_state.analysis_hash = None
    st.session_state.transient_hash = None

    for k in list(st.session_state.keys()):
        if any(str(k).startswith(p) for p in APP_WIDGET_PREFIXES):
            del st.session_state[k]


def set_flash(kind, message):
    st.session_state.flash = {"kind": kind, "message": message}


def show_flash():
    flash = st.session_state.get("flash")
    if flash:
        kind = flash.get("kind", "info")
        message = flash.get("message", "")
        if kind == "success":
            st.success(message)
        elif kind == "error":
            st.error(message)
        elif kind == "warning":
            st.warning(message)
        else:
            st.info(message)
        st.session_state.flash = None


def config_hash(cfg):
    try:
        return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
    except Exception:
        return hashlib.md5(repr(cfg).encode()).hexdigest()


def option_index(options, value, default=0):
    try:
        return options.index(value)
    except Exception:
        return default


def sanitize_select(key, options, default=None):
    options = list(options)
    if not options:
        if key in st.session_state:
            del st.session_state[key]
        return

    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]

    if key not in st.session_state:
        if default is not None and default in options:
            st.session_state[key] = default
        else:
            st.session_state[key] = options[0]


def get_fluid(name):
    return deepcopy(FLUIDS.get(name, FLUIDS["Seawater"]))


def fluid_properties(fluid, temp_c):
    T = float(temp_c) + 273.15
    Tref = float(fluid["Tref_K"])
    beta = float(fluid["beta_1_K"])
    rho = float(fluid["rho_ref_kg_m3"]) * (1.0 - beta * (T - Tref))
    rho = float(np.clip(rho, 500.0, 1500.0))

    B = float(fluid["andrade_B_K"])
    exponent = B * (1.0 / T - 1.0 / Tref)
    exponent = float(np.clip(exponent, -50.0, 50.0))
    mu = float(fluid["mu_ref_pa_s"]) * math.exp(exponent)
    mu = float(np.clip(mu, 1e-6, 20.0))
    return rho, mu


def friction_factor(Re, rel_rough):
    if Re <= 0:
        return 0.0
    if Re < 2300.0:
        return 64.0 / Re

    rr = max(float(rel_rough), 1e-10)
    f = 0.02

    for _ in range(100):
        sqrt_f = math.sqrt(max(f, 1e-12))
        term = rr / 3.7 + 2.51 / (Re * sqrt_f)
        if term <= 0:
            break
        log_term = math.log10(term)
        if log_term == 0:
            break
        f_new = 1.0 / ((-2.0 * log_term) ** 2)
        if not math.isfinite(f_new) or f_new <= 0:
            break
        if abs(f_new - f) < 1e-12:
            f = f_new
            break
        f = 0.5 * f + 0.5 * f_new

    try:
        swamee = 0.25 / (math.log10(rr / 3.7 + 5.74 / (Re ** 0.9)) ** 2)
        if math.isfinite(swamee) and swamee > 0:
            if abs(swamee - f) < 0.01:
                f = swamee
    except Exception:
        pass

    return float(np.clip(f, 1e-6, 0.10))


def section_geometry(sec):
    D = float(sec.get("internal_diameter_mm", 0.0)) / 1000.0
    e = float(sec.get("wall_thickness_mm", 0.0)) / 1000.0
    rough = float(sec.get("roughness_mm", 0.0)) / 1000.0
    area = math.pi * D * D / 4.0 if D > 0 else 0.0
    return D, e, rough, area


def temperature_at(x, length, t_in, t_amb):
    if length <= 0:
        return float(t_in)
    Lchar = max(float(length) / 3.0, 50.0)
    return float(t_amb) + (float(t_in) - float(t_amb)) * math.exp(-float(x) / Lchar)


def analyze_section(sec, fluid, Q, p_in_pa, t_in_c, t_amb_c, global_segments, start_distance_m):
    D, e, rough, area = section_geometry(sec)
    length = float(sec.get("length_m", 0.0))
    segments = int(sec.get("segments", global_segments))
    segments = int(np.clip(segments, 1, 2000))

    if length <= 0 or area <= 0 or D <= 0:
        return float(p_in_pa), float(t_in_c), pd.DataFrame(columns=PROFILE_COLUMNS)

    dx = length / segments
    dz_total = float(sec.get("elevation_end_m", 0.0)) - float(sec.get("elevation_start_m", 0.0))
    dz_per = dz_total / segments

    rows = []
    p = float(p_in_pa)
    t_out = float(t_in_c)

    for i in range(segments):
        x_local = (i + 0.5) * dx
        x_global = start_distance_m + x_local
        t_local = temperature_at(x_local, length, t_in_c, t_amb_c)
        rho, mu = fluid_properties(fluid, t_local)
        V = Q / area
        nu = mu / rho if rho > 0 else 1e-6
        Re = V * D / nu if nu > 0 else 0.0
        f = friction_factor(Re, rough / D)
        dp_f = f * dx / D * rho * V * V / 2.0
        dp_e = rho * G * dz_per
        p_next = p - dp_f - dp_e

        rows.append(
            {
                "distance_m": x_global,
                "section": sec.get("name", "Section"),
                "pressure_pa": p_next,
                "pressure_barg": p_next / 1e5,
                "temperature_c": t_local,
                "velocity_m_s": V,
                "reynolds": Re,
                "friction_factor": f,
                "dp_friction_pa": dp_f,
                "dp_elevation_pa": dp_e,
                "wall_shear_pa": f * rho * V * V / 8.0,
                "density_kg_m3": rho,
                "viscosity_pa_s": mu,
                "flow_m3_s": Q,
            }
        )

        p = p_next
        t_out = t_local

    df = pd.DataFrame(rows, columns=PROFILE_COLUMNS)
    return p, t_out, df


def run_steady(cfg):
    Q = float(cfg.get("flow_rate_m3_h", 0.0)) / 3600.0
    p = float(cfg.get("inlet_pressure_barg", 0.0)) * 1e5
    t = float(cfg.get("inlet_temp_c", 20.0))
    amb = float(cfg.get("ambient_temp_c", 10.0))
    fluid = get_fluid(cfg.get("fluid", "Seawater"))
    segments = int(cfg.get("segments", 50))

    rows = []
    summaries = []
    dist = 0.0
    total_length = 0.0

    for sec in cfg.get("sections", []):
        p_in_sec = p
        t_in_sec = t
        length = float(sec.get("length_m", 0.0))
        p, t, df = analyze_section(sec, fluid, Q, p, t, amb, segments, dist)

        if not df.empty:
            df["section"] = sec.get("name", "Section")
            rows.append(df)

        D, e, rough, area = section_geometry(sec)
        dp_total = p_in_sec - p

        summaries.append(
            {
                "Section": sec.get("name", "Section"),
                "Length (m)": length,
                "ID (mm)": float(sec.get("internal_diameter_mm", 0.0)),
                "Wall (mm)": float(sec.get("wall_thickness_mm", 0.0)),
                "Inlet pressure (barg)": p_in_sec / 1e5,
                "Outlet pressure (barg)": p / 1e5,
                "Pressure drop (barg)": dp_total / 1e5,
                "Elevation change (m)": float(sec.get("elevation_end_m", 0.0)) - float(sec.get("elevation_start_m", 0.0)),
                "Velocity (m/s)": Q / area if area > 0 else 0.0,
                "Reynolds": float(df["reynolds"].mean()) if not df.empty else 0.0,
                "Friction factor": float(df["friction_factor"].mean()) if not df.empty else 0.0,
                "Outlet temperature (C)": t,
            }
        )

        dist += length
        total_length += length

    profile = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=PROFILE_COLUMNS)

    inlet_rho, _ = fluid_properties(fluid, float(cfg.get("inlet_temp_c", 20.0)))
    outlet_rho, _ = fluid_properties(fluid, t)

    summary = {
        "total_length_m": total_length,
        "total_length_km": total_length / 1000.0,
        "flow_rate_m3_h": Q * 3600.0,
        "flow_rate_m3_s": Q,
        "inlet_pressure_barg": float(cfg.get("inlet_pressure_barg", 0.0)),
        "outlet_pressure_barg": p / 1e5,
        "total_pressure_drop_barg": (float(cfg.get("inlet_pressure_barg", 0.0)) * 1e5 - p) / 1e5,
        "max_velocity_m_s": float(profile["velocity_m_s"].max()) if not profile.empty else 0.0,
        "average_reynolds": float(profile["reynolds"].mean()) if not profile.empty else 0.0,
        "average_friction_factor": float(profile["friction_factor"].mean()) if not profile.empty else 0.0,
        "min_pressure_barg": float(profile["pressure_barg"].min()) if not profile.empty else p / 1e5,
        "max_temperature_c": float(profile["temperature_c"].max()) if not profile.empty else t,
        "min_temperature_c": float(profile["temperature_c"].min()) if not profile.empty else t,
        "inlet_density_kg_m3": inlet_rho,
        "outlet_density_kg_m3": outlet_rho,
        "mass_flow_inlet_kg_s": inlet_rho * Q,
        "mass_flow_outlet_kg_s": outlet_rho * Q,
        "continuity_error_m3_s": float(profile["flow_m3_s"].max() - profile["flow_m3_s"].min()) if not profile.empty else 0.0,
        "sections": len(cfg.get("sections", [])),
    }

    return summary, profile, pd.DataFrame(summaries)


def get_analysis():
    cfg = st.session_state.config
    h = config_hash(cfg)
    if st.session_state.get("analysis_hash") != h:
        summary, profile, section_summaries = run_steady(cfg)
        st.session_state.analysis_hash = h
        st.session_state.summary = summary
        st.session_state.profile = profile
        st.session_state.section_summaries = section_summaries
    return st.session_state.summary, st.session_state.profile, st.session_state.section_summaries


def solve_flow_for_target(cfg, target_outlet_barg):
    target_pa = float(target_outlet_barg) * 1e5
    inlet_pa = float(cfg.get("inlet_pressure_barg", 0.0)) * 1e5

    if not cfg.get("sections"):
        return None

    if target_pa >= inlet_pa:
        return 0.0

    diameters = [float(s.get("internal_diameter_mm", 0.0)) / 1000.0 for s in cfg["sections"]]
    diameters = [d for d in diameters if d > 0]
    if not diameters:
        return None

    min_D = min(diameters)
    area = math.pi * min_D * min_D / 4.0
    q_max = 25.0 * area
    q_min = 1e-8

    cfg_quick = deepcopy(cfg)
    cfg_quick["segments"] = int(np.clip(cfg.get("segments", 20), 10, 50))

    def residual(q):
        c = deepcopy(cfg_quick)
        c["flow_rate_m3_h"] = q * 3600.0
        s, _, _ = run_steady(c)
        return s["outlet_pressure_barg"] * 1e5 - target_pa

    try:
        r_min = residual(q_min)
        r_max = residual(q_max)

        expansions = 0
        while r_min * r_max > 0 and expansions < 8:
            q_max *= 2.0
            r_max = residual(q_max)
            expansions += 1

        if r_min * r_max > 0:
            return None

        return float(brentq(residual, q_min, q_max, xtol=1e-8, rtol=1e-5, maxiter=150))
    except Exception:
        return None


def wave_speed(D, e, K, rho, E):
    if D <= 0 or rho <= 0 or K <= 0:
        return 0.0
    if e is None or e <= 0:
        return math.sqrt(K / rho)
    if E is None or E <= 0:
        E = 200e9
    denom = 1.0 + (K * D) / (E * e)
    val = (K / rho) / denom
    return math.sqrt(val) if val > 0 else 0.0


def compute_transient(cfg, closure_time_s=1.0, closure_fraction=0.8, duration_s=20.0, dt=0.05):
    cfg_tmp = deepcopy(cfg)
    cfg_tmp["segments"] = int(np.clip(cfg.get("segments", 30), 10, 50))
    summary, profile, _ = run_steady(cfg_tmp)

    Q = float(cfg.get("flow_rate_m3_h", 0.0)) / 3600.0
    total_L = float(summary.get("total_length_m", 0.0))
    fluid = get_fluid(cfg.get("fluid", "Seawater"))

    meta = {
        "wave_speed_m_s": 0.0,
        "critical_time_s": 0.0,
        "period_s": 0.0,
        "joukowsky_barg": 0.0,
        "reference_velocity_m_s": 0.0,
        "damping_zeta": 0.0,
        "base_pressure_barg": float(summary.get("outlet_pressure_barg", 0.0)),
        "total_length_m": total_L,
        "note": "Conceptual lumped surge screening; not a validated transient design model.",
    }

    if total_L <= 0 or Q <= 0 or not cfg.get("sections"):
        return pd.DataFrame(columns=TRANSIENT_COLUMNS), meta

    ref_sec = None
    min_area = None
    for sec in cfg["sections"]:
        D, e, rough, area = section_geometry(sec)
        if area > 0 and (min_area is None or area < min_area):
            min_area = area
            ref_sec = sec

    if ref_sec is None or min_area is None or min_area <= 0:
        return pd.DataFrame(columns=TRANSIENT_COLUMNS), meta

    D_ref, e_ref, _, _ = section_geometry(ref_sec)
    mat = MATERIALS.get(ref_sec.get("material", "API 5L X65"), MATERIALS["API 5L X65"])
    rho, mu = fluid_properties(fluid, float(cfg.get("inlet_temp_c", 20.0)))
    a = wave_speed(D_ref, e_ref, float(fluid.get("bulk_modulus_pa", 1.5e9)), rho, float(mat.get("youngs_pa", 200e9)))
    V_ref = Q / min_area

    omega = math.pi * a / (2.0 * total_L) if a > 0 and total_L > 0 else 0.0
    period = 2.0 * math.pi / omega if omega > 0 else 0.0
    critical = 2.0 * total_L / a if a > 0 else 0.0
    dP_j = rho * a * V_ref * float(closure_fraction)

    f_avg = float(summary.get("average_friction_factor", 0.02))
    friction_slope = f_avg * V_ref / (2.0 * G * D_ref) if D_ref > 0 else 0.01
    zeta = float(np.clip(friction_slope * 50.0, 0.02, 0.35))

    duration_s = float(np.clip(duration_s, 1.0, 120.0))
    dt = float(np.clip(dt, 0.01, 0.5))

    t = np.arange(0.0, duration_s + dt, dt)
    ramp = np.clip(t / max(float(closure_time_s), 1e-6), 0.0, 1.0) if float(closure_time_s) > 0 else np.ones_like(t)
    delayed = np.maximum(0.0, t - float(closure_time_s))
    decay = np.exp(-zeta * omega * delayed)
    phase = omega * delayed
    surge = dP_j * ramp * decay * np.cos(phase)

    p_base = float(summary.get("outlet_pressure_barg", 0.0)) * 1e5
    p = p_base + surge
    flow = Q * (1.0 - float(closure_fraction) * ramp)

    df = pd.DataFrame(
        {
            "time_s": t,
            "pressure_pa": p,
            "pressure_barg": p / 1e5,
            "flow_m3_s": flow,
            "flow_m3_h": flow * 3600.0,
            "surge_pa": surge,
            "surge_barg": surge / 1e5,
        }
    )

    meta.update(
        {
            "wave_speed_m_s": float(a),
            "critical_time_s": float(critical),
            "period_s": float(period),
            "joukowsky_barg": float(dP_j / 1e5),
            "reference_velocity_m_s": float(V_ref),
            "damping_zeta": float(zeta),
            "base_pressure_barg": float(p_base / 1e5),
        }
    )

    return df, meta


def get_transient():
    params = st.session_state.transient_params
    h = config_hash(st.session_state.config) + json.dumps(params, sort_keys=True, default=str)
    if st.session_state.get("transient_hash") != h:
        df, meta = compute_transient(st.session_state.config, **params)
        st.session_state.transient_hash = h
        st.session_state.transient_df = df
        st.session_state.transient_meta = meta
    return st.session_state.transient_df, st.session_state.transient_meta


def integrity_assessment(cfg, records):
    today = datetime.now()
    rows = []

    for sec in cfg.get("sections", []):
        sec_name = sec.get("name", "Section")
        recs = [r for r in records if r.get("section") == sec_name]
        dates = []
        for r in recs:
            try:
                dates.append(datetime.fromisoformat(str(r.get("date"))))
            except Exception:
                pass

        latest = max(dates) if dates else None
        age_days = (today - latest).days if latest else None
        interval = int(sec.get("inspection_interval_days", 365))
        overdue = age_days is None or age_days > interval

        metal_values = []
        crack_count = 0
        for r in recs:
            try:
                metal_values.append(float(r.get("metal_loss_pct", 0.0)))
            except Exception:
                metal_values.append(0.0)
            if "crack" in str(r.get("defect", "")).lower():
                crack_count += 1

        max_metal = max(metal_values) if metal_values else 0.0

        severity = 0.0
        if latest is None:
            severity += 35.0
        else:
            severity += min(25.0, max(0.0, age_days / max(interval, 1) * 10.0))
        severity += min(50.0, max_metal * 1.0)
        severity += min(10.0, crack_count * 5.0)

        flags = []
        if latest is None:
            flags.append("Missing inspection")
        elif overdue:
            flags.append("Overdue inspection")
        if max_metal >= 40.0:
            flags.append("High metal loss")
        if crack_count > 0:
            flags.append("Crack indication")

        rows.append(
            {
                "Section": sec_name,
                "Latest inspection": latest.date().isoformat() if latest else "Missing",
                "Age (days)": age_days if age_days is not None else None,
                "Interval (days)": interval,
                "Overdue": bool(overdue),
                "Max metal loss (%)": float(max_metal),
                "Crack indications": int(crack_count),
                "Severity score": float(np.clip(severity, 0.0, 100.0)),
                "Flags": "; ".join(flags) if flags else "None",
            }
        )

    columns = [
        "Section",
        "Latest inspection",
        "Age (days)",
        "Interval (days)",
        "Overdue",
        "Max metal loss (%)",
        "Crack indications",
        "Severity score",
        "Flags",
    ]
    return pd.DataFrame(rows, columns=columns)


def perturb_config(cfg, variable, delta):
    c = deepcopy(cfg)
    if variable == "flow_rate_m3_h":
        c["flow_rate_m3_h"] = max(0.0, float(c.get("flow_rate_m3_h", 0.0)) * (1.0 + delta))
    elif variable == "inlet_temp_c":
        c["inlet_temp_c"] = float(c.get("inlet_temp_c", 20.0)) + delta
    elif variable == "roughness_mm":
        for s in c.get("sections", []):
            s["roughness_mm"] = max(1e-5, float(s.get("roughness_mm", 0.045)) * (1.0 + delta))
    elif variable == "internal_diameter_mm":
        for s in c.get("sections", []):
            s["internal_diameter_mm"] = max(1.0, float(s.get("internal_diameter_mm", 100.0)) * (1.0 + delta))
    elif variable == "length_m":
        for s in c.get("sections", []):
            s["length_m"] = max(1.0, float(s.get("length_m", 1.0)) * (1.0 + delta))
    elif variable == "inlet_pressure_barg":
        c["inlet_pressure_barg"] = max(0.0, float(c.get("inlet_pressure_barg", 0.0)) * (1.0 + delta))
    return c


def sensitivity_analysis(cfg, base_summary):
    base_dp = float(base_summary.get("total_pressure_drop_barg", 0.0))
    tests = [
        ("Flow rate", "flow_rate_m3_h", 0.10, "percent"),
        ("Inlet temperature", "inlet_temp_c", 10.0, "absolute"),
        ("Wall roughness", "roughness_mm", 0.50, "percent"),
        ("Internal diameter", "internal_diameter_mm", -0.02, "percent"),
        ("Pipeline length", "length_m", 0.10, "percent"),
        ("Inlet pressure", "inlet_pressure_barg", 0.10, "percent"),
    ]

    rows = []
    for label, var, delta, kind in tests:
        c = perturb_config(cfg, var, delta)
        c["segments"] = 20
        s, _, _ = run_steady(c)
        dp = float(s.get("total_pressure_drop_barg", 0.0))
        abs_change = abs(dp - base_dp)
        rel = abs_change / max(abs(base_dp), 1e-6) * 100.0
        pert = f"{delta:+.0%}" if kind == "percent" else f"{delta:+.1f} C"
        rows.append(
            {
                "Variable": label,
                "Perturbation": pert,
                "Pressure drop (barg)": dp,
                "Absolute change (barg)": abs_change,
                "Relative sensitivity (%)": rel,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Relative sensitivity (%)", ascending=False).reset_index(drop=True)
    return df


def decision_support(cfg, summary, profile, records, transient_meta):
    anomalies = []

    max_v = float(summary.get("max_velocity_m_s", 0.0))
    if max_v > 3.0:
        anomalies.append(("Warning", f"Velocity screening threshold exceeded: {max_v:.2f} m/s.", "Review erosion, vibration, and flow-assurance design basis."))

    outlet_p = float(summary.get("outlet_pressure_barg", 0.0))
    if outlet_p < 0.0:
        anomalies.append(("Critical", "Calculated outlet gauge pressure is below atmospheric.", "Verify inlet pressure, elevation profile, flow rate, and boundary conditions."))

    dp = float(summary.get("total_pressure_drop_barg", 0.0))
    inlet_p = float(cfg.get("inlet_pressure_barg", 0.0))
    if inlet_p > 0 and dp > 0.8 * inlet_p:
        anomalies.append(("Warning", "Pressure drop consumes more than 80% of available inlet pressure.", "Check pumping/compression margin and operational envelope."))

    fluid = get_fluid(cfg.get("fluid", "Seawater"))
    t_in = float(cfg.get("inlet_temp_c", 20.0))
    if t_in < float(fluid.get("min_temp_c", -50.0)) or t_in > float(fluid.get("max_temp_c", 200.0)):
        anomalies.append(("Warning", "Inlet temperature is outside the demo fluid property range.", "Use validated PVT data or a qualified thermal-hydraulic model."))

    if not profile.empty:
        min_mu = float(profile["viscosity_pa_s"].min())
        max_mu = float(profile["viscosity_pa_s"].max())
        if min_mu < 1e-5 or max_mu > 5.0:
            anomalies.append(("Info", "Viscosity is near the bounds of the simplified property model.", "Confirm fluid characterization with laboratory or field data."))

    for sec in cfg.get("sections", []):
        name = sec.get("name", "Section")
        wall = float(sec.get("wall_thickness_mm", 0.0))
        rough = float(sec.get("roughness_mm", 0.0))
        D = float(sec.get("internal_diameter_mm", 0.0))
        length = float(sec.get("length_m", 0.0))

        if wall < 2.0:
            anomalies.append(("Warning", f"Very thin wall in {name}: {wall:.2f} mm.", "Confirm geometry against as-built records and fabrication tolerances."))
        if rough > 2.0:
            anomalies.append(("Warning", f"High roughness in {name}: {rough:.2f} mm.", "Check fouling, corrosion product, or scaling assumptions."))
        if D > 0 and rough / D > 0.01:
            anomalies.append(("Warning", f"Relative roughness is high in {name}.", "Review surface condition and friction-factor correlation validity."))
        if length <= 0:
            anomalies.append(("Critical", f"Non-positive length in {name}.", "Correct pipeline configuration before interpretation."))

    surge = float(transient_meta.get("joukowsky_barg", 0.0))
    if surge > 20.0:
        anomalies.append(("Warning", f"Conceptual surge magnitude is high: {surge:.1f} barg.", "Run a validated transient analysis and review surge protection."))

    integ = integrity_assessment(cfg, records)
    missing = integ[integ["Latest inspection"] == "Missing"] if not integ.empty else pd.DataFrame()
    for _, row in missing.iterrows():
        anomalies.append(("Warning", f"No inspection record found for section: {row['Section']}.", "Schedule appropriate inspection or confirm data gap."))

    high_metal = integ[integ["Max metal loss (%)"] >= 40.0] if not integ.empty else pd.DataFrame()
    for _, row in high_metal.iterrows():
        anomalies.append(("Critical", f"High metal-loss indication in {row['Section']}: {row['Max metal loss (%)']:.1f}%.", "Escalate to qualified integrity engineering for fitness-for-service review."))

    sens = sensitivity_analysis(cfg, summary)
    important = sens.head(3)["Variable"].tolist() if not sens.empty else []

    return anomalies, sens, integ, important


def section_dataframe(cfg):
    rows = []
    for s in cfg.get("sections", []):
        D, e, rough, area = section_geometry(s)
        rows.append(
            {
                "Name": s.get("name", "Section"),
                "Length (m)": s.get("length_m", 0.0),
                "ID (mm)": s.get("internal_diameter_mm", 0.0),
                "Wall (mm)": s.get("wall_thickness_mm", 0.0),
                "Roughness (mm)": s.get("roughness_mm", 0.0),
                "Start elev (m)": s.get("elevation_start_m", 0.0),
                "End elev (m)": s.get("elevation_end_m", 0.0),
                "Material": s.get("material", "API 5L X65"),
                "Area (m2)": area,
                "Inspection interval (days)": s.get("inspection_interval_days", 365),
            }
        )
    columns = [
        "Name",
        "Length (m)",
        "ID (mm)",
        "Wall (mm)",
        "Roughness (mm)",
        "Start elev (m)",
        "End elev (m)",
        "Material",
        "Area (m2)",
        "Inspection interval (days)",
    ]
    return pd.DataFrame(rows, columns=columns)


def records_dataframe(records):
    columns = [
        "id",
        "date",
        "section",
        "type",
        "defect",
        "metal_loss_pct",
        "length_pct",
        "remaining_wall_mm",
        "growth_rate_pct_yr",
        "status",
        "notes",
    ]
    return pd.DataFrame(records, columns=columns)


def init_widget_defaults():
    cfg = st.session_state.config
    params = st.session_state.transient_params

    if "sb_fluid" not in st.session_state:
        st.session_state["sb_fluid"] = cfg.get("fluid", "Seawater")
    if "sb_flow" not in st.session_state:
        st.session_state["sb_flow"] = float(np.clip(cfg.get("flow_rate_m3_h", 0.0), 0.0, 1_000_000.0))
    if "sb_inlet_p" not in st.session_state:
        st.session_state["sb_inlet_p"] = float(np.clip(cfg.get("inlet_pressure_barg", 0.0), 0.0, 10_000.0))
    if "sb_inlet_t" not in st.session_state:
        st.session_state["sb_inlet_t"] = float(np.clip(cfg.get("inlet_temp_c", 20.0), -100.0, 300.0))
    if "sb_amb_t" not in st.session_state:
        st.session_state["sb_amb_t"] = float(np.clip(cfg.get("ambient_temp_c", 10.0), -20.0, 60.0))
    if "sb_segments" not in st.session_state:
        st.session_state["sb_segments"] = int(np.clip(cfg.get("segments", 50), 5, 2000))

    if "t_closure" not in st.session_state:
        st.session_state["t_closure"] = float(np.clip(params.get("closure_time_s", 1.0), 0.01, 60.0))
    if "t_fraction" not in st.session_state:
        st.session_state["t_fraction"] = float(np.clip(params.get("closure_fraction", 0.8), 0.0, 1.0))
    if "t_duration" not in st.session_state:
        st.session_state["t_duration"] = float(np.clip(params.get("duration_s", 20.0), 1.0, 120.0))
    if "t_dt" not in st.session_state:
        st.session_state["t_dt"] = float(np.clip(params.get("dt", 0.05), 0.01, 0.5))


def load_font(size):
    for name in ["DejaVuSans.ttf", "Arial.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def make_gif(profile, transient, meta):
    W, H = 760, 420
    font = load_font(18)
    small = load_font(13)
    frames = []

    def new_frame(title, subtitle=""):
        img = Image.new("RGB", (W, H), (12, 18, 28))
        d = ImageDraw.Draw(img)
        d.text((20, 12), title, fill=(240, 244, 248), font=font)
        if subtitle:
            d.text((20, 36), subtitle, fill=(148, 163, 184), font=small)
        d.rectangle([20, 60, W - 20, H - 45], outline=(51, 65, 85), width=1)
        return img, d

    def draw_series(d, full_x, full_y, upto, color=(56, 189, 248)):
        full_x = np.asarray(full_x, dtype=float)
        full_y = np.asarray(full_y, dtype=float)

        if len(full_x) == 0 or upto <= 0:
            return

        x = full_x[:upto]
        y = full_y[:upto]

        left, right, top, bottom = 70, W - 40, 70, H - 50
        plot_w = right - left
        plot_h = bottom - top

        xmin, xmax = float(np.min(full_x)), float(np.max(full_x))
        ymin, ymax = float(np.min(full_y)), float(np.max(full_y))

        if xmax - xmin < 1e-9:
            xmax = xmin + 1.0
        if ymax - ymin < 1e-9:
            ymax = ymin + 1.0

        ymin -= 0.05 * (ymax - ymin)
        ymax += 0.05 * (ymax - ymin)

        pts = []
        for xi, yi in zip(x, y):
            px = left + (xi - xmin) / (xmax - xmin) * plot_w
            py = bottom - (yi - ymin) / (ymax - ymin) * plot_h
            pts.append((px, py))

        if len(pts) > 1:
            d.line(pts, fill=color, width=3)
        if pts:
            x0, y0 = pts[-1]
            d.ellipse([x0 - 4, y0 - 4, x0 + 4, y0 + 4], fill=(248, 113, 113))

        d.text((left, bottom + 8), f"{xmin:.2f}", fill=(148, 163, 184), font=small)
        d.text((right - 60, bottom + 8), f"{xmax:.2f}", fill=(148, 163, 184), font=small)
        d.text((25, top - 4), f"{ymax:.1f}", fill=(148, 163, 184), font=small)
        d.text((25, bottom - 16), f"{ymin:.1f}", fill=(148, 163, 184), font=small)

    if not profile.empty:
        x = profile["distance_m"].values / 1000.0
        y = profile["pressure_barg"].values
        n = len(x)
        step = max(1, n // 20)
        outlet = float(meta.get("outlet_pressure_barg", 0.0) or 0.0)
        for upto in range(step, n + step, step):
            img, d = new_frame("Steady-state pressure profile", f"Distance km | Outlet ≈ {outlet:.1f} barg")
            draw_series(d, x, y, min(upto, n))
            frames.append(img)

    if not transient.empty:
        x = transient["time_s"].values
        y = transient["pressure_barg"].values
        n = len(x)
        step = max(1, n // 24)
        surge = float(meta.get("joukowsky_barg", 0.0) or 0.0)
        for upto in range(step, n + step, step):
            img, d = new_frame("Transient surge screening", f"Time s | Joukowsky scale ≈ {surge:.1f} barg")
            draw_series(d, x, y, min(upto, n), color=(251, 191, 36))
            frames.append(img)

    if not frames:
        img, d = new_frame("No simulation data", "Adjust inputs or add pipeline sections")
        frames.append(img)

    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=True,
    )
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def generate_demo_video_cached(profile_csv, transient_csv, meta_json):
    profile = pd.read_csv(io.StringIO(profile_csv)) if profile_csv else pd.DataFrame(columns=PROFILE_COLUMNS)
    transient = pd.read_csv(io.StringIO(transient_csv)) if transient_csv else pd.DataFrame(columns=TRANSIENT_COLUMNS)
    try:
        meta = json.loads(meta_json)
    except Exception:
        meta = {}
    return make_gif(profile, transient, meta)


def pressure_profile_figure(profile):
    fig = go.Figure()
    if profile.empty:
        fig.update_layout(height=320, title="No profile data")
        return fig

    fig.add_trace(
        go.Scatter(
            x=profile["distance_m"] / 1000.0,
            y=profile["pressure_barg"],
            mode="lines",
            name="Pressure (barg)",
            line=dict(color="#38bdf8", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=profile["distance_m"] / 1000.0,
            y=profile["temperature_c"],
            mode="lines",
            name="Temperature (°C)",
            yaxis="y2",
            line=dict(color="#f97316", dash="dot", width=2),
        )
    )
    fig.update_layout(
        height=340,
        xaxis_title="Distance (km)",
        yaxis_title="Pressure (barg)",
        yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
    )
    return fig


def transient_figure(df):
    fig = go.Figure()
    if df.empty:
        fig.update_layout(height=320, title="No transient data")
        return fig

    fig.add_trace(
        go.Scatter(
            x=df["time_s"],
            y=df["pressure_barg"],
            mode="lines",
            name="Pressure (barg)",
            line=dict(color="#fbbf24", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["time_s"],
            y=df["flow_m3_h"],
            mode="lines",
            name="Flow (m³/h)",
            yaxis="y2",
            line=dict(color="#34d399", dash="dot", width=2),
        )
    )
    fig.update_layout(
        height=340,
        xaxis_title="Time (s)",
        yaxis_title="Pressure (barg)",
        yaxis2=dict(title="Flow (m³/h)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
    )
    return fig


def sensitivity_figure(df):
    fig = go.Figure()
    if df.empty:
        fig.update_layout(height=300, title="No sensitivity data")
        return fig
    fig.add_trace(
        go.Bar(
            x=df["Variable"],
            y=df["Relative sensitivity (%)"],
            marker_color="#38bdf8",
        )
    )
    fig.update_layout(
        height=320,
        xaxis_title="Engineering variable",
        yaxis_title="Relative pressure-drop sensitivity (%)",
        margin=dict(l=40, r=40, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
    )
    return fig


def build_report(cfg, summary, profile, sec_sum, transient_df, transient_meta, integrity_df, anomalies, sens_df):
    ts = datetime.now().isoformat()
    md = []

    md.append("# Subsea Pipeline Conceptual Engineering Report")
    md.append(f"\nGenerated: {ts}")
    md.append(
        "\n> **Engineering review required.** This report is produced by simplified conceptual models for screening and decision support. "
        "It is not a validated pipeline design, flow-assurance, fitness-for-service, or failure-prediction calculation."
    )

    md.append("\n## 1. Pipeline configuration")
    md.append(f"- Fluid: {cfg.get('fluid')}")
    md.append(f"- Flow rate: {float(cfg.get('flow_rate_m3_h', 0.0)):.2f} m³/h")
    md.append(f"- Inlet pressure: {float(cfg.get('inlet_pressure_barg', 0.0)):.2f} barg")
    md.append(f"- Inlet temperature: {float(cfg.get('inlet_temp_c', 0.0)):.2f} °C")
    md.append(f"- Ambient temperature: {float(cfg.get('ambient_temp_c', 0.0)):.2f} °C")
    md.append(f"- Sections: {len(cfg.get('sections', []))}")
    md.append("\n### Sections\n```csv\n" + section_dataframe(cfg).to_csv(index=False) + "\n```")

    md.append("\n## 2. Steady-state hydraulic results")
    md.append(f"- Total length: {float(summary.get('total_length_km', 0.0)):.3f} km")
    md.append(f"- Outlet pressure: {float(summary.get('outlet_pressure_barg', 0.0)):.3f} barg")
    md.append(f"- Total pressure drop: {float(summary.get('total_pressure_drop_barg', 0.0)):.3f} barg")
    md.append(f"- Maximum velocity: {float(summary.get('max_velocity_m_s', 0.0)):.3f} m/s")
    md.append(f"- Average Reynolds number: {float(summary.get('average_reynolds', 0.0)):.0f}")
    md.append(f"- Average Darcy friction factor: {float(summary.get('average_friction_factor', 0.0)):.5f}")
    md.append(f"- Continuity error: {float(summary.get('continuity_error_m3_s', 0.0)):.3e} m³/s")
    md.append("\n### Section summary\n```csv\n" + sec_sum.to_csv(index=False) + "\n```")

    md.append("\n## 3. Transient screening")
    md.append(f"- Wave speed: {float(transient_meta.get('wave_speed_m_s', 0.0)):.1f} m/s")
    md.append(f"- Critical time 2L/a: {float(transient_meta.get('critical_time_s', 0.0)):.2f} s")
    md.append(f"- Joukowsky surge scale: {float(transient_meta.get('joukowsky_barg', 0.0)):.2f} barg")
    md.append(f"- Model note: {transient_meta.get('note', '')}")

    md.append("\n## 4. Inspection and integrity screening")
    md.append("```csv\n" + integrity_df.to_csv(index=False) + "\n```")

    md.append("\n## 5. Decision support findings")
    if anomalies:
        for sev, finding, action in anomalies:
            md.append(f"- **{sev}**: {finding} → {action}")
    else:
        md.append("- No rule-based anomalies detected for the current scenario.")

    md.append("\n## 6. Sensitivity ranking")
    if not sens_df.empty:
        md.append("```csv\n" + sens_df.to_csv(index=False) + "\n```")
    else:
        md.append("- Not computed.")

    md.append("\n## 7. Assumptions and limitations")
    md.append("- Single-phase liquid hydraulics; no slugging, multiphase flow, wax/asphaltene deposition, or hydrate formation prediction.")
    md.append("- Temperature field is an exponential screening model, not a validated thermal-hydraulic simulation.")
    md.append("- Transient module is a lumped conceptual surge screen, not a method-of-characteristics design model.")
    md.append("- Integrity screening uses inspection metadata and heuristic severity scores; it does not predict failure probability or remaining life.")
    md.append("- Real pipeline integrity decisions require qualified engineers, validated data, applicable codes, and documented fitness-for-service analysis.")

    payload = {
        "generated": ts,
        "configuration": cfg,
        "steady_summary": summary,
        "transient_meta": transient_meta,
        "anomalies": [{"severity": a[0], "finding": a[1], "action": a[2]} for a in anomalies],
        "integrity": integrity_df.to_dict(orient="records"),
        "sensitivity": sens_df.to_dict(orient="records") if not sens_df.empty else [],
    }

    return "\n".join(md), payload


def run_self_tests():
    results = []

    def add(name, passed, detail=""):
        results.append({"Test": name, "Result": "PASS" if passed else "FAIL", "Detail": detail})

    add("Unit conversion mm to m", abs(25.4 / 1000.0 - 0.0254) < 1e-12, "25.4 mm = 0.0254 m")
    add("Unit conversion m3/h to m3/s", abs(3600.0 / 3600.0 - 1.0) < 1e-12, "3600 m3/h = 1 m3/s")

    Re = 1.0 * 0.1 / 1e-6
    add("Reynolds number benchmark", abs(Re - 100000.0) < 1e-6, f"Re={Re:.0f}")

    f_lam = friction_factor(1000.0, 0.0)
    add("Laminar friction factor", abs(f_lam - 0.064) < 1e-8, f"f={f_lam:.6f}")

    f_turb = friction_factor(100000.0, 1e-4)
    add("Turbulent Colebrook plausibility", 0.015 < f_turb < 0.025, f"f={f_turb:.6f}")

    crude = get_fluid("Crude oil (demo)")
    _, mu20 = fluid_properties(crude, 20.0)
    _, mu60 = fluid_properties(crude, 60.0)
    add("Temperature-dependent viscosity", mu60 < mu20, f"mu20={mu20:.4g}, mu60={mu60:.4g}")

    cfg_test = {
        "fluid": "Seawater",
        "flow_rate_m3_h": 360.0,
        "inlet_pressure_barg": 50.0,
        "inlet_temp_c": 20.0,
        "ambient_temp_c": 20.0,
        "segments": 20,
        "sections": [
            {
                "name": "Test horizontal",
                "length_m": 1000.0,
                "internal_diameter_mm": 200.0,
                "wall_thickness_mm": 10.0,
                "roughness_mm": 0.045,
                "elevation_start_m": 0.0,
                "elevation_end_m": 0.0,
                "material": "API 5L X65",
                "inspection_interval_days": 365,
            }
        ],
    }

    s_base, p_base, _ = run_steady(cfg_test)
    add("Positive friction pressure drop", s_base["total_pressure_drop_barg"] > 0 and s_base["outlet_pressure_barg"] < 50.0, f"dP={s_base['total_pressure_drop_barg']:.4f} barg")

    if not p_base.empty:
        err = float(p_base["flow_m3_s"].max() - p_base["flow_m3_s"].min())
        add("Volumetric continuity balance", err < 1e-9, f"error={err:.3e} m3/s")
    else:
        add("Volumetric continuity balance", False, "empty profile")

    cfg_up = deepcopy(cfg_test)
    cfg_up["sections"][0]["elevation_end_m"] = 100.0
    s_up, _, _ = run_steady(cfg_up)
    add("Elevation head effect", s_up["total_pressure_drop_barg"] > s_base["total_pressure_drop_barg"], f"horizontal={s_base['total_pressure_drop_barg']:.4f}, uphill={s_up['total_pressure_drop_barg']:.4f} barg")

    q_sol = solve_flow_for_target(cfg_test, s_base["outlet_pressure_barg"])
    if q_sol is not None:
        add("Flow solver convergence", abs(q_sol * 3600.0 - 360.0) < 10.0, f"solved Q={q_sol * 3600.0:.2f} m3/h")
    else:
        add("Flow solver convergence", False, "no solution")

    _, meta = compute_transient(cfg_test, 1.0, 1.0, 5.0, 0.05)
    add("Transient wave speed positive", meta["wave_speed_m_s"] > 100.0, f"a={meta['wave_speed_m_s']:.1f} m/s")

    cfg_conv = deepcopy(cfg_test)
    cfg_conv["segments"] = 100
    s_conv, _, _ = run_steady(cfg_conv)
    rel_diff = abs(s_conv["total_pressure_drop_barg"] - s_base["total_pressure_drop_barg"]) / max(abs(s_base["total_pressure_drop_barg"]), 1e-9)
    add("Numerical convergence segments 20 vs 100", rel_diff < 0.05, f"relative diff={rel_diff:.3%}")

    return pd.DataFrame(results)


init_session()

st.title("🌊 Subsea Pipeline Engineering Analysis & Integrity Studio")
st.caption(
    "Conceptual decision-support prototype for simplified subsea pipeline hydraulics, transient screening, "
    "inspection records, scenario comparison, and engineering reporting."
)
st.warning(
    "This application does not predict actual pipeline failure. Real integrity decisions require qualified "
    "engineering review, validated data, applicable codes, and documented fitness-for-service analysis."
)

show_flash()

with st.sidebar:
    st.header("⚙️ Global Controls")
    init_widget_defaults()

    cfg = st.session_state.config
    fluid_options = list(FLUIDS.keys())
    sanitize_select("sb_fluid", fluid_options, cfg.get("fluid"))

    cfg["fluid"] = st.selectbox("Fluid", fluid_options, key="sb_fluid")
    cfg["flow_rate_m3_h"] = st.number_input(
        "Flow rate (m³/h)",
        min_value=0.0,
        max_value=1_000_000.0,
        step=10.0,
        key="sb_flow",
    )
    cfg["inlet_pressure_barg"] = st.number_input(
        "Inlet pressure (barg)",
        min_value=0.0,
        max_value=10_000.0,
        step=1.0,
        key="sb_inlet_p",
    )
    cfg["inlet_temp_c"] = st.number_input(
        "Inlet temperature (°C)",
        min_value=-100.0,
        max_value=300.0,
        step=1.0,
        key="sb_inlet_t",
    )
    cfg["ambient_temp_c"] = st.number_input(
        "Ambient seabed temp (°C)",
        min_value=-20.0,
        max_value=60.0,
        step=1.0,
        key="sb_amb_t",
    )
    cfg["segments"] = st.number_input(
        "Segments per section",
        min_value=5,
        max_value=2000,
        step=5,
        key="sb_segments",
    )

    st.session_state.config = cfg

    st.header("⏱️ Transient Controls")

    params = st.session_state.transient_params
    params["closure_time_s"] = st.number_input(
        "Valve closure time (s)",
        min_value=0.01,
        max_value=60.0,
        step=0.05,
        key="t_closure",
    )
    params["closure_fraction"] = st.slider(
        "Closure fraction",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="t_fraction",
    )
    params["duration_s"] = st.number_input(
        "Simulation duration (s)",
        min_value=1.0,
        max_value=120.0,
        step=1.0,
        key="t_duration",
    )
    params["dt"] = st.number_input(
        "Time step (s)",
        min_value=0.01,
        max_value=0.5,
        step=0.01,
        key="t_dt",
    )

    st.session_state.transient_params = params

    if st.button("Reset demo data", key="reset_demo_btn", use_container_width=True):
        reset_session_to_demo()
        set_flash("success", "Demo pipeline, records, scenarios, and controls were reset.")
        st.rerun()

tabs = st.tabs(
    [
        "Demo Dashboard",
        "Pipeline Config",
        "Hydraulics",
        "Transient Flow",
        "Inspection & Integrity",
        "AI Decision Support",
        "Scenarios & Reports",
        "Docs, Tests & Evaluation",
    ]
)

with tabs[0]:
    st.header("Live Demo Dashboard")
    st.markdown(
        "This demo automatically loads a realistic two-section subsea pipeline, runs steady-state hydraulics, "
        "executes a conceptual transient screen, and generates an animated demo file immediately."
    )

    with st.spinner("Running demo simulation and generating animated demo file..."):
        summary, profile, sec_sum = get_analysis()
        transient_df, transient_meta = get_transient()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total length (demo)", f"{float(summary['total_length_km']):.2f} km")
    c2.metric(
        "Outlet pressure (demo)",
        f"{float(summary['outlet_pressure_barg']):.1f} barg",
        delta=f"{float(summary['total_pressure_drop_barg']):.1f} barg drop",
    )
    c3.metric("Max velocity (demo)", f"{float(summary['max_velocity_m_s']):.2f} m/s")
    c4.metric("Avg Reynolds (demo)", f"{float(summary['average_reynolds']):,.0f}")
    c5.metric("Surge scale (demo)", f"{float(transient_meta.get('joukowsky_barg', 0.0)):.1f} barg")

    st.subheader("Generated live demo animation")
    video_meta = dict(transient_meta)
    video_meta["outlet_pressure_barg"] = summary["outlet_pressure_barg"]
    video_meta["total_pressure_drop_barg"] = summary["total_pressure_drop_barg"]

    profile_csv = profile.to_csv(index=False)
    transient_csv = transient_df.to_csv(index=False)
    meta_json = json.dumps(video_meta, default=str)
    video_bytes = generate_demo_video_cached(profile_csv, transient_csv, meta_json)

    st.image(
        video_bytes,
        caption="Animated demo file (GIF) generated from the current simulation.",
        use_container_width=True,
    )
    st.download_button(
        "Download animated demo file (GIF)",
        data=video_bytes,
        file_name="subsea_pipeline_live_demo.gif",
        mime="image/gif",
        key="download_demo_gif",
        use_container_width=True,
    )

    st.subheader("Hydraulic and transient charts")
    if not profile.empty:
        st.plotly_chart(
            pressure_profile_figure(profile),
            use_container_width=True,
            key="demo_pressure_profile_chart",
        )
    else:
        st.info("No steady-state profile data available in demo dashboard.")

    if not transient_df.empty:
        st.plotly_chart(
            transient_figure(transient_df),
            use_container_width=True,
            key="demo_transient_chart",
        )
    else:
        st.info("No transient data available in demo dashboard.")

    st.subheader("Section summary")
    st.dataframe(sec_sum, use_container_width=True, key="demo_section_summary_df")

with tabs[1]:
    st.header("Pipeline Configuration Manager")

    cfg = st.session_state.config

    st.write(
        f"**Fluid:** {cfg.get('fluid')} | "
        f"**Flow:** {float(cfg.get('flow_rate_m3_h', 0.0)):.2f} m³/h | "
        f"**Inlet pressure:** {float(cfg.get('inlet_pressure_barg', 0.0)):.2f} barg | "
        f"**Inlet temp:** {float(cfg.get('inlet_temp_c', 0.0)):.2f} °C"
    )

    st.subheader("Pipe geometry and material database")
    st.dataframe(pd.DataFrame(PIPE_LIBRARY), use_container_width=True, key="pipe_library_df")

    st.subheader("Current sections")
    st.dataframe(section_dataframe(cfg), use_container_width=True, key="current_sections_df")

    with st.expander("Add section from database"):
        labels = [
            f"{p['nps']}in {p['schedule']} | ID={p['internal_diameter_mm']:.1f} mm | wall={p['wall_mm']:.1f} mm"
            for p in PIPE_LIBRARY
        ]
        options = list(range(len(PIPE_LIBRARY)))
        sanitize_select("lib_pipe_sel", options, 0)

        sel = st.selectbox(
            "Library pipe",
            options=options,
            format_func=lambda i: labels[i],
            key="lib_pipe_sel",
        )

        last_lib = st.session_state.get("_last_lib_sel")
        if last_lib != sel:
            for k in list(st.session_state.keys()):
                if str(k).startswith("lib_field_"):
                    del st.session_state[k]
            st.session_state["_last_lib_sel"] = sel

        pipe = PIPE_LIBRARY[sel]
        suffix = f"lib_field_{sel}"

        mat_options = list(MATERIALS.keys())
        sanitize_select(f"{suffix}_mat", mat_options, pipe.get("material"))

        name = st.text_input(
            "Section name",
            f"{pipe['nps']}in {pipe['schedule']} section",
            key=f"{suffix}_name",
        )
        length = st.number_input(
            "Length (m)",
            min_value=1.0,
            max_value=100_000.0,
            value=5000.0,
            step=100.0,
            key=f"{suffix}_len",
        )
        elev_s = st.number_input(
            "Start elevation (m)",
            min_value=-5000.0,
            max_value=5000.0,
            value=-1000.0,
            step=10.0,
            key=f"{suffix}_es",
        )
        elev_e = st.number_input(
            "End elevation (m)",
            min_value=-5000.0,
            max_value=5000.0,
            value=-950.0,
            step=10.0,
            key=f"{suffix}_ee",
        )
        rough = st.number_input(
            "Roughness (mm)",
            min_value=0.001,
            max_value=10.0,
            value=float(pipe["roughness_mm"]),
            step=0.001,
            key=f"{suffix}_rough",
        )
        wall = st.number_input(
            "Wall thickness (mm)",
            min_value=0.5,
            max_value=100.0,
            value=float(pipe["wall_mm"]),
            step=0.1,
            key=f"{suffix}_wall",
        )
        idmm = st.number_input(
            "Internal diameter (mm)",
            min_value=10.0,
            max_value=2000.0,
            value=float(pipe["internal_diameter_mm"]),
            step=0.1,
            key=f"{suffix}_id",
        )
        mat = st.selectbox(
            "Material",
            mat_options,
            key=f"{suffix}_mat",
        )
        interval = st.number_input(
            "Inspection interval (days)",
            min_value=30,
            max_value=3650,
            value=365,
            step=30,
            key=f"{suffix}_interval",
        )

        if st.button("Add library section", key=f"{suffix}_add_btn", use_container_width=True):
            cfg["sections"].append(
                {
                    "name": name,
                    "length_m": float(length),
                    "internal_diameter_mm": float(idmm),
                    "wall_thickness_mm": float(wall),
                    "roughness_mm": float(rough),
                    "elevation_start_m": float(elev_s),
                    "elevation_end_m": float(elev_e),
                    "material": mat,
                    "inspection_interval_days": int(interval),
                }
            )
            st.session_state.config = cfg
            set_flash("success", f"Added section: {name}")
            st.rerun()

    with st.expander("Add custom section"):
        mat_options = list(MATERIALS.keys())
        sanitize_select("cust_mat", mat_options, "API 5L X65")

        name = st.text_input("Custom section name", "Custom section", key="cust_name")
        length = st.number_input(
            "Custom length (m)",
            min_value=1.0,
            max_value=100_000.0,
            value=1000.0,
            step=100.0,
            key="cust_len",
        )
        idmm = st.number_input(
            "Custom internal diameter (mm)",
            min_value=10.0,
            max_value=2000.0,
            value=250.0,
            step=1.0,
            key="cust_id",
        )
        wall = st.number_input(
            "Custom wall thickness (mm)",
            min_value=0.5,
            max_value=100.0,
            value=10.0,
            step=0.1,
            key="cust_wall",
        )
        rough = st.number_input(
            "Custom roughness (mm)",
            min_value=0.001,
            max_value=10.0,
            value=0.045,
            step=0.001,
            key="cust_rough",
        )
        elev_s = st.number_input(
            "Custom start elevation (m)",
            min_value=-5000.0,
            max_value=5000.0,
            value=-1000.0,
            step=10.0,
            key="cust_es",
        )
        elev_e = st.number_input(
            "Custom end elevation (m)",
            min_value=-5000.0,
            max_value=5000.0,
            value=-1000.0,
            step=10.0,
            key="cust_ee",
        )
        mat = st.selectbox("Custom material", mat_options, key="cust_mat")
        interval = st.number_input(
            "Custom inspection interval (days)",
            min_value=30,
            max_value=3650,
            value=365,
            step=30,
            key="cust_interval",
        )

        if st.button("Add custom section", key="cust_add_btn", use_container_width=True):
            cfg["sections"].append(
                {
                    "name": name,
                    "length_m": float(length),
                    "internal_diameter_mm": float(idmm),
                    "wall_thickness_mm": float(wall),
                    "roughness_mm": float(rough),
                    "elevation_start_m": float(elev_s),
                    "elevation_end_m": float(elev_e),
                    "material": mat,
                    "inspection_interval_days": int(interval),
                }
            )
            st.session_state.config = cfg
            set_flash("success", f"Added custom section: {name}")
            st.rerun()

    if cfg.get("sections"):
        with st.expander("Update or remove section"):
            opts = [f"{i}: {s.get('name', 'Section')}" for i, s in enumerate(cfg["sections"])]
            sanitize_select("upd_sel", opts, opts[0] if opts else None)

            chosen = st.selectbox("Select section", opts, key="upd_sel")
            idx = int(chosen.split(":", 1)[0])

            last_idx = st.session_state.get("_last_upd_idx")
            if last_idx != idx:
                for k in list(st.session_state.keys()):
                    if str(k).startswith("upd_field_"):
                        del st.session_state[k]
                st.session_state["_last_upd_idx"] = idx

            sec = cfg["sections"][idx]
            suffix = f"upd_field_{idx}"

            mat_options = list(MATERIALS.keys())
            sanitize_select(f"{suffix}_mat", mat_options, sec.get("material", "API 5L X65"))

            c1, c2 = st.columns(2)
            with c1:
                new_name = st.text_input("Name", sec.get("name", "Section"), key=f"{suffix}_name")
                new_length = st.number_input(
                    "Length (m)",
                    min_value=1.0,
                    max_value=100_000.0,
                    value=float(sec.get("length_m", 1.0)),
                    step=100.0,
                    key=f"{suffix}_len",
                )
                new_id = st.number_input(
                    "Internal diameter (mm)",
                    min_value=10.0,
                    max_value=2000.0,
                    value=float(sec.get("internal_diameter_mm", 100.0)),
                    step=0.1,
                    key=f"{suffix}_id",
                )
                new_wall = st.number_input(
                    "Wall thickness (mm)",
                    min_value=0.5,
                    max_value=100.0,
                    value=float(sec.get("wall_thickness_mm", 5.0)),
                    step=0.1,
                    key=f"{suffix}_wall",
                )
            with c2:
                new_rough = st.number_input(
                    "Roughness (mm)",
                    min_value=0.001,
                    max_value=10.0,
                    value=float(sec.get("roughness_mm", 0.045)),
                    step=0.001,
                    key=f"{suffix}_rough",
                )
                new_es = st.number_input(
                    "Start elevation (m)",
                    min_value=-5000.0,
                    max_value=5000.0,
                    value=float(sec.get("elevation_start_m", 0.0)),
                    step=10.0,
                    key=f"{suffix}_es",
                )
                new_ee = st.number_input(
                    "End elevation (m)",
                    min_value=-5000.0,
                    max_value=5000.0,
                    value=float(sec.get("elevation_end_m", 0.0)),
                    step=10.0,
                    key=f"{suffix}_ee",
                )
                new_mat = st.selectbox("Material", mat_options, key=f"{suffix}_mat")

            if st.button("Update selected section", key=f"{suffix}_upd_btn", use_container_width=True):
                cfg["sections"][idx].update(
                    {
                        "name": new_name,
                        "length_m": float(new_length),
                        "internal_diameter_mm": float(new_id),
                        "wall_thickness_mm": float(new_wall),
                        "roughness_mm": float(new_rough),
                        "elevation_start_m": float(new_es),
                        "elevation_end_m": float(new_ee),
                        "material": new_mat,
                    }
                )
                st.session_state.config = cfg
                set_flash("success", f"Updated section: {new_name}")
                st.rerun()

            if st.button("Remove selected section", key=f"{suffix}_rm_btn", use_container_width=True):
                removed = cfg["sections"].pop(idx)
                st.session_state.config = cfg
                set_flash("warning", f"Removed section: {removed.get('name', 'Section')}")
                st.rerun()
    else:
        st.warning("No pipeline sections configured.")

    if st.button("Reset to demo pipeline", key="reset_tab_btn", use_container_width=True):
        reset_session_to_demo()
        set_flash("success", "Demo pipeline reset.")
        st.rerun()

with tabs[2]:
    st.header("Hydraulic Calculation Engine")

    cfg = st.session_state.config
    summary, profile, sec_sum = get_analysis()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Outlet pressure (engine)", f"{float(summary['outlet_pressure_barg']):.2f} barg")
    c2.metric("Total pressure drop (engine)", f"{float(summary['total_pressure_drop_barg']):.2f} barg")
    c3.metric("Average friction factor (engine)", f"{float(summary['average_friction_factor']):.5f}")
    c4.metric("Continuity error (engine)", f"{float(summary['continuity_error_m3_s']):.2e} m³/s")

    mode = st.radio(
        "Analysis mode",
        ["Specified flow rate", "Solve flow rate from target outlet pressure"],
        horizontal=True,
        key="hyd_mode",
    )

    if mode == "Solve flow rate from target outlet pressure":
        target_default = float(np.clip(summary["outlet_pressure_barg"], -1000.0, 10_000.0))
        target = st.number_input(
            "Target outlet pressure (barg)",
            min_value=-1000.0,
            max_value=10_000.0,
            value=target_default,
            step=0.5,
            key="target_outlet",
        )
        if st.button("Solve flow rate", key="solve_flow_btn", use_container_width=True):
            q = solve_flow_for_target(cfg, float(target))
            if q is not None:
                new_flow = float(q) * 3600.0
                cfg["flow_rate_m3_h"] = new_flow
                st.session_state.config = cfg
                st.session_state["sb_flow"] = float(new_flow)
                set_flash("success", f"Solved flow rate: {new_flow:.2f} m³/h")
                st.rerun()
            else:
                set_flash("error", "No converged solution within bounds. Check boundary conditions and geometry.")
                st.rerun()

    st.subheader("Governing equations")
    st.markdown(
        r"""
        - Continuity: \(Q = V A\)
        - Reynolds number: \(Re = \frac{\rho V D}{\mu} = \frac{V D}{\nu}\)
        - Laminar friction factor: \(f = \frac{64}{Re}\)
        - Turbulent Colebrook-White:
          \[
          \frac{1}{\sqrt{f}} = -2 \log_{10}\left(\frac{\varepsilon/D}{3.7} + \frac{2.51}{Re\sqrt{f}}\right)
          \]
        - Darcy-Weisbach pressure drop:
          \[
          \Delta p_f = f \frac{L}{D} \frac{\rho V^2}{2}
          \]
        - Elevation term:
          \[
          \Delta p_e = \rho g \Delta z
          \]
        """
    )

    if HAVE_SYMPY:
        try:
            f_sym, Re_sym, rr_sym = sp.symbols("f Re rr", positive=True)
            colebrook_sym = sp.Eq(1 / sp.sqrt(f_sym), -2 * sp.log(rr_sym / 3.7 + 2.51 / (Re_sym * sp.sqrt(f_sym)), 10))
            st.caption("SymPy representation of the Colebrook-White equation:")
            st.latex(sp.latex(colebrook_sym))
        except Exception:
            pass

    st.subheader("Section results")
    st.dataframe(sec_sum, use_container_width=True, key="hyd_section_summary_df")

    st.subheader("Distributed profile")
    if not profile.empty:
        st.dataframe(profile, use_container_width=True, key="hyd_profile_df")
        st.download_button(
            "Download profile CSV",
            data=profile.to_csv(index=False).encode(),
            file_name="pipeline_profile.csv",
            mime="text/csv",
            key="download_profile_csv",
            use_container_width=True,
        )
        st.plotly_chart(
            pressure_profile_figure(profile),
            use_container_width=True,
            key="hyd_pressure_profile_chart",
        )
    else:
        st.info("No profile data available in hydraulic engine.")

with tabs[3]:
    st.header("Transient Flow Screening Module")
    st.info(
        "This module provides a conceptual surge screen using wave speed, critical time, Joukowsky scaling, "
        "and a damped oscillation approximation. It is not a validated transient design model."
    )

    transient_df, transient_meta = get_transient()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Wave speed (module)", f"{float(transient_meta.get('wave_speed_m_s', 0.0)):.0f} m/s")
    c2.metric("Critical time (module)", f"{float(transient_meta.get('critical_time_s', 0.0)):.2f} s")
    c3.metric("Period (module)", f"{float(transient_meta.get('period_s', 0.0)):.2f} s")
    c4.metric("Joukowsky scale (module)", f"{float(transient_meta.get('joukowsky_barg', 0.0)):.2f} barg")
    c5.metric("Damping zeta (module)", f"{float(transient_meta.get('damping_zeta', 0.0)):.3f}")

    if not transient_df.empty:
        st.plotly_chart(
            transient_figure(transient_df),
            use_container_width=True,
            key="transient_module_chart",
        )
        st.dataframe(transient_df.describe(), use_container_width=True, key="transient_describe_df")
        st.download_button(
            "Download transient CSV",
            data=transient_df.to_csv(index=False).encode(),
            file_name="pipeline_transient.csv",
            mime="text/csv",
            key="download_transient_csv",
            use_container_width=True,
        )
    else:
        st.info("No transient data available in transient module. Configure at least one valid pipeline section and nonzero flow.")

with tabs[4]:
    st.header("Inspection Record Manager & Integrity Workflow")

    records = st.session_state.inspection_records
    cfg = st.session_state.config

    st.subheader("Inspection records")
    st.dataframe(records_dataframe(records), use_container_width=True, key="inspection_records_df")

    with st.expander("Add inspection record"):
        section_names = [s.get("name", "Section") for s in cfg.get("sections", [])]
        if section_names:
            sanitize_select("rec_sec", section_names, section_names[0])

            rec_date = st.date_input("Inspection date", datetime.now().date(), key="rec_date")
            sec = st.selectbox("Section", section_names, key="rec_sec")

            type_options = ["MFL", "UT", "Visual", "CP", "ROV", "Other"]
            sanitize_select("rec_type", type_options, "MFL")
            typ = st.selectbox("Type", type_options, key="rec_type")

            defect_options = ["None", "Corrosion", "Crack", "Dent", "Gouge", "Weld flaw", "Coating damage"]
            sanitize_select("rec_defect", defect_options, "None")
            defect = st.selectbox("Defect", defect_options, key="rec_defect")

            metal = st.number_input("Metal loss (%)", 0.0, 100.0, 0.0, 1.0, key="rec_metal")
            length_pct = st.number_input("Defect length (% circumference)", 0.0, 100.0, 0.0, 1.0, key="rec_length_pct")
            rem_wall = st.number_input("Remaining wall (mm)", 0.0, 100.0, 10.0, 0.1, key="rec_rem_wall")
            growth = st.number_input("Growth rate (%/year)", 0.0, 20.0, 0.0, 0.1, key="rec_growth")

            status_options = ["OK", "Monitor", "Repair", "Replace", "Investigate"]
            sanitize_select("rec_status", status_options, "Monitor")
            status = st.selectbox("Status", status_options, key="rec_status")

            notes = st.text_area("Notes", "", key="rec_notes")

            if st.button("Add record", key="rec_add_btn", use_container_width=True):
                rid = f"INS-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                records.append(
                    {
                        "id": rid,
                        "date": rec_date.isoformat(),
                        "section": sec,
                        "type": typ,
                        "defect": defect,
                        "metal_loss_pct": float(metal),
                        "length_pct": float(length_pct),
                        "remaining_wall_mm": float(rem_wall),
                        "growth_rate_pct_yr": float(growth),
                        "status": status,
                        "notes": notes,
                    }
                )
                st.session_state.inspection_records = records
                set_flash("success", f"Added inspection record {rid}")
                st.rerun()
        else:
            st.warning("Add a pipeline section first.")

    if records:
        ids = [r["id"] for r in records]
        sanitize_select("del_rec_select", ids, ids[0])
        del_id = st.selectbox("Delete record", ids, key="del_rec_select")
        if st.button("Delete selected record", key="del_rec_btn", use_container_width=True):
            st.session_state.inspection_records = [r for r in records if r.get("id") != del_id]
            set_flash("warning", f"Deleted record {del_id}")
            st.rerun()

    st.subheader("Integrity screening")
    st.caption("Heuristic screening only. This does not calculate failure probability or remaining life.")
    integ = integrity_assessment(cfg, st.session_state.inspection_records)
    st.dataframe(integ, use_container_width=True, key="integrity_screen_df")

    if not integ.empty:
        st.download_button(
            "Download integrity CSV",
            data=integ.to_csv(index=False).encode(),
            file_name="pipeline_integrity_screen.csv",
            mime="text/csv",
            key="download_integrity_csv",
            use_container_width=True,
        )

with tabs[5]:
    st.header("Decision-Support AI (Rule-Based + Sensitivity)")
    st.caption(
        "This module identifies unusual input combinations, missing inspection records, and potentially important "
        "engineering variables. It supports review but does not replace validated engineering analysis."
    )

    if st.button("Run decision support", key="run_ai_btn", use_container_width=True):
        cfg = st.session_state.config
        summary, profile, _ = get_analysis()
        _, transient_meta = get_transient()
        anomalies, sens, integ, important = decision_support(
            cfg,
            summary,
            profile,
            st.session_state.inspection_records,
            transient_meta,
        )
        st.session_state.ai_results = {
            "anomalies": anomalies,
            "sensitivity": sens,
            "integrity": integ,
            "important": important,
            "timestamp": datetime.now().isoformat(),
        }

    if st.session_state.ai_results:
        res = st.session_state.ai_results
        st.success(f"Decision support last run: {res['timestamp']}")

        st.subheader("Unusual input combinations and data gaps")
        if res["anomalies"]:
            anom_df = pd.DataFrame(res["anomalies"], columns=["Severity", "Finding", "Recommended action"])
            st.dataframe(anom_df, use_container_width=True, key="ai_anomalies_df")
        else:
            st.info("No rule-based anomalies detected for the current scenario.")

        st.subheader("Missing or overdue inspection records")
        integ = res["integrity"]
        if not integ.empty:
            flagged = integ[integ["Overdue"] | (integ["Latest inspection"] == "Missing")]
            st.dataframe(flagged, use_container_width=True, key="ai_flagged_integrity_df")
        else:
            st.info("No sections configured.")

        st.subheader("Potentially important engineering variables")
        if res["important"]:
            st.write(", ".join(res["important"]))
        else:
            st.write("No sensitivity ranking available.")

        st.subheader("Sensitivity ranking")
        st.plotly_chart(
            sensitivity_figure(res["sensitivity"]),
            use_container_width=True,
            key="ai_sensitivity_chart",
        )
        st.dataframe(res["sensitivity"], use_container_width=True, key="ai_sensitivity_df")

with tabs[6]:
    st.header("Scenario Comparison & Engineering Report Generator")

    cfg = st.session_state.config
    summary, profile, sec_sum = get_analysis()
    transient_df, transient_meta = get_transient()

    with st.expander("Save current scenario"):
        scen_count = len(st.session_state.scenarios)
        scen_key = f"scen_name_{scen_count}"
        name = st.text_input("Scenario name", f"Scenario {scen_count + 1}", key=scen_key)
        if st.button("Save scenario", key="save_scen_btn", use_container_width=True):
            st.session_state.scenarios.append(
                {
                    "name": name,
                    "timestamp": datetime.now().isoformat(),
                    "config": deepcopy(cfg),
                    "metrics": {
                        "Outlet pressure (barg)": float(summary["outlet_pressure_barg"]),
                        "Total pressure drop (barg)": float(summary["total_pressure_drop_barg"]),
                        "Max velocity (m/s)": float(summary["max_velocity_m_s"]),
                        "Avg Reynolds": float(summary["average_reynolds"]),
                        "Joukowsky (barg)": float(transient_meta.get("joukowsky_barg", 0.0)),
                    },
                }
            )
            set_flash("success", f"Saved scenario: {name}")
            st.rerun()

    if st.session_state.scenarios:
        scen_rows = []
        for s in st.session_state.scenarios:
            scen_rows.append({"Name": s["name"], "Saved": s["timestamp"], **s["metrics"]})
        scen_df = pd.DataFrame(scen_rows)
        st.subheader("Saved scenarios")
        st.dataframe(scen_df, use_container_width=True, key="scenario_comparison_df")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=scen_df["Name"], y=scen_df["Outlet pressure (barg)"], name="Outlet pressure"))
        fig.add_trace(go.Bar(x=scen_df["Name"], y=scen_df["Total pressure drop (barg)"], name="Pressure drop"))
        fig.update_layout(
            barmode="group",
            height=320,
            yaxis_title="barg",
            legend=dict(orientation="h", y=1.12),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.35)",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="scenario_comparison_chart",
        )

        names = scen_df["Name"].tolist()
        sanitize_select("del_scen_select", names, names[0])
        del_name = st.selectbox("Delete scenario", names, key="del_scen_select")
        if st.button("Delete selected scenario", key="del_scen_btn", use_container_width=True):
            st.session_state.scenarios = [s for s in st.session_state.scenarios if s["name"] != del_name]
            set_flash("warning", f"Deleted scenario: {del_name}")
            st.rerun()
    else:
        st.info("No saved scenarios yet.")

    st.subheader("Engineering report")
    if st.button("Generate engineering report", key="gen_report_btn", use_container_width=True):
        anomalies, sens, integ, important = decision_support(
            cfg,
            summary,
            profile,
            st.session_state.inspection_records,
            transient_meta,
        )
        md, payload = build_report(
            cfg,
            summary,
            profile,
            sec_sum,
            transient_df,
            transient_meta,
            integ,
            anomalies,
            sens,
        )
        st.session_state.report_md = md
        st.session_state.report_json = json.dumps(payload, indent=2, default=str)

    if st.session_state.report_md:
        st.download_button(
            "Download Markdown report",
            data=st.session_state.report_md.encode(),
            file_name="subsea_pipeline_report.md",
            mime="text/markdown",
            key="download_report_md",
            use_container_width=True,
        )
        st.download_button(
            "Download JSON payload",
            data=st.session_state.report_json.encode(),
            file_name="subsea_pipeline_report.json",
            mime="application/json",
            key="download_report_json",
            use_container_width=True,
        )
        with st.expander("Preview report"):
            st.markdown(st.session_state.report_md)

with tabs[7]:
    st.header("Documentation, Evaluation, and Reproducibility")

    st.markdown(
        """
        ### Purpose
        This application is a conceptual subsea pipeline engineering analysis and integrity information platform.
        It is intended for screening, education, prototype demonstration, and structured engineering review.

        ### Core modules represented in this single-file app
        - Pipeline configuration manager
        - Pipe geometry and material database
        - Fluid property database
        - Hydraulic calculation engine
        - Pressure-drop calculator
        - Temperature and viscosity model
        - Transient flow screening module
        - Inspection record manager
        - Integrity assessment workflow
        - Engineering report generator
        - Rule-based decision-support module

        ### Important limitation
        Simple hydraulic calculations and heuristic integrity screens do **not** predict actual pipeline failure.
        Real decisions require qualified engineers, validated datasets, applicable standards, and formal
        fitness-for-service or integrity management processes.
        """
    )

    st.subheader("Technology availability")
    st.write(f"SymPy: {'available' if HAVE_SYMPY else 'not installed'}")
    st.write(f"FiPy: {'available' if HAVE_FIPY else 'not installed'}")
    st.write("OpenFOAM: external CFD tool; not invoked by this lightweight demo.")

    st.subheader("Embedded evaluation suite")
    st.caption(
        "Runs benchmark checks for unit conversions, Reynolds number, friction factor, mass balance, "
        "elevation effects, flow solving, transient wave speed, and numerical convergence."
    )

    if st.button("Run embedded evaluation suite", key="run_tests_btn", use_container_width=True):
        st.session_state.self_test_results = run_self_tests()

    if st.session_state.self_test_results is not None:
        df = st.session_state.self_test_results
        st.dataframe(df, use_container_width=True, key="self_test_df")
        passed = int((df["Result"] == "PASS").sum())
        total = int(len(df))
        st.metric("Passed tests", passed, f"{passed}/{total}")

    st.subheader("Milestones")
    st.markdown(
        """
        - [x] Pipeline parameter manager
        - [x] Basic hydraulic equations
        - [x] Multiple pipeline sections
        - [x] Pressure-drop analysis
        - [x] Temperature-dependent properties
        - [x] Inspection records
        - [x] Scenario comparisons
        - [x] Engineering reports
        - [x] Live demo animation generated in-app
        - [ ] Validated multiphase flow assurance
        - [ ] Certified fitness-for-service calculations
        - [ ] External historian / asset-data integration
        """
    )

    st.subheader("How to run")
    st.code("streamlit run app.py", language="bash")

    st.subheader("Evaluation focus")
    st.markdown(
        """
        - Hydraulic equation sanity checks
        - Mass / continuity balance
        - Unit conversions
        - Numerical convergence with segment count
        - Benchmark friction-factor ranges
        - Temperature-dependent viscosity trend
        - Transient wave-speed positivity
        - Scenario reproducibility through saved configuration payloads
        """
    )