# ============================================
# ICFES Córdoba — Dash (UPLOAD + OPTIMIZADO)
# Tabs:
#   - Q1 Brechas territoriales
#   - Q2 Socioeconómico + riesgo
#   - Q3 Edad + riesgo
# NO Tukey / NO ANOVA
# + UI mejorada
# + Violin plot en Q1
# + Q2 elimina "SIN ESTRATO"
# + Heatmap con valores + colores pasteles
# ============================================

import base64, io, warnings
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.express as px

warnings.filterwarnings("ignore")

# ---------- UI / estilo ----------
THEME = dbc.themes.FLATLY
PLOT_TEMPLATE = "plotly_white"

MAX_SAMPLE = 2500  # baja a 1500 si tu PC sufre
COLOR_ZONA = {"URBANO": "#2E86AB", "RURAL": "#F18F01"}

AREAS = {
    "punt_global": "Puntaje Global",
    "punt_matematicas": "Matemáticas",
    "punt_lectura_critica": "Lectura Crítica",
    "punt_sociales_ciudadanas": "Sociales",
    "punt_c_naturales": "C. Naturales",
    "punt_ingles": "Inglés"
}

# ====== Memoria servidor (rápido) ======
DATA = {"df": None, "version": 0, "filename": None}

# ------------------ utils ------------------
def sample_for_plot(d: pd.DataFrame, seed=1) -> pd.DataFrame:
    if d is None or len(d) == 0:
        return d
    if len(d) <= MAX_SAMPLE:
        return d
    return d.sample(n=MAX_SAMPLE, random_state=seed)

def norm_txt(x):
    if pd.isna(x):
        return x
    return str(x).strip().upper()

def kpi_card(title, value, subtitle=""):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-muted", style={"fontSize": "0.9rem"}),
            html.Div(value, style={"fontSize": "1.7rem", "fontWeight": "900"}),
            html.Div(subtitle, className="text-muted", style={"fontSize": "0.85rem"})
        ]),
        className="shadow-sm",
        style={"borderRadius": "16px"}
    )

def make_table(df_small, max_rows=12):
    if df_small is None or len(df_small) == 0:
        return dbc.Alert("Sin datos para mostrar.", color="secondary")
    show = df_small.head(max_rows).copy()
    return dbc.Table.from_dataframe(show, striped=True, bordered=False, hover=True, size="sm")

def parse_contents(contents, filename):
    if contents is None:
        raise ValueError("No se recibió contenido.")
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    fname = (filename or "").lower()

    if fname.endswith(".csv"):
        try:
            return pd.read_csv(io.StringIO(decoded.decode("utf-8")), low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(io.StringIO(decoded.decode("latin-1")), low_memory=False)

    if fname.endswith((".xls", ".xlsx")):
        return pd.read_excel(io.BytesIO(decoded))

    raise ValueError("Formato no soportado. Sube un .csv o .xlsx")

# ====== Edad (robusta a 'periodo'=20224) ======
def infer_exam_year(d: pd.DataFrame) -> Optional[pd.Series]:
    # periodo típico: 20224 -> 2022
    if "periodo" in d.columns:
        s = d["periodo"].astype(str).str.extract(r"(\d{4})")[0]
        return pd.to_numeric(s, errors="coerce")
    for c in ["estu_anio", "anio", "year"]:
        if c in d.columns:
            return pd.to_numeric(d[c], errors="coerce")
    return None

def infer_birth_year(d: pd.DataFrame) -> Optional[pd.Series]:
    for c in ["estu_fechanacimiento", "fecha_nacimiento", "birth_date"]:
        if c in d.columns:
            dt = pd.to_datetime(d[c], errors="coerce")
            return dt.dt.year
    for c in ["estu_anionacimiento", "anio_nacimiento", "birth_year"]:
        if c in d.columns:
            return pd.to_numeric(d[c], errors="coerce")
    return None

def add_age_features(d: pd.DataFrame) -> pd.DataFrame:
    exam_year = infer_exam_year(d)
    birth_year = infer_birth_year(d)

    d2 = d.copy()
    d2["anio_presentacion"] = exam_year if exam_year is not None else np.nan
    d2["anio_nacimiento"] = birth_year if birth_year is not None else np.nan
    d2["edad"] = d2["anio_presentacion"] - d2["anio_nacimiento"]

    # limpia outliers
    d2.loc[(d2["edad"] < 10) | (d2["edad"] > 60), "edad"] = np.nan

    bins = [10, 14, 16, 18, 20, 25, 60]
    labels = ["10–13", "14–15", "16–17", "18–19", "20–24", "25+"]
    d2["edad_grupo"] = pd.cut(d2["edad"], bins=bins, labels=labels, include_lowest=True)
    return d2

def compute_low_perf_flag(d: pd.DataFrame, area: str) -> pd.Series:
    q25 = d[area].quantile(0.25)
    return (d[area] <= q25).astype(int)

# ====== Limpieza ======
def clean_df(dff: pd.DataFrame) -> pd.DataFrame:
    d = dff.copy()

    # normaliza textos clave
    for col in [
        "cole_mcpio_ubicacion", "cole_area_ubicacion", "cole_naturaleza",
        "fami_estratovivienda", "fami_educacionmadre", "fami_educacionpadre"
    ]:
        if col in d.columns:
            d[col] = d[col].map(norm_txt)

    # puntajes a num
    for c in AREAS:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    if "punt_global" in d.columns:
        d = d[d["punt_global"].notna() & (d["punt_global"] > 0)].copy()

    # reduce memoria
    for c in [
        "cole_mcpio_ubicacion", "cole_area_ubicacion", "cole_naturaleza",
        "fami_estratovivienda", "fami_educacionmadre", "fami_educacionpadre"
    ]:
        if c in d.columns:
            d[c] = d[c].astype("category")

    d = add_age_features(d)
    return d

# ====== Filtro ======
def _as_tuple(x):
    if x is None:
        return tuple()
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return tuple(x)
    return (x,)

def filter_df(base: pd.DataFrame, munis, zonas, nats, estratos, edu_m, edu_p, area: str) -> pd.DataFrame:
    d = base
    if munis and "cole_mcpio_ubicacion" in d.columns:
        d = d[d["cole_mcpio_ubicacion"].isin(munis)]
    if zonas and "cole_area_ubicacion" in d.columns:
        d = d[d["cole_area_ubicacion"].isin(zonas)]
    if nats and "cole_naturaleza" in d.columns:
        d = d[d["cole_naturaleza"].isin(nats)]
    if estratos and "fami_estratovivienda" in d.columns:
        d = d[d["fami_estratovivienda"].isin(estratos)]
    if edu_m and "fami_educacionmadre" in d.columns:
        d = d[d["fami_educacionmadre"].isin(edu_m)]
    if edu_p and "fami_educacionpadre" in d.columns:
        d = d[d["fami_educacionpadre"].isin(edu_p)]
    d = d[d[area].notna()]
    return d

# ====== Cache agregados por TAB (rápido) ======
@lru_cache(maxsize=256)
def cached_q1(version: int, area: str, munis: Tuple, zonas: Tuple, nats: Tuple, estratos: Tuple, edu_m: Tuple, edu_p: Tuple):
    base = DATA["df"]
    d = filter_df(base, list(munis), list(zonas), list(nats), list(estratos), list(edu_m), list(edu_p), area)
    if "cole_mcpio_ubicacion" not in d.columns:
        return pd.DataFrame(), pd.DataFrame()

    muni_stats = (
        d.groupby("cole_mcpio_ubicacion", observed=True)[area]
         .agg(media="mean", n="size")
         .reset_index()
         .sort_values("media", ascending=False)
    )

    muni_desc = (
        d.groupby("cole_mcpio_ubicacion", observed=True)[area]
         .agg(N="size", Promedio="mean", Std="std")
         .reset_index()
         .sort_values("Promedio")
    )
    muni_desc["Promedio"] = muni_desc["Promedio"].round(1)
    muni_desc["Std"] = muni_desc["Std"].round(1)
    return muni_stats, muni_desc

@lru_cache(maxsize=256)
def cached_q2(version: int, area: str,
              munis: Tuple, zonas: Tuple, nats: Tuple, estratos: Tuple, edu_m: Tuple, edu_p: Tuple):

    base = DATA["df"]
    d = filter_df(base, list(munis), list(zonas), list(nats), list(estratos), list(edu_m), list(edu_p), area)

    needed = ["fami_estratovivienda", "fami_educacionmadre", area]
    missing = [c for c in needed if c not in d.columns]
    if missing:
        return {"ok": False, "error": f"Faltan columnas: {', '.join(missing)}"}

    d_se = d.dropna(subset=["fami_estratovivienda", "fami_educacionmadre", area]).copy()
    if d_se.empty:
        return {"ok": True, "d_se": None, "riesgo": pd.DataFrame(), "mean": pd.DataFrame(), "pivot": None}

    # --- limpieza robusta de estrato ---
    estr = d_se["fami_estratovivienda"].astype(str).str.strip().str.upper()
    d_se = d_se[~estr.isin(["SIN ESTRATO", "SIN_ESTRATO", "NO APLICA", "N/A", "NA", "", "NONE", "NAN"])].copy()

    # si tras filtrar queda vacío
    if d_se.empty:
        return {"ok": True, "d_se": None, "riesgo": pd.DataFrame(), "mean": pd.DataFrame(), "pivot": None}

    # bandera de riesgo: 25% más bajo
    d_se["bajo_desempeno"] = compute_low_perf_flag(d_se, area)

    riesgo_estrato = (
        d_se.groupby("fami_estratovivienda", observed=True)["bajo_desempeno"]
        .mean().mul(100).reset_index(name="riesgo_pct")
        .sort_values("riesgo_pct", ascending=False)
    )

    mean_estrato = (
        d_se.groupby("fami_estratovivienda", observed=True)[area]
        .agg(Promedio="mean", N="size").reset_index()
        .sort_values("Promedio")
    )
    mean_estrato["Promedio"] = mean_estrato["Promedio"].round(1)

    # pivote estable para heatmap (promedio del puntaje)
    pivot = (
        d_se.pivot_table(
            index="fami_estratovivienda",
            columns="fami_educacionmadre",
            values=area,
            aggfunc="mean"
        )
        .sort_index()
    )

    return {"ok": True, "d_se": d_se, "riesgo": riesgo_estrato, "mean": mean_estrato, "pivot": pivot}

@lru_cache(maxsize=256)
def cached_q3(version: int, area: str, munis: Tuple, zonas: Tuple, nats: Tuple, estratos: Tuple, edu_m: Tuple, edu_p: Tuple):
    base = DATA["df"]
    d = filter_df(base, list(munis), list(zonas), list(nats), list(estratos), list(edu_m), list(edu_p), area)

    if "edad" not in d.columns or d["edad"].dropna().empty:
        return None

    d_age = d.dropna(subset=["edad", area]).copy()
    if d_age.empty:
        return (pd.DataFrame(), pd.DataFrame(), np.nan)

    corr_age = float(d_age[["edad", area]].corr().iloc[0, 1])

    d_age["bajo_desempeno"] = compute_low_perf_flag(d_age, area)

    riesgo_age = (
        d_age.groupby("edad_grupo", observed=True)["bajo_desempeno"]
        .mean().mul(100).reset_index(name="riesgo_pct").dropna()
    )

    mean_age = (
        d_age.groupby("edad_grupo", observed=True)[area]
        .agg(Promedio="mean", N="size").reset_index().dropna()
    )
    mean_age["Promedio"] = mean_age["Promedio"].round(1)

    return riesgo_age, mean_age, corr_age

# ==============================
# APP
# ==============================
app = dash.Dash(__name__, external_stylesheets=[THEME])
app.title = "ICFES Córdoba — Tablero"

tabs = dbc.Tabs(
    [
        dbc.Tab(label="Brechas territoriales (Q1)", tab_id="tab-q1"),
        dbc.Tab(label="Socioeconómico + riesgo (Q2)", tab_id="tab-q2"),
        dbc.Tab(label="Edad + riesgo (Q3)", tab_id="tab-age"),
    ],
    id="tabs",
    active_tab="tab-q1"
)

sidebar = dbc.Card(
    dbc.CardBody([
        html.Div("ICFES Córdoba", style={"fontWeight": "900", "fontSize": "1.2rem"}),
        html.Div("Brechas territoriales, socioeconómicas y por edad.", className="text-muted"),
        html.Hr(),

        html.Div("1) Sube tu archivo (CSV o Excel)", className="fw-semibold"),
        dcc.Upload(
            id="upload-data",
            children=dbc.Button("📤 Subir archivo", color="secondary", className="w-100"),
            multiple=False
        ),
        html.Div(id="upload-status", className="text-muted", style={"fontSize": "0.9rem", "marginTop": "8px"}),
        dcc.Store(id="store-version"),
        html.Hr(),

        html.Div("Área / puntaje", className="fw-semibold"),
        dcc.Dropdown(
            id="f-area",
            options=[{"label": v, "value": k} for k, v in AREAS.items()],
            value="punt_global",
            clearable=False
        ),
        html.Br(),

        html.Div("Municipio (opcional)", className="fw-semibold"),
        dcc.Dropdown(id="f-muni", options=[], multi=True, placeholder="Todos"),
        html.Br(),

        html.Div("Zona (urbano/rural)", className="fw-semibold"),
        dbc.Checklist(id="f-zona", options=[], value=[], inline=False),
        html.Br(),

        html.Div("Naturaleza del colegio", className="fw-semibold"),
        dcc.Dropdown(id="f-nat", options=[], multi=True, placeholder="Todas"),
        html.Hr(),

        html.Div("Socioeconómico (opcional)", className="fw-semibold"),
        html.Div("Estrato", className="text-muted", style={"fontSize": "0.9rem"}),
        dcc.Dropdown(id="f-estrato", options=[], multi=True, placeholder="Todos"),
        html.Br(),

        html.Div("Educación madre", className="text-muted", style={"fontSize": "0.9rem"}),
        dcc.Dropdown(id="f-edu-m", options=[], multi=True, placeholder="Todas"),
        html.Br(),

        html.Div("Educación padre", className="text-muted", style={"fontSize": "0.9rem"}),
        dcc.Dropdown(id="f-edu-p", options=[], multi=True, placeholder="Todas"),
        html.Br(),

        dbc.Button("Aplicar filtros", id="btn", color="primary", className="w-100")
    ]),
    className="shadow-sm",
    style={"borderRadius": "18px", "position": "sticky", "top": "14px"}
)

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Brechas Educativas — Córdoba", style={"fontWeight": "900"}),
        dbc.Badge("Analítica computacional para la toma de decisiones", color="secondary", className="ms-2")
    ]),
    color="white",
    dark=False,
    className="shadow-sm",
    style={"borderRadius": "18px", "marginBottom": "12px"}
)

app.layout = dbc.Container([
    navbar,
    dbc.Row([
        dbc.Col(sidebar, width=3),
        dbc.Col([
            dbc.Row([
                dbc.Col(html.Div(id="kpi-n"), md=3),
                dbc.Col(html.Div(id="kpi-mean"), md=3),
                dbc.Col(html.Div(id="kpi-std"), md=3),
                dbc.Col(html.Div(id="kpi-gap"), md=3),
            ], className="g-3"),

            html.Br(),
            dbc.Alert(id="insight", color="secondary", className="shadow-sm"),
            tabs,
            html.Br(),
            html.Div(id="contenido")
        ], width=9),
    ], className="g-3")
], fluid=True)

# ==============================
# CALLBACK: CARGA ARCHIVO
# ==============================
@app.callback(
    Output("upload-status", "children"),
    Output("store-version", "data"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True
)
def cargar_archivo(contents, filename):
    try:
        dff = parse_contents(contents, filename)
        dff = clean_df(dff)

        DATA["df"] = dff
        DATA["filename"] = filename
        DATA["version"] += 1

        cached_q1.cache_clear()
        cached_q2.cache_clear()
        cached_q3.cache_clear()

        msg = f"✅ Cargado: {filename} | Filas: {len(dff):,}".replace(",", ".")
        return msg, {"version": DATA["version"]}
    except Exception as e:
        DATA["df"] = None
        DATA["filename"] = None
        DATA["version"] += 1
        cached_q1.cache_clear()
        cached_q2.cache_clear()
        cached_q3.cache_clear()
        return f"❌ Error leyendo archivo: {e}", {"version": DATA["version"]}

# ==============================
# CALLBACK: POBLAR FILTROS
# ==============================
@app.callback(
    Output("f-muni", "options"),
    Output("f-zona", "options"),
    Output("f-zona", "value"),
    Output("f-nat", "options"),
    Output("f-estrato", "options"),
    Output("f-edu-m", "options"),
    Output("f-edu-p", "options"),
    Input("store-version", "data"),
    prevent_initial_call=True
)
def poblar_filtros(_data):
    if DATA["df"] is None:
        return [], [], [], [], [], [], []

    df0 = DATA["df"]

    def get_sorted(col):
        if col not in df0.columns:
            return []
        vals = df0[col].dropna().astype(str).unique().tolist()
        vals = [v for v in vals if v.strip().upper() not in ("", "NAN", "NONE")]
        return sorted(vals)

    munis = get_sorted("cole_mcpio_ubicacion")
    zonas = get_sorted("cole_area_ubicacion")
    nats  = get_sorted("cole_naturaleza")
    estr  = get_sorted("fami_estratovivienda")
    em    = get_sorted("fami_educacionmadre")
    ep    = get_sorted("fami_educacionpadre")

    zona_opts = [{"label": z.title(), "value": z} for z in zonas]

    return (
        [{"label": m.title(), "value": m} for m in munis],
        zona_opts,
        zonas,
        [{"label": n.title(), "value": n} for n in nats],
        [{"label": e.title(), "value": e} for e in estr],
        [{"label": e.title(), "value": e} for e in em],
        [{"label": e.title(), "value": e} for e in ep],
    )

# ==============================
# CALLBACK PRINCIPAL
# ==============================
@app.callback(
    Output("contenido", "children"),
    Output("kpi-n", "children"),
    Output("kpi-mean", "children"),
    Output("kpi-std", "children"),
    Output("kpi-gap", "children"),
    Output("insight", "children"),
    Input("btn", "n_clicks"),
    Input("tabs", "active_tab"),
    Input("f-area", "value"),
    State("f-muni", "value"),
    State("f-zona", "value"),
    State("f-nat", "value"),
    State("f-estrato", "value"),
    State("f-edu-m", "value"),
    State("f-edu-p", "value"),
    prevent_initial_call=False
)
def actualizar(_nclicks, active_tab, area, munis, zonas, nats, estratos, edu_m, edu_p):

    if DATA["df"] is None:
        empty = kpi_card("—", "—")
        return (
            dbc.Alert("Primero sube un archivo (CSV o Excel) desde el panel izquierdo.", color="primary"),
            empty, empty, empty, empty,
            "Sube el archivo para habilitar el análisis."
        )

    munis_t = _as_tuple(munis)
    zonas_t = _as_tuple(zonas)
    nats_t  = _as_tuple(nats)
    estr_t  = _as_tuple(estratos)
    em_t    = _as_tuple(edu_m)
    ep_t    = _as_tuple(edu_p)

    d = filter_df(DATA["df"], list(munis_t), list(zonas_t), list(nats_t), list(estr_t), list(em_t), list(ep_t), area)

    if d.empty:
        empty = kpi_card("—", "—")
        return dbc.Alert("No hay datos con esos filtros.", color="warning"), empty, empty, empty, empty, "No hay datos para interpretar."

    # KPIs
    n = len(d)
    mean = float(d[area].mean())
    std = float(d[area].std())

    gap = "—"
    if "cole_naturaleza" in d.columns:
        nat_means = d.groupby("cole_naturaleza", observed=True)[area].mean().dropna()
        if len(nat_means) >= 2:
            gap = f"{(nat_means.max() - nat_means.min()):.1f} pts"

    kpi_n = kpi_card("Estudiantes (N)", f"{n:,}".replace(",", "."))
    kpi_mean = kpi_card("Promedio", f"{mean:.1f}", AREAS.get(area, area))
    kpi_std = kpi_card("Dispersión (Std)", f"{std:.1f}")
    kpi_gap = kpi_card("Brecha por naturaleza", gap, "máx − mín")

    # ======================
    # Q1: Territorial
    # ======================
    if active_tab == "tab-q1":
        v = DATA["version"]
        muni_stats, muni_desc = cached_q1(v, area, munis_t, zonas_t, nats_t, estr_t, em_t, ep_t)

        if muni_stats is None or muni_stats.empty:
            return dbc.Alert("No hay columna de municipio (cole_mcpio_ubicacion) en este archivo.", color="warning"), kpi_n, kpi_mean, kpi_std, kpi_gap, "Falta municipio."

        fig_rank_top = px.bar(
            muni_stats.head(12),
            x="media", y="cole_mcpio_ubicacion", orientation="h",
            hover_data=["n"],
            title=f"Top 12 municipios por {AREAS.get(area, area)}",
            template=PLOT_TEMPLATE
        ).update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))

        fig_rank_bottom = px.bar(
            muni_stats.tail(12).sort_values("media"),
            x="media", y="cole_mcpio_ubicacion", orientation="h",
            hover_data=["n"],
            title=f"Bottom 12 municipios por {AREAS.get(area, area)}",
            template=PLOT_TEMPLATE
        ).update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))

        d_plot = sample_for_plot(d, seed=1)

        fig_box_zona = px.box(
            d_plot, x="cole_area_ubicacion", y=area, points=False,
            color="cole_area_ubicacion",
            color_discrete_map=COLOR_ZONA,
            title=f"Distribución por zona (muestra ≤ {MAX_SAMPLE})",
            template=PLOT_TEMPLATE
        ).update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))

        fig_box_nat = px.box(
            d_plot, x="cole_naturaleza", y=area, points=False,
            title=f"Distribución por naturaleza (muestra ≤ {MAX_SAMPLE})",
            template=PLOT_TEMPLATE
        ).update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))

        # --- Violin elegante ---
        fig_violin = px.violin(
            d_plot,
            x="cole_area_ubicacion",
            y=area,
            color="cole_area_ubicacion",
            color_discrete_map=COLOR_ZONA,
            box=True,
            points=False,
            template=PLOT_TEMPLATE
        )
        fig_violin.update_layout(
            title="Distribución detallada por zona",
            height=420,
            margin=dict(l=10, r=10, t=60, b=10),
            violinmode="group"
        )

        contenido = dbc.Container([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_rank_top, config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(figure=fig_rank_bottom, config={"displayModeBar": False}), md=6),
            ], className="g-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_box_zona, config={"displayModeBar": False}), md=4),
                dbc.Col(dcc.Graph(figure=fig_box_nat, config={"displayModeBar": False}), md=4),
                dbc.Col(dcc.Graph(figure=fig_violin, config={"displayModeBar": False}), md=4),
            ], className="g-3"),
            html.Hr(),
            html.H5("Resumen por municipio", style={"fontWeight": "900"}),
            make_table(muni_desc, max_rows=15)
        ], fluid=True)

        insight = "Q1 ranking municipal"
        return contenido, kpi_n, kpi_mean, kpi_std, kpi_gap, insight

    # ======================
    # Q2: Socioeconómico + riesgo
    # ======================

    if active_tab == "tab-q2":
        v = DATA["version"]
        out = cached_q2(v, area, munis_t, zonas_t, nats_t, estr_t, em_t, ep_t)

        if not out.get("ok", False):
            return dbc.Container([
                dbc.Alert(f"Q2 no puede correr: {out.get('error','Error desconocido')}", color="warning"),
                dbc.Badge("Tip: revisa que existan fami_estratovivienda y fami_educacionmadre.", color="secondary")
            ], fluid=True), kpi_n, kpi_mean, kpi_std, kpi_gap, "Q2 con columnas faltantes."

        riesgo_estrato = out["riesgo"]
        mean_estrato = out["mean"]
        pivot = out["pivot"]
        d_se = out["d_se"]

        if d_se is None or riesgo_estrato.empty or pivot is None or pivot.empty:
            return dbc.Alert("Q2: no hay datos válidos (tras filtrar SIN ESTRATO y vacíos).", color="warning"), \
                kpi_n, kpi_mean, kpi_std, kpi_gap, "Q2 sin datos."

        # muestra para plots rápidos
        d_plot = sample_for_plot(d_se, seed=2)

        fig_riesgo = px.bar(
            riesgo_estrato,
            x="fami_estratovivienda",
            y="riesgo_pct",
            title="Riesgo de bajo desempeño por estrato (25% más bajo)",
            template=PLOT_TEMPLATE
        ).update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10))

        fig_box = px.box(
            d_plot,
            x="fami_estratovivienda",
            y=area,
            points=False,
            title=f"Puntaje por estrato (muestra ≤ {MAX_SAMPLE})",
            template=PLOT_TEMPLATE
        ).update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))

        # colores pastel seguros (lista explícita → no depende de nombres)
        pastel_scale = [
            "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
            "#4292c6", "#2171b5"
        ]

        # Heatmap estable: px.imshow con valores
        fig_heat = px.imshow(
            pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            color_continuous_scale=pastel_scale,
            text_auto=".1f",
            aspect="auto",
            template=PLOT_TEMPLATE
        )
        fig_heat.update_layout(
            title="Promedio: estrato × educación madre",
            height=520,
            margin=dict(l=10, r=10, t=60, b=10),
            xaxis_title="Educación madre",
            yaxis_title="Estrato"
        )

        contenido = dbc.Container([
            html.H5("Q2: Brechas socioeconómicas (estrato/educación) + riesgo", style={"fontWeight": "900"}),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_riesgo, config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(figure=fig_box, config={"displayModeBar": False}), md=6),
            ], className="g-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_heat, config={"displayModeBar": False}), md=12),
            ], className="g-3"),

            html.Hr(),
            html.H5("Promedios por estrato", style={"fontWeight": "900"}),
            make_table(mean_estrato.rename(columns={"fami_estratovivienda": "Estrato"}), max_rows=20)
        ], fluid=True)

        insight = "Q2 diferencias socioeconómicas"
        return contenido, kpi_n, kpi_mean, kpi_std, kpi_gap, insight

    # ======================
    # Q3: Edad + riesgo
    # ======================
    if active_tab == "tab-age":
        v = DATA["version"]
        out = cached_q3(v, area, munis_t, zonas_t, nats_t, estr_t, em_t, ep_t)
        if out is None:
            return dbc.Container([
                dbc.Alert(
                    "Q3 no puede calcular edad. Necesitas 'periodo' (o año) y 'estu_fechanacimiento' (o similar).",
                    color="warning"
                ),
                html.Div("Columnas detectadas (muestra):", className="fw-semibold"),
                dbc.Badge(", ".join(list(d.columns)[:60]) + (" ..." if len(d.columns) > 60 else ""), color="secondary")
            ], fluid=True), kpi_n, kpi_mean, kpi_std, kpi_gap, "No hay edad."

        riesgo_age, mean_age, corr_age = out
        if riesgo_age.empty:
            return dbc.Alert("Q3: no hay edad válida con esos filtros.", color="warning"), kpi_n, kpi_mean, kpi_std, kpi_gap, "Q3 sin datos."

        d_age = d.dropna(subset=["edad", area]).copy()
        d_plot = sample_for_plot(d_age, seed=3)

        fig_hist = px.histogram(
            d_plot, x="edad", nbins=25,
            title=f"Distribución de edad (muestra ≤ {MAX_SAMPLE})",
            template=PLOT_TEMPLATE
        ).update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10))

        fig_scatter = px.scatter(
            d_plot, x="edad", y=area,
            title=f"Edad vs {AREAS.get(area, area)} (muestra ≤ {MAX_SAMPLE})",
            template=PLOT_TEMPLATE,
            opacity=0.55,
            render_mode="webgl"
        ).update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))

        fig_box = px.box(
            d_plot, x="edad_grupo", y=area, points=False,
            title=f"Puntaje por grupo de edad (muestra ≤ {MAX_SAMPLE})",
            template=PLOT_TEMPLATE
        ).update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))

        fig_riesgo = px.bar(
            riesgo_age, x="edad_grupo", y="riesgo_pct",
            title="Riesgo de bajo desempeño por grupo de edad (25% más bajo)",
            template=PLOT_TEMPLATE
        ).update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10))

        texto = f"Correlación simple (edad vs puntaje) ≈ {corr_age:+.2f} (no implica causalidad)."

        contenido = dbc.Container([
            html.H5("Q3: Edad + riesgo", style={"fontWeight": "900"}),
            dbc.Alert(texto, color="info"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_hist, config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(figure=fig_riesgo, config={"displayModeBar": False}), md=6),
            ], className="g-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_scatter, config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(figure=fig_box, config={"displayModeBar": False}), md=6),
            ], className="g-3"),
            html.Hr(),
            html.H5("Promedios por grupo de edad", style={"fontWeight": "900"}),
            make_table(mean_age.rename(columns={"edad_grupo": "Grupo edad"}).sort_values("Promedio"), max_rows=20)
        ], fluid=True)

        insight = "Q3 relación edad-riesgo"
        return contenido, kpi_n, kpi_mean, kpi_std, kpi_gap, insight

    return dbc.Alert("Selecciona una pestaña.", color="secondary"), kpi_n, kpi_mean, kpi_std, kpi_gap, "—"

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, port=8050)