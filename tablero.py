import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import warnings

from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")

# ==============================
# CONFIG VISUAL BONITO
# ==============================
PLOT_TEMPLATE = "plotly_white"

# Colores consistentes (no “arcoíris”)
# (Plotly asigna, pero aquí lo forzamos por categorías conocidas)
COLOR_ZONA = {"URBANO": "#2E86AB", "RURAL": "#F18F01"}

AREAS = {
    "punt_global": "Puntaje Global",
    "punt_matematicas": "Matemáticas",
    "punt_lectura_critica": "Lectura Crítica",
    "punt_sociales_ciudadanas": "Sociales",
    "punt_c_naturales": "C. Naturales",
    "punt_ingles": "Inglés"
}

# ==============================
# CARGA / LIMPIEZA
# ==============================
df = pd.read_csv("Resultados_ICFES_Cordoba_clean.csv")

def norm_txt(x):
    if pd.isna(x):
        return x
    return str(x).strip().upper()

# Normaliza textos clave (evita "Urbano" vs "URBANO")
for col in [
    "cole_mcpio_ubicacion", "cole_area_ubicacion", "cole_naturaleza",
    "fami_estratovivienda", "fami_educacionmadre", "fami_educacionpadre"
]:
    if col in df.columns:
        df[col] = df[col].map(norm_txt)

# Asegurar numéricos de puntajes
for c in AREAS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df[df["punt_global"].notna() & (df["punt_global"] > 0)].copy()

# Listas para filtros
MUNICIPIOS = sorted(df["cole_mcpio_ubicacion"].dropna().unique())
ZONAS = sorted(df["cole_area_ubicacion"].dropna().unique())
NATURALEZAS = sorted(df["cole_naturaleza"].dropna().unique())
ESTRATOS = sorted(df["fami_estratovivienda"].dropna().unique())
EDU_MADRE = sorted(df["fami_educacionmadre"].dropna().unique())
EDU_PADRE = sorted(df["fami_educacionpadre"].dropna().unique())

# ==============================
# HELPERS
# ==============================
def kpi_card(title, value, subtitle=""):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-muted", style={"fontSize": "0.9rem"}),
            html.Div(value, style={"fontSize": "1.7rem", "fontWeight": "800"}),
            html.Div(subtitle, className="text-muted", style={"fontSize": "0.85rem"})
        ]),
        className="shadow-sm",
        style={"borderRadius": "16px"}
    )

def make_table(df_small, max_rows=12):
    show = df_small.head(max_rows).copy()
    return dbc.Table.from_dataframe(show, striped=True, bordered=False, hover=True, size="sm")

def safe_title(s):
    return (s or "").title()

def filtrar(munis, zonas, nats, estratos, edu_m, edu_p, area):
    d = df.copy()
    if munis:
        d = d[d["cole_mcpio_ubicacion"].isin(munis)]
    if zonas:
        d = d[d["cole_area_ubicacion"].isin(zonas)]
    if nats:
        d = d[d["cole_naturaleza"].isin(nats)]
    if estratos:
        d = d[d["fami_estratovivienda"].isin(estratos)]
    if edu_m:
        d = d[d["fami_educacionmadre"].isin(edu_m)]
    if edu_p:
        d = d[d["fami_educacionpadre"].isin(edu_p)]

    d = d[d[area].notna()]
    return d

def descriptivas(d, area):
    # Global
    out = {}
    out["n"] = len(d)
    out["mean"] = float(d[area].mean())
    out["std"] = float(d[area].std())
    out["p25"] = float(d[area].quantile(0.25))
    out["p50"] = float(d[area].quantile(0.50))
    out["p75"] = float(d[area].quantile(0.75))
    return out

def anova_1way(d, area, factor):
    # ANOVA 1 vía (rápida y útil)
    dd = d[[area, factor]].dropna()
    if dd[factor].nunique() < 2 or len(dd) < 30:
        return None, None

    model = ols(f"{area} ~ C({factor})", data=dd).fit()
    aov = sm.stats.anova_lm(model, typ=2)

    # eta^2 (tamaño de efecto simple)
    ss_factor = aov.loc[f"C({factor})", "sum_sq"]
    ss_total = aov["sum_sq"].sum()
    eta2 = float(ss_factor / ss_total) if ss_total > 0 else np.nan

    return aov.reset_index(), eta2

def tukey_posthoc(d, area, factor):
    dd = d[[area, factor]].dropna()
    if dd[factor].nunique() < 3 or len(dd) < 60:
        return None
    try:
        res = pairwise_tukeyhsd(endog=dd[area], groups=dd[factor], alpha=0.05)
        tbl = pd.DataFrame(data=res.summary().data[1:], columns=res.summary().data[0])
        return tbl
    except Exception:
        return None

def insight_text(d, area):
    # Textos “qué se ve” sin ponernos intensos
    # Brecha Urbano-Rural
    ur = None
    if "URBANO" in d["cole_area_ubicacion"].unique() and "RURAL" in d["cole_area_ubicacion"].unique():
        mu_u = d.loc[d["cole_area_ubicacion"] == "URBANO", area].mean()
        mu_r = d.loc[d["cole_area_ubicacion"] == "RURAL", area].mean()
        ur = mu_u - mu_r

    # Naturaleza (mayor vs menor promedio)
    nat_rank = (
        d.groupby("cole_naturaleza")[area].mean()
        .sort_values(ascending=False)
        .dropna()
    )
    top_nat = nat_rank.index[0] if len(nat_rank) else None
    bot_nat = nat_rank.index[-1] if len(nat_rank) else None
    gap_nat = (nat_rank.iloc[0] - nat_rank.iloc[-1]) if len(nat_rank) >= 2 else None

    lines = []
    if ur is not None:
        lines.append(f"En promedio, **URBANO vs RURAL** difiere en **{ur:+.1f} puntos** (positivo = urbano mayor).")
    if top_nat and bot_nat and gap_nat is not None:
        lines.append(f"Por **naturaleza**, el promedio más alto es **{safe_title(top_nat)}** y el más bajo **{safe_title(bot_nat)}**; la diferencia es **{gap_nat:.1f} puntos**.")
    if not lines:
        lines.append("Con los filtros actuales no se ve una brecha clara (o falta alguna categoría). Prueba ampliar zona o naturaleza.")
    return " ".join(lines)

# ==============================
# APP
# ==============================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "ICFES Córdoba — Tablero"

sidebar = dbc.Card(
    dbc.CardBody([
        html.H4("ICFES Córdoba", style={"fontWeight": "900"}),
        html.Div("Brechas territoriales y socioeconómicas (Saber 11).", className="text-muted"),
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
        dcc.Dropdown(
            id="f-muni",
            options=[{"label": m.title(), "value": m} for m in MUNICIPIOS],
            multi=True,
            placeholder="Todos"
        ),
        html.Br(),

        html.Div("Zona (urbano/rural)", className="fw-semibold"),
        dcc.Checklist(
            id="f-zona",
            options=[{"label": z.title(), "value": z} for z in ZONAS],
            value=ZONAS,
            inputStyle={"marginRight": "8px", "marginLeft": "4px"}
        ),
        html.Br(),

        html.Div("Naturaleza del colegio", className="fw-semibold"),
        dcc.Dropdown(
            id="f-nat",
            options=[{"label": n.title(), "value": n} for n in NATURALEZAS],
            multi=True,
            placeholder="Todas"
        ),
        html.Hr(),

        html.Div("Socioeconómico (opcional)", className="fw-semibold"),
        html.Div("Estrato", className="text-muted", style={"fontSize": "0.9rem"}),
        dcc.Dropdown(
            id="f-estrato",
            options=[{"label": e.title(), "value": e} for e in ESTRATOS],
            multi=True,
            placeholder="Todos"
        ),
        html.Br(),

        html.Div("Educación madre", className="text-muted", style={"fontSize": "0.9rem"}),
        dcc.Dropdown(
            id="f-edu-m",
            options=[{"label": e.title(), "value": e} for e in EDU_MADRE],
            multi=True,
            placeholder="Todas"
        ),
        html.Br(),

        html.Div("Educación padre", className="text-muted", style={"fontSize": "0.9rem"}),
        dcc.Dropdown(
            id="f-edu-p",
            options=[{"label": e.title(), "value": e} for e in EDU_PADRE],
            multi=True,
            placeholder="Todas"
        ),
        html.Br(),

        dbc.Button("Aplicar filtros", id="btn", color="primary", className="w-100")
    ]),
    className="shadow-sm",
    style={"borderRadius": "18px"}
)

tabs = dbc.Tabs(
    [
        dbc.Tab(label="Brechas territoriales (Q1)", tab_id="tab-q1"),
        dbc.Tab(label="Socioeconómico + riesgo (Q2)", tab_id="tab-q2"),
        dbc.Tab(label="Estadística (ANOVA + Tukey)", tab_id="tab-stats"),
    ],
    id="tabs",
    active_tab="tab-q1"
)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(sidebar, width=3),
        dbc.Col([
            html.H3("Brechas Educativas — Córdoba", style={"fontWeight": "900"}),
            html.Div("Gráficas claras + textos interpretativos + estadística básica.", className="text-muted"),
            html.Br(),

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
# CALLBACK
# ==============================
@app.callback(
    Output("contenido", "children"),
    Output("kpi-n", "children"),
    Output("kpi-mean", "children"),
    Output("kpi-std", "children"),
    Output("kpi-gap", "children"),
    Output("insight", "children"),
    Input("btn", "n_clicks"),
    State("tabs", "active_tab"),
    State("f-muni", "value"),
    State("f-zona", "value"),
    State("f-nat", "value"),
    State("f-estrato", "value"),
    State("f-edu-m", "value"),
    State("f-edu-p", "value"),
    State("f-area", "value"),
    prevent_initial_call=False
)
def actualizar(_, active_tab, munis, zonas, nats, estratos, edu_m, edu_p, area):

    d = filtrar(munis, zonas, nats, estratos, edu_m, edu_p, area)
    if d.empty:
        empty = kpi_card("—", "—")
        return dbc.Alert("No hay datos con esos filtros.", color="warning"), empty, empty, empty, empty, "No hay datos para interpretar."

    desc = descriptivas(d, area)

    # KPI "gap" (máx - mín por naturaleza, si aplica)
    nat_means = d.groupby("cole_naturaleza")[area].mean().dropna()
    gap = "—"
    if nat_means.size >= 2:
        gap = f"{(nat_means.max() - nat_means.min()):.1f} pts"

    kpi_n = kpi_card("Estudiantes (N)", f"{desc['n']:,}".replace(",", "."))
    kpi_mean = kpi_card("Promedio", f"{desc['mean']:.1f}", AREAS.get(area, area))
    kpi_std = kpi_card("Dispersión (Std)", f"{desc['std']:.1f}", "más alto = más variación")
    kpi_gap = kpi_card("Brecha por naturaleza", gap, "máx − mín (promedios)")

    insight = insight_text(d, area)

    # =========================
    # TAB Q1: TERRITORIAL
    # =========================
    if active_tab == "tab-q1":
        # 1) Ranking municipios
        muni_stats = (
            d.groupby("cole_mcpio_ubicacion")[area]
             .agg(media="mean", n="size")
             .reset_index()
             .sort_values("media", ascending=False)
        )

        fig_rank_top = px.bar(
            muni_stats.head(12),
            x="media", y="cole_mcpio_ubicacion", orientation="h",
            hover_data=["n"],
            title=f"Top 12 municipios por {AREAS[area]}",
            template=PLOT_TEMPLATE
        )
        fig_rank_top.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        fig_rank_top.update_xaxes(title="Promedio (pts)")
        fig_rank_top.update_yaxes(title="Municipio")

        fig_rank_bottom = px.bar(
            muni_stats.tail(12).sort_values("media"),
            x="media", y="cole_mcpio_ubicacion", orientation="h",
            hover_data=["n"],
            title=f"Bottom 12 municipios por {AREAS[area]} (prioridad potencial)",
            template=PLOT_TEMPLATE
        )
        fig_rank_bottom.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        fig_rank_bottom.update_xaxes(title="Promedio (pts)")
        fig_rank_bottom.update_yaxes(title="Municipio")

        # 2) Violín legible: Zona
        fig_violin_zona = px.violin(
            d, x="cole_area_ubicacion", y=area,
            box=True, points=False,
            color="cole_area_ubicacion",
            color_discrete_map=COLOR_ZONA,
            title="Distribución (violín) por zona",
            template=PLOT_TEMPLATE
        )
        fig_violin_zona.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        fig_violin_zona.update_xaxes(title="Zona")
        fig_violin_zona.update_yaxes(title="Puntaje")

        # 3) Violín por naturaleza (clave)
        fig_violin_nat = px.violin(
            d, x="cole_naturaleza", y=area,
            box=True, points=False,
            title="Distribución (violín) por naturaleza del colegio",
            template=PLOT_TEMPLATE
        )
        fig_violin_nat.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        fig_violin_nat.update_xaxes(title="Naturaleza")
        fig_violin_nat.update_yaxes(title="Puntaje")

        # 4) Interacción simple: promedios zona × naturaleza
        inter = (
            d.groupby(["cole_area_ubicacion", "cole_naturaleza"])[area]
             .mean().reset_index()
        )
        fig_inter = px.bar(
            inter, x="cole_naturaleza", y=area,
            color="cole_area_ubicacion",
            barmode="group",
            color_discrete_map=COLOR_ZONA,
            title="Promedios: naturaleza × zona (brechas visibles)",
            template=PLOT_TEMPLATE
        )
        fig_inter.update_layout(height=460, margin=dict(l=10, r=10, t=60, b=10))
        fig_inter.update_xaxes(title="Naturaleza")
        fig_inter.update_yaxes(title="Promedio (pts)")

        # Descriptivas por municipio (tabla compacta)
        muni_desc = (
            d.groupby("cole_mcpio_ubicacion")[area]
             .agg(N="size", Promedio="mean", Std="std")
             .reset_index()
             .sort_values("Promedio")
        )
        muni_desc["Promedio"] = muni_desc["Promedio"].round(1)
        muni_desc["Std"] = muni_desc["Std"].round(1)

        texto = html.Div([
            html.H5("Cómo leer esta pestaña", style={"fontWeight": "800"}),
            html.Ul([
                html.Li("Los rankings muestran territorios con desempeño alto/bajo; los de abajo son candidatos naturales a intervención."),
                html.Li("Los violines muestran la distribución: si un grupo tiene violín más “bajo” y “ancho”, tiende a concentrarse en puntajes menores y con más variación."),
                html.Li("La barra naturaleza×zona ayuda a ver si la brecha cambia según el contexto rural/urbano.")
            ], className="text-muted")
        ])

        return dbc.Container([
            texto,
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_rank_top), md=6),
                dbc.Col(dcc.Graph(figure=fig_rank_bottom), md=6),
            ], className="g-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_violin_zona), md=6),
                dbc.Col(dcc.Graph(figure=fig_violin_nat), md=6),
            ], className="g-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_inter), md=12),
            ], className="g-3"),

            html.Hr(),
            html.H5("Resumen descriptivo por municipio (compacto)", style={"fontWeight": "800"}),
            make_table(muni_desc, max_rows=15)
        ], fluid=True), kpi_n, kpi_mean, kpi_std, kpi_gap, insight

    # =========================
    # TAB Q2: SOCIOECONÓMICO
    # =========================
    if active_tab == "tab-q2":
        # 1) Violín por estrato (siempre sirve)
        fig_violin_estrato = px.violin(
            d, x="fami_estratovivienda", y=area,
            box=True, points=False,
            title="Distribución (violín) por estrato",
            template=PLOT_TEMPLATE
        )
        fig_violin_estrato.update_layout(height=440, margin=dict(l=10, r=10, t=60, b=10))
        fig_violin_estrato.update_xaxes(title="Estrato")
        fig_violin_estrato.update_yaxes(title="Puntaje")

        # 2) Heatmap: estrato × educación madre (promedio)
        heat = (
            d.groupby(["fami_estratovivienda", "fami_educacionmadre"])[area]
             .mean().reset_index()
        )
        fig_heat = px.density_heatmap(
            heat,
            x="fami_educacionmadre",
            y="fami_estratovivienda",
            z=area,
            histfunc="avg",
            title="Mapa de promedios: estrato × educación de la madre",
            template=PLOT_TEMPLATE
        )
        fig_heat.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
        fig_heat.update_xaxes(title="Educación madre")
        fig_heat.update_yaxes(title="Estrato")

        # 3) Promedios por educación padres (barras)
        edu_stats = (
            d.groupby("fami_educacionmadre")[area]
             .agg(Promedio="mean", N="size")
             .reset_index()
             .sort_values("Promedio")
        )
        edu_stats["Promedio"] = edu_stats["Promedio"].round(1)

        fig_edu = px.bar(
            edu_stats,
            x="Promedio",
            y="fami_educacionmadre",
            orientation="h",
            hover_data=["N"],
            title="Promedio por educación de la madre (con N)",
            template=PLOT_TEMPLATE
        )
        fig_edu.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
        fig_edu.update_xaxes(title="Promedio (pts)")
        fig_edu.update_yaxes(title="Educación madre")

        texto = html.Div([
            html.H5("Qué buscar aquí", style={"fontWeight": "800"}),
            html.Ul([
                html.Li("Si el violín por estrato se desplaza hacia abajo en estratos bajos, hay un gradiente socioeconómico fuerte."),
                html.Li("El heatmap deja ver combinaciones críticas (estrato bajo + baja educación parental)."),
                html.Li("La barra por educación parental te ayuda a describir la brecha de forma simple y contundente.")
            ], className="text-muted")
        ])

        return dbc.Container([
            texto,
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_violin_estrato), md=6),
                dbc.Col(dcc.Graph(figure=fig_heat), md=6),
            ], className="g-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_edu), md=12),
            ], className="g-3"),
        ], fluid=True), kpi_n, kpi_mean, kpi_std, kpi_gap, insight

    # =========================
    # TAB STATS: ANOVA + TUKEY + DESCRIPTIVAS
    # =========================
    # Descriptivas por grupo (naturaleza y zona)
    desc_nat = (
        d.groupby("cole_naturaleza")[area]
         .agg(N="size", Promedio="mean", Std="std")
         .reset_index()
         .sort_values("Promedio", ascending=False)
    )
    desc_nat["Promedio"] = desc_nat["Promedio"].round(1)
    desc_nat["Std"] = desc_nat["Std"].round(1)

    desc_zona = (
        d.groupby("cole_area_ubicacion")[area]
         .agg(N="size", Promedio="mean", Std="std")
         .reset_index()
         .sort_values("Promedio", ascending=False)
    )
    desc_zona["Promedio"] = desc_zona["Promedio"].round(1)
    desc_zona["Std"] = desc_zona["Std"].round(1)

    # ANOVA 1 vía: naturaleza
    aov_nat, eta_nat = anova_1way(d, area, "cole_naturaleza")
    aov_zona, eta_zona = anova_1way(d, area, "cole_area_ubicacion")

    tukey_nat = tukey_posthoc(d, area, "cole_naturaleza")

    blocks = []

    blocks.append(html.Div([
        html.H5("Descriptivas", style={"fontWeight": "800"}),
        html.P("Estas tablas resumen tamaño de muestra, promedio y dispersión por grupo.", className="text-muted"),
        html.H6("Por naturaleza", style={"fontWeight": "700"}),
        make_table(desc_nat, max_rows=20),
        html.Br(),
        html.H6("Por zona", style={"fontWeight": "700"}),
        make_table(desc_zona, max_rows=10),
    ]))

    blocks.append(html.Hr())

    # ANOVA
    aov_texts = []
    if aov_nat is not None:
        p = float(aov_nat.loc[aov_nat["index"] == "C(cole_naturaleza)", "PR(>F)"].values[0])
        aov_texts.append(f"ANOVA (naturaleza): p-value = {p:.4g} | tamaño de efecto (eta²) ≈ {eta_nat:.3f}")
    else:
        aov_texts.append("ANOVA (naturaleza): no se pudo estimar (pocos grupos o pocos datos).")

    if aov_zona is not None:
        p = float(aov_zona.loc[aov_zona["index"] == "C(cole_area_ubicacion)", "PR(>F)"].values[0])
        aov_texts.append(f"ANOVA (zona): p-value = {p:.4g} | tamaño de efecto (eta²) ≈ {eta_zona:.3f}")
    else:
        aov_texts.append("ANOVA (zona): no se pudo estimar (pocos grupos o pocos datos).")

    blocks.append(html.Div([
        html.H5("ANOVA (¿hay diferencias significativas?)", style={"fontWeight": "800"}),
        dbc.Alert(" | ".join(aov_texts), color="info")
    ]))

    # Tukey (posthoc)
    blocks.append(html.Div([
        html.H5("Post-hoc Tukey (¿entre qué grupos está la diferencia?)", style={"fontWeight": "800"}),
        html.P("Solo se calcula si hay suficientes categorías y datos.", className="text-muted"),
        make_table(tukey_nat, max_rows=15) if tukey_nat is not None else dbc.Alert(
            "Tukey no disponible con los filtros actuales (pocos grupos o poco N).",
            color="secondary"
        )
    ]))

    return dbc.Container(blocks, fluid=True), kpi_n, kpi_mean, kpi_std, kpi_gap, insight


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=8050)