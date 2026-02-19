import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ==============================
# CONFIG VISUAL (bonito y legible)
# ==============================
PLOT_TEMPLATE = "plotly_white"

COLOR_ZONA = {
    "URBANO": "#2E86AB",
    "RURAL": "#F18F01"
}

# Para "oficial vs no oficial" (cole_caracter suele tener textos como OFICIAL / NO OFICIAL)
COLOR_CARACTER = {
    "OFICIAL": "#3D9970",
    "NO OFICIAL": "#B10DC9"
}

AREAS = {
    "punt_global": "Puntaje Global",
    "punt_matematicas": "Matemáticas",
    "punt_lectura_critica": "Lectura Crítica",
    "punt_sociales_ciudadanas": "Sociales",
    "punt_c_naturales": "C. Naturales",
    "punt_ingles": "Inglés"
}

# ==============================
# CARGA DE DATOS
# ==============================
df = pd.read_csv("Resultados_ICFES_Cordoba_clean.csv")

# Asegurar numéricos
for c in AREAS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df[df["punt_global"].notna() & (df["punt_global"] > 0)].copy()

# Limpieza básica (estandarizar textos clave)
def norm_txt(s):
    if pd.isna(s): 
        return s
    return str(s).strip().upper()

for col in ["cole_area_ubicacion", "cole_caracter", "cole_mcpio_ubicacion", "cole_naturaleza",
            "fami_estratovivienda", "fami_educacionmadre", "fami_educacionpadre"]:
    if col in df.columns:
        df[col] = df[col].map(norm_txt)

# Valores para filtros
MUNICIPIOS = sorted(df["cole_mcpio_ubicacion"].dropna().unique())
ZONAS = sorted(df["cole_area_ubicacion"].dropna().unique())
CARACTERES = sorted(df["cole_caracter"].dropna().unique())  # OFICIAL / NO OFICIAL (ideal)
ESTRATOS = sorted(df["fami_estratovivienda"].dropna().unique())
EDU_MADRE = sorted(df["fami_educacionmadre"].dropna().unique())
EDU_PADRE = sorted(df["fami_educacionpadre"].dropna().unique())

# ==============================
# APP
# ==============================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],  # Flatly es limpia
    suppress_callback_exceptions=True
)
app.title = "ICFES Córdoba — Brechas Educativas"

# ==============================
# COMPONENTES UI
# ==============================
def kpi_card(title, value, subtitle=None):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-muted", style={"fontSize": "0.9rem"}),
            html.Div(value, style={"fontSize": "1.7rem", "fontWeight": "700"}),
            html.Div(subtitle or "", className="text-muted", style={"fontSize": "0.85rem"})
        ]),
        className="shadow-sm",
        style={"borderRadius": "14px"}
    )

sidebar = dbc.Card(
    dbc.CardBody([
        html.H4("ICFES Córdoba", style={"fontWeight": "800"}),
        html.Div("Explora brechas territoriales y socioeconómicas en Saber 11.", className="text-muted"),
        html.Hr(),

        html.Div("Área / puntaje", className="fw-semibold"),
        dcc.Dropdown(
            id="f-area",
            options=[{"label": v, "value": k} for k, v in AREAS.items()],
            value="punt_global",
            clearable=False
        ),
        html.Br(),

        html.Div("Municipio", className="fw-semibold"),
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

        html.Div("Carácter (oficial/no oficial)", className="fw-semibold"),
        dcc.Checklist(
            id="f-caracter",
            options=[{"label": c.title(), "value": c} for c in CARACTERES],
            value=CARACTERES,
            inputStyle={"marginRight": "8px", "marginLeft": "4px"}
        ),
        html.Hr(),

        html.Div("Socioeconómico", className="fw-semibold"),
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
            placeholder="Todos"
        ),
        html.Br(),

        html.Div("Umbral 'bajo desempeño' (puntaje)", className="fw-semibold"),
        dcc.Slider(
            id="f-umbral",
            min=150, max=400, step=10, value=250,
            marks={150:"150", 200:"200", 250:"250", 300:"300", 350:"350", 400:"400"}
        ),
        html.Br(),

        dbc.Button("Aplicar filtros", id="btn", color="primary", className="w-100")
    ]),
    className="shadow-sm",
    style={"borderRadius": "16px"}
)

tabs = dbc.Tabs(
    [
        dbc.Tab(label="Q1 — Brechas territoriales", tab_id="tab-q1"),
        dbc.Tab(label="Q2 — Socioeconómico y riesgo", tab_id="tab-q2"),
    ],
    id="tabs",
    active_tab="tab-q1"
)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(sidebar, width=3),
        dbc.Col([
            html.H3("Brechas Educativas — Córdoba", style={"fontWeight": "800"}),
            html.Div("Responde directamente a: brechas por municipio/zona/carácter y efecto socioeconómico.", className="text-muted"),
            html.Br(),

            dbc.Row([
                dbc.Col(kpi_card("Estudiantes (N)", "—"), md=3, id="kpi-n"),
                dbc.Col(kpi_card("Promedio", "—"), md=3, id="kpi-mean"),
                dbc.Col(kpi_card("Brecha Urbano–Rural", "—"), md=3, id="kpi-ur"),
                dbc.Col(kpi_card("Brecha Oficial–No oficial", "—"), md=3, id="kpi-on"),
            ], className="g-3"),

            html.Br(),
            tabs,
            html.Br(),

            html.Div(id="contenido")
        ], width=9),
    ], className="g-3")
], fluid=True)

# ==============================
# FILTRAR
# ==============================
def filtrar(munis, zonas, caracteres, estratos, edu_m, umbral, area):
    d = df.copy()

    if munis:
        d = d[d["cole_mcpio_ubicacion"].isin(munis)]

    if zonas:
        d = d[d["cole_area_ubicacion"].isin(zonas)]

    if caracteres:
        d = d[d["cole_caracter"].isin(caracteres)]

    if estratos:
        d = d[d["fami_estratovivienda"].isin(estratos)]

    if edu_m:
        d = d[d["fami_educacionmadre"].isin(edu_m)]

    d = d[d[area].notna()]
    d["bajo"] = (d[area] < umbral).astype(int)
    return d

# ==============================
# CALLBACK PRINCIPAL
# ==============================
@app.callback(
    Output("contenido", "children"),
    Output("kpi-n", "children"),
    Output("kpi-mean", "children"),
    Output("kpi-ur", "children"),
    Output("kpi-on", "children"),
    Input("btn", "n_clicks"),
    State("tabs", "active_tab"),
    State("f-muni", "value"),
    State("f-zona", "value"),
    State("f-caracter", "value"),
    State("f-estrato", "value"),
    State("f-edu-m", "value"),
    State("f-umbral", "value"),
    State("f-area", "value"),
    prevent_initial_call=False
)
def actualizar(_, active_tab, munis, zonas, caracteres, estratos, edu_m, umbral, area):
    d = filtrar(munis, zonas, caracteres, estratos, edu_m, umbral, area)

    if d.empty:
        alerta = dbc.Alert("No hay datos con esos filtros. Prueba ampliar zona/carácter o quitar municipio.", color="warning")
        empty_kpi = kpi_card("—", "—")
        return alerta, empty_kpi, empty_kpi, empty_kpi, empty_kpi

    # KPIs
    n = len(d)
    mean = d[area].mean()

    # Brecha Urbano–Rural
    ur = "—"
    if "URBANO" in d["cole_area_ubicacion"].unique() and "RURAL" in d["cole_area_ubicacion"].unique():
        mu_u = d.loc[d["cole_area_ubicacion"] == "URBANO", area].mean()
        mu_r = d.loc[d["cole_area_ubicacion"] == "RURAL", area].mean()
        ur = f"{mu_u - mu_r:+.1f} pts"

    # Brecha Oficial–No oficial
    on = "—"
    if "OFICIAL" in d["cole_caracter"].unique() and "NO OFICIAL" in d["cole_caracter"].unique():
        mu_o = d.loc[d["cole_caracter"] == "OFICIAL", area].mean()
        mu_n = d.loc[d["cole_caracter"] == "NO OFICIAL", area].mean()
        on = f"{mu_n - mu_o:+.1f} pts"

    kpi_n = kpi_card("Estudiantes (N)", f"{n:,}".replace(",", "."))
    kpi_mean = kpi_card("Promedio", f"{mean:.1f}", AREAS.get(area, area))
    kpi_ur = kpi_card("Brecha Urbano–Rural", ur, "positivo = urbano mayor")
    kpi_on = kpi_card("Brecha No oficial–Oficial", on, "positivo = no oficial mayor")

    # ======================
    # TAB Q1: Brechas territoriales
    # ======================
    if active_tab == "tab-q1":
        # Ranking municipios (promedio) + tamaño de muestra
        muni_stats = (
            d.groupby("cole_mcpio_ubicacion")[area]
             .agg(media="mean", n="size")
             .reset_index()
             .sort_values("media", ascending=False)
        )

        fig_rank = px.bar(
            muni_stats.head(15),
            x="media",
            y="cole_mcpio_ubicacion",
            orientation="h",
            hover_data=["n"],
            title=f"Top 15 municipios por {AREAS[area]} (con filtros actuales)",
            template=PLOT_TEMPLATE
        )
        fig_rank.update_layout(height=480, margin=dict(l=10, r=10, t=60, b=10))
        fig_rank.update_xaxes(title="Promedio (pts)")
        fig_rank.update_yaxes(title="Municipio")

        # Brecha por municipio: Urbano - Rural (si aplica)
        fig_brecha_ur = None
        if set(["URBANO", "RURAL"]).issubset(set(d["cole_area_ubicacion"].unique())):
            pivot = (
                d.groupby(["cole_mcpio_ubicacion", "cole_area_ubicacion"])[area]
                 .mean().reset_index()
                 .pivot(index="cole_mcpio_ubicacion", columns="cole_area_ubicacion", values=area)
                 .dropna()
            )
            pivot["brecha_UR"] = pivot["URBANO"] - pivot["RURAL"]
            brechas = pivot["brecha_UR"].sort_values(ascending=False).reset_index()

            fig_brecha_ur = px.bar(
                brechas.head(15),
                x="brecha_UR",
                y="cole_mcpio_ubicacion",
                orientation="h",
                title="Top 15 brechas (Urbano − Rural) por municipio",
                template=PLOT_TEMPLATE
            )
            fig_brecha_ur.update_layout(height=480, margin=dict(l=10, r=10, t=60, b=10))
            fig_brecha_ur.update_xaxes(title="Brecha (pts)")
            fig_brecha_ur.update_yaxes(title="Municipio")

        # Oficial vs No oficial (box simple, legible)
        fig_car = px.box(
            d,
            x="cole_caracter",
            y=area,
            points="outliers",
            title=f"Distribución de {AREAS[area]} por carácter del establecimiento",
            template=PLOT_TEMPLATE,
            color="cole_caracter",
            color_discrete_map=COLOR_CARACTER
        )
        fig_car.update_layout(height=430, margin=dict(l=10, r=10, t=60, b=10))
        fig_car.update_xaxes(title="Carácter")
        fig_car.update_yaxes(title="Puntaje (pts)")

        # Interacción zona × carácter (promedios)
        inter = (
            d.groupby(["cole_area_ubicacion", "cole_caracter"])[area]
             .mean().reset_index()
        )
        fig_inter = px.bar(
            inter,
            x="cole_area_ubicacion",
            y=area,
            color="cole_caracter",
            barmode="group",
            title="Promedio por zona y carácter (interacción clave Q1)",
            template=PLOT_TEMPLATE,
            color_discrete_map=COLOR_CARACTER
        )
        fig_inter.update_layout(height=430, margin=dict(l=10, r=10, t=60, b=10))
        fig_inter.update_xaxes(title="Zona")
        fig_inter.update_yaxes(title="Promedio (pts)")

        return dbc.Container([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_rank), md=6),
                dbc.Col(dcc.Graph(figure=fig_brecha_ur) if fig_brecha_ur else dbc.Alert(
                    "No se puede calcular brecha Urbano−Rural con los filtros actuales (falta URBANO o RURAL).",
                    color="info"
                ), md=6),
            ], className="g-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_car), md=6),
                dbc.Col(dcc.Graph(figure=fig_inter), md=6),
            ], className="g-3"),

            dbc.Row([
                dbc.Col(dbc.Alert(
                    "Interpretación práctica: prioriza municipios con (1) promedio bajo y (2) brechas Urbano−Rural altas, "
                    "y revisa si la brecha se amplifica en OFICIAL vs NO OFICIAL.",
                    color="secondary"
                ), md=12)
            ])
        ], fluid=True), kpi_n, kpi_mean, kpi_ur, kpi_on

    # ======================
    # TAB Q2: Socioeconómico y riesgo
    # ======================
    # Heatmap estrato × educación madre (promedio)
    heat = (
        d.groupby(["fami_estratovivienda", "fami_educacionmadre"])[area]
         .mean().reset_index()
    )

    if heat.empty:
        fig_heat = None
    else:
        fig_heat = px.density_heatmap(
            heat,
            x="fami_educacionmadre",
            y="fami_estratovivienda",
            z=area,
            histfunc="avg",
            title=f"Mapa: promedio de {AREAS[area]} por estrato y educación de la madre",
            template=PLOT_TEMPLATE
        )
        fig_heat.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
        fig_heat.update_xaxes(title="Educación madre")
        fig_heat.update_yaxes(title="Estrato")

    # Riesgo: % bajo desempeño por estrato
    riesgo_estrato = (
        d.groupby("fami_estratovivienda")["bajo"]
         .mean().reset_index()
    )
    riesgo_estrato["riesgo_pct"] = 100 * riesgo_estrato["bajo"]

    fig_riesgo = px.bar(
        riesgo_estrato.sort_values("riesgo_pct", ascending=False),
        x="fami_estratovivienda",
        y="riesgo_pct",
        title=f"Riesgo (% bajo desempeño < {umbral}) por estrato",
        template=PLOT_TEMPLATE
    )
    fig_riesgo.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    fig_riesgo.update_xaxes(title="Estrato")
    fig_riesgo.update_yaxes(title="% en bajo desempeño")

    # Box por estrato (legible)
    fig_box_e = px.box(
        d,
        x="fami_estratovivienda",
        y=area,
        title=f"Distribución de {AREAS[area]} por estrato",
        template=PLOT_TEMPLATE
    )
    fig_box_e.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    fig_box_e.update_xaxes(title="Estrato")
    fig_box_e.update_yaxes(title="Puntaje (pts)")

    return dbc.Container([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_heat) if fig_heat else dbc.Alert(
                "No hay suficientes combinaciones para el heatmap con los filtros actuales.",
                color="info"
            ), md=12),
        ], className="g-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_riesgo), md=6),
            dbc.Col(dcc.Graph(figure=fig_box_e), md=6),
        ], className="g-3"),

        dbc.Row([
            dbc.Col(dbc.Alert(
                "Interpretación práctica: los grupos de mayor riesgo son los que combinan estratos bajos "
                "con menor educación parental y alta proporción bajo el umbral. Úsalo para focalizar intervención.",
                color="secondary"
            ), md=12)
        ])
    ], fluid=True), kpi_n, kpi_mean, kpi_ur, kpi_on


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=8050)