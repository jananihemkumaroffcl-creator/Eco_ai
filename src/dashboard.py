import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import psutil
from sklearn.linear_model import LinearRegression
psutil.cpu_percent(interval=None)  # initialize CPU measurement
CARBON_BUDGET = 0.002

app = dash.Dash(__name__)
server = app.server


# ---------------- LOAD LOGS ----------------
def load_logs():
    # codecarbon final summaries — used for Overview/comparison
    eco = pd.read_csv("data/energy_log.csv")
    base = pd.read_csv("data/baseline_energy_log.csv")

    # per-epoch log — used for Analytics graph
    epoch_log = pd.read_csv("data/eco_epoch_log.csv")
    epoch_log["timestamp"] = pd.to_datetime(epoch_log["timestamp"])
    epoch_log["energy_consumed"] = pd.to_numeric(epoch_log["energy_consumed"], errors="coerce")
    epoch_log = epoch_log.sort_values("timestamp")
    epoch_log["timestamp"] = epoch_log["timestamp"] + pd.to_timedelta(np.arange(len(epoch_log)), unit="ms")

    return eco, base, epoch_log


# ---------------- LAYOUT ----------------
app.layout = html.Div([

    html.H1(
        "🌱 Eco-AI Carbon Intelligence Dashboard",
        style={"textAlign": "center", "fontSize": "36px", "fontWeight": "bold"}
    ),

    dcc.Tabs([

        # ---------- TAB 1 OVERVIEW ----------
        dcc.Tab(label="Overview", children=[

            html.Div(id="top-cards",
                     style={"display": "flex",
                            "justifyContent": "space-around",
                            "marginBottom": "30px"}),

            dcc.Graph(id="carbon-gauge"),

            dcc.Graph(id="saving-gauge"),

            dcc.Graph(id="comparison-bar"),

        ]),

        # ---------- TAB 2 ANALYTICS ----------
        dcc.Tab(label="Energy Analytics", children=[

            html.Div([
                html.Label("Select Metric"),
                dcc.Dropdown(
                    id="metric-select",
                    options=[
                        {"label": "Energy Consumption", "value": "energy_consumed"},
                        {"label": "CO₂ Emissions", "value": "emissions"},
                        {"label": "Duration", "value": "duration"}
                    ],
                    value="energy_consumed",
                    style={"width": "300px",
                           "color": "#101227"}
                )
            ], style={"padding": "20px"}),

            dcc.Graph(id="energy-graph"),

            html.Div(id="bottom-cards",
                     style={"display": "flex",
                            "justifyContent": "space-around",
                            "marginTop": "30px"}),

        ]),

        # ---------- TAB 3 SYSTEM ----------
        dcc.Tab(label="System Monitor", children=[

            html.Div([
                dcc.Graph(id="cpu-gauge", style={"width": "50%"}),
                dcc.Graph(id="ram-gauge", style={"width": "50%"})
            ], style={"display": "flex"})

        ])

    ]),

    dcc.Interval(id="interval", interval=3000)

], style={"padding": "20px", "fontFamily": "Arial"})


# ---------------- CALLBACK ----------------
@app.callback(
    [
        Output("top-cards", "children"),
        Output("carbon-gauge", "figure"),
        Output("saving-gauge", "figure"),
        Output("comparison-bar", "figure"),
        Output("energy-graph", "figure"),
        Output("cpu-gauge", "figure"),
        Output("ram-gauge", "figure"),
        Output("bottom-cards", "children"),
    ],
    [
        Input("interval", "n_intervals"),
        Input("metric-select", "value")
    ]
)
def update(n, metric):

    eco, base, epoch_log = load_logs()

    eco_last = eco.iloc[-1]
    base_last = base.iloc[-1]

    # ---------- TOP CARDS ----------
    def card(title, value):
        return html.Div([
            html.H4(title),
            html.H2(value)
        ],
            style={
                "padding": "15px",
                "backgroundColor": "#712525",
                "borderRadius": "10px",
                "width": "22%",
                "textAlign": "center",
                "boxShadow": "2px 2px 6px rgba(0,0,0,0.1)"
            })

    top_cards = [

        card("🌍 Eco CO₂", f"{eco_last['emissions']:.8f} kg"),

        card("🔥 Baseline CO₂", f"{base_last['emissions']:.8f} kg"),

        card("⚡ Energy Used", f"{eco_last['energy_consumed']:.8f} kWh"),

        card("⏱ Runtime", f"{eco_last['duration']:.2f} s")
    ]

    # ---------- CARBON BUDGET ----------
    usage = eco_last["emissions"]

    carbon_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=usage,
        number={'suffix': " kg"},
        title={'text': "Carbon Emission"},
        gauge={
            'axis': {'range': [0, CARBON_BUDGET]},
            'bar': {'color': "#379450"},
            'steps': [
                {'range': [0, CARBON_BUDGET * 0.5], 'color': "#D4EFDF"},
                {'range': [CARBON_BUDGET * 0.5, CARBON_BUDGET * 0.8], 'color': "#F9E79F"},
                {'range': [CARBON_BUDGET * 0.8, CARBON_BUDGET], 'color': "#F5B7B1"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': CARBON_BUDGET
            }
        }
    ))

    # ---------- CARBON SAVINGS ----------
    saving = (base_last["emissions"] - eco_last["emissions"]) / base_last["emissions"] * 100

    saving_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=saving,
        number={"suffix": "%"},
        title={"text": "♻ Carbon Saved"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#379450"},
            "steps": [
                {"range": [0, 40], "color": "#F5B7B1"},   # red zone
                {"range": [40, 70], "color": "#F9E79F"},  # yellow zone
                {"range": [70, 100], "color": "#D4EFDF"}  # green zone
            ],
        }
    ))

#------- COMPARISON ----------

    comp = pd.DataFrame({
        "Type": ["Baseline", "Eco-AI"],
        "CO2": [base_last["emissions"] * 1e6,
                eco_last["emissions"] * 1e6]
    })

    comp_fig = px.bar(
        comp,
        x="Type",
        y="CO2",
        title=f"Carbon Comparison (µg) — Eco-AI saves {saving:.1f}%",
        color="Type",
        color_discrete_map={
            "Baseline": "#E68585",
            "Eco-AI": "#81C09B"
        },
        text=comp["CO2"].apply(lambda v: f"{v:.2f} µg")
    )
    comp_fig.update_traces(textposition="outside")
    comp_fig.update_layout(yaxis_title="CO₂ (µg)", showlegend=False)

    # ---------- ENERGY GRAPH ----------
    epoch_log["model"] = "Eco-AI"
    base_epoch = pd.DataFrame({
        "timestamp": pd.to_datetime(base["timestamp"]),
        "energy_consumed": base["energy_consumed"],
        "emissions": base["emissions"],
        "duration": base["duration"],
        "model": "Baseline"
    })
    combined = pd.concat([epoch_log, base_epoch], ignore_index=True)

    energy_fig = px.line(
        combined,
        x="duration",
        y=metric,
        color="model",
        markers=True,
        title=f"{metric} — Eco-AI vs Baseline",
        color_discrete_map={"Eco-AI": "#81C09B", "Baseline": "#E68585"}
    )
    energy_fig.update_layout(transition_duration=500, xaxis_title="Epoch / Step")

    # ---------- CPU ----------
    cpu = psutil.cpu_percent(interval=0.5)

    cpu_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cpu,
        number={'suffix': "%"},
        title={'text': "CPU Usage"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#0B9429"},
            "steps": [
                {"range": [0, 50], "color": "#D4EFDF"},
                {"range": [50, 80], "color": "#F9E79F"},
                {"range": [80, 100], "color": "#F5B7B1"}
            ]
        }
    ))

    # ---------- RAM ----------
    ram = psutil.virtual_memory().percent

    ram_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ram,
        number={"suffix": "%"},
        title={"text": "RAM Usage"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#0B9429"},
            "steps": [
                {"range": [0, 50], "color": "#D4EFDF"},
                {"range": [50, 80], "color": "#F9E79F"},
                {"range": [80, 100], "color": "#F5B7B1"}
            ]
        }
    ))

    # ---------- ENERGY PREDICTION ----------
    X = np.arange(len(epoch_log)).reshape(-1, 1)
    y = epoch_log["energy_consumed"].values

    model = LinearRegression().fit(X, y)

    future = model.predict([[len(epoch_log) + 50]])[0]

    prediction_card = html.Div([
        html.H4("🔮 Future Energy Prediction"),
        html.H2(f"{future:.8f} kWh")
    ],
        style={
            "padding": "15px",
            "backgroundColor": "#12274B",
            "borderRadius": "10px",
            "width": "45%",
            "textAlign": "center"
        })

    efficiency_card = html.Div([
        html.H4("🏆 Eco-AI Efficiency Score"),
        html.H2(f"{int(saving)}/100"),
        html.P(f"{saving:.1f}% CO₂ reduction")
    ],
        style={
            "padding": "15px",
            "backgroundColor": "#12274B",
            "borderRadius": "10px",
            "width": "45%",
            "textAlign": "center"
        })

    bottom_cards = [prediction_card, efficiency_card]

    return (
        top_cards,
        carbon_gauge,
        saving_gauge,
        comp_fig,
        energy_fig,
        cpu_gauge,
        ram_gauge,
        bottom_cards
    )


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)