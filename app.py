"""Clinic Operations Analytics.

Adapted from Plotly's open-source `dash-clinical-analytics` sample app (MIT licence,
see LICENSE). Extended by Pawel Sogra with a KPI summary, auto-generated findings,
a wait-time vs patient-experience analysis, an offline model check, a faster data
pipeline and compatibility updates for current Dash / pandas.
"""
import datetime
import json
import os
import pathlib
from datetime import datetime as dt

import dash
import numpy as np
import pandas as pd
from dash import ClientsideFunction, Input, Output, dcc, html

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Clinic Operations Analytics | Pawel Sogra"

server = app.server
app.config.suppress_callback_exceptions = True

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

# Read data
df = pd.read_csv(DATA_PATH.joinpath("clinical_analytics.csv.gz"))

clinic_list = df["Clinic Name"].unique()
df["Admit Source"] = df["Admit Source"].fillna("Not Identified")
admit_list = df["Admit Source"].unique().tolist()

# Date
# Parse check-in time once (vectorised) and derive weekday / hour labels
df["Check-In Time"] = pd.to_datetime(
    df["Check-In Time"], format="%Y-%m-%d %I:%M:%S %p"
)
df["Days of Wk"] = df["Check-In Time"].dt.strftime("%A")  # e.g. "Monday"
df["Check-In Hour"] = df["Check-In Time"].dt.strftime("%I %p")  # e.g. "11 PM"

# Offline model check (see scripts/model_check.py)
try:
    MODEL_CHECK = json.loads(DATA_PATH.joinpath("model_check.json").read_text())
except FileNotFoundError:
    MODEL_CHECK = None

PORTFOLIO_URL = os.environ.get("PORTFOLIO_URL", "")

day_list = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

# Register all departments for callbacks
all_departments = df["Department"].unique().tolist()
wait_time_inputs = [
    Input((i + "_wait_time_graph"), "selectedData") for i in all_departments
]
score_inputs = [Input((i + "_score_graph"), "selectedData") for i in all_departments]


def description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Clinic Operations Analytics"),
            html.H3("How long do patients wait, and what does it cost in experience?"),
            html.Div(
                id="intro",
                children="Explore patient volume by time of day, waiting time and care score. "
                "Click a heatmap cell to drill into that hour, and use the wait-time target below "
                "to see how longer waits relate to patient experience.",
            ),
        ],
    )


def generate_control_card():
    """

    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P("Select Clinic"),
            dcc.Dropdown(
                id="clinic-select",
                options=[{"label": i, "value": i} for i in clinic_list],
                value=clinic_list[0],
            ),
            html.Br(),
            html.P("Select Check-In Time"),
            dcc.DatePickerRange(
                id="date-picker-select",
                start_date=dt(2014, 1, 1),
                end_date=dt(2014, 1, 15),
                min_date_allowed=dt(2014, 1, 1),
                max_date_allowed=dt(2014, 12, 31),
                initial_visible_month=dt(2014, 1, 1),
            ),
            html.Br(),
            html.Br(),
            html.P("Select Admit Source"),
            dcc.Dropdown(
                id="admit-select",
                options=[{"label": i, "value": i} for i in admit_list],
                value=admit_list[:],
                multi=True,
            ),
            html.Br(),
            # html.Div(
            #     id="reset-btn-outer",
            #     children=html.Button(id="reset-btn", children="Reset", n_clicks=0),
            # ),
        ],
    )


def generate_patient_volume_heatmap(start, end, clinic, hm_click, admit_type, reset):
    """
    :param: start: start date from selection.
    :param: end: end date from selection.
    :param: clinic: clinic from selection.
    :param: hm_click: clickData from heatmap.
    :param: admit_type: admission type from selection.
    :param: reset (boolean): reset heatmap graph if True.

    :return: Patient volume annotated heatmap.
    """

    filtered_df = df[
        (df["Clinic Name"] == clinic) & (df["Admit Source"].isin(admit_type))
    ]
    filtered_df = filtered_df.sort_values("Check-In Time").set_index("Check-In Time")[
        start:end
    ]

    x_axis = [datetime.time(i).strftime("%I %p") for i in range(24)]  # 24hr time list
    y_axis = day_list

    hour_of_day = ""
    weekday = ""
    shapes = []

    if hm_click is not None:
        hour_of_day = hm_click["points"][0]["x"]
        weekday = hm_click["points"][0]["y"]

        # Add shapes
        x0 = x_axis.index(hour_of_day) / 24
        x1 = x0 + 1 / 24
        y0 = y_axis.index(weekday) / 7
        y1 = y0 + 1 / 7

        shapes = [
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                line=dict(color="#ff6347"),
            )
        ]

    # Get z value : sum(number of records) based on x, y (vectorised)
    counts = (
        filtered_df.groupby(["Days of Wk", "Check-In Hour"])["Number of Records"]
        .sum()
        .to_dict()
    )
    z = np.zeros((7, 24))
    annotations = []

    for ind_y, day in enumerate(y_axis):
        for ind_x, x_val in enumerate(x_axis):
            sum_of_record = counts.get((day, x_val), 0)
            z[ind_y][ind_x] = sum_of_record

            annotation_dict = dict(
                showarrow=False,
                text="<b>" + str(sum_of_record) + "<b>",
                xref="x",
                yref="y",
                x=x_val,
                y=day,
                font=dict(family="sans-serif"),
            )
            # Highlight annotation text by self-click
            if x_val == hour_of_day and day == weekday:
                if not reset:
                    annotation_dict.update(size=15, font=dict(color="#ff6347"))

            annotations.append(annotation_dict)

    # Heatmap
    hovertemplate = "<b> %{y}  %{x} <br><br> %{z} Patient Records"

    data = [
        dict(
            x=x_axis,
            y=y_axis,
            z=z,
            type="heatmap",
            name="",
            hovertemplate=hovertemplate,
            showscale=False,
            colorscale=[[0, "#caf3ff"], [1, "#2c82ff"]],
        )
    ]

    layout = dict(
        margin=dict(l=70, b=50, t=50, r=50),
        modebar={"orientation": "v"},
        font=dict(family="Open Sans"),
        annotations=annotations,
        shapes=shapes,
        xaxis=dict(
            side="top",
            ticks="",
            ticklen=2,
            tickfont=dict(family="sans-serif"),
            tickcolor="#ffffff",
        ),
        yaxis=dict(
            side="left", ticks="", tickfont=dict(family="sans-serif"), ticksuffix=" "
        ),
        hovermode="closest",
        showlegend=False,
    )
    return {"data": data, "layout": layout}


def generate_table_row(id, style, col1, col2, col3):
    """ Generate table rows.

    :param id: The ID of table row.
    :param style: Css style of this row.
    :param col1 (dict): Defining id and children for the first column.
    :param col2 (dict): Defining id and children for the second column.
    :param col3 (dict): Defining id and children for the third column.
    """

    return html.Div(
        id=id,
        className="row table-row",
        style=style,
        children=[
            html.Div(
                id=col1["id"],
                style={"display": "table", "height": "100%"},
                className="two columns row-department",
                children=col1["children"],
            ),
            html.Div(
                id=col2["id"],
                style={"textAlign": "center", "height": "100%"},
                className="five columns",
                children=col2["children"],
            ),
            html.Div(
                id=col3["id"],
                style={"textAlign": "center", "height": "100%"},
                className="five columns",
                children=col3["children"],
            ),
        ],
    )


def generate_table_row_helper(department):
    """Helper function.

    :param: department (string): Name of department.
    :return: Table row.
    """
    return generate_table_row(
        department,
        {},
        {"id": department + "_department", "children": html.B(department)},
        {
            "id": department + "wait_time",
            "children": dcc.Graph(
                id=department + "_wait_time_graph",
                style={"height": "100%", "width": "100%"},
                className="wait_time_graph",
                config={
                    "staticPlot": False,
                    "editable": False,
                    "displayModeBar": False,
                },
                figure={
                    "layout": dict(
                        margin=dict(l=0, r=0, b=0, t=0, pad=0),
                        xaxis=dict(
                            showgrid=False,
                            showline=False,
                            showticklabels=False,
                            zeroline=False,
                        ),
                        yaxis=dict(
                            showgrid=False,
                            showline=False,
                            showticklabels=False,
                            zeroline=False,
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                },
            ),
        },
        {
            "id": department + "_patient_score",
            "children": dcc.Graph(
                id=department + "_score_graph",
                style={"height": "100%", "width": "100%"},
                className="patient_score_graph",
                config={
                    "staticPlot": False,
                    "editable": False,
                    "displayModeBar": False,
                },
                figure={
                    "layout": dict(
                        margin=dict(l=0, r=0, b=0, t=0, pad=0),
                        xaxis=dict(
                            showgrid=False,
                            showline=False,
                            showticklabels=False,
                            zeroline=False,
                        ),
                        yaxis=dict(
                            showgrid=False,
                            showline=False,
                            showticklabels=False,
                            zeroline=False,
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                },
            ),
        },
    )


def initialize_table():
    """
    :return: empty table children. This is intialized for registering all figure ID at page load.
    """

    # header_row
    header = [
        generate_table_row(
            "header",
            {"height": "50px"},
            {"id": "header_department", "children": html.B("Department")},
            {"id": "header_wait_time_min", "children": html.B("Wait Time Minutes")},
            {"id": "header_care_score", "children": html.B("Care Score")},
        )
    ]

    # department_row
    rows = [generate_table_row_helper(department) for department in all_departments]
    header.extend(rows)
    empty_table = header

    return empty_table


def generate_patient_table(figure_list, departments, wait_time_xrange, score_xrange):
    """
    :param score_xrange: score plot xrange [min, max].
    :param wait_time_xrange: wait time plot xrange [min, max].
    :param figure_list:  A list of figures from current selected metrix.
    :param departments:  List of departments for making table.
    :return: Patient table.
    """
    # header_row
    header = [
        generate_table_row(
            "header",
            {"height": "50px"},
            {"id": "header_department", "children": html.B("Department")},
            {"id": "header_wait_time_min", "children": html.B("Wait Time Minutes")},
            {"id": "header_care_score", "children": html.B("Care Score")},
        )
    ]

    # department_row
    # print(type(departments))
    # print(departments)
    rows = [generate_table_row_helper(department) for department in departments]
    # empty_row
    empty_departments = [item for item in all_departments if item not in departments]
    empty_rows = [
        generate_table_row_helper(department) for department in empty_departments
    ]

    # fill figures into row contents and hide empty rows
    for ind, department in enumerate(departments):
        rows[ind].children[1].children.figure = figure_list[ind]
        rows[ind].children[2].children.figure = figure_list[ind + len(departments)]
    for row in empty_rows[1:]:
        row.style = {"display": "none"}

    # convert empty row[0] to axis row
    empty_rows[0].children[0].children = html.B(
        "graph_ax", style={"visibility": "hidden"}
    )

    empty_rows[0].children[1].children.figure["layout"].update(
        dict(margin=dict(t=-70, b=50, l=0, r=0, pad=0))
    )

    empty_rows[0].children[1].children.config["staticPlot"] = True

    empty_rows[0].children[1].children.figure["layout"]["xaxis"].update(
        dict(
            showline=True,
            showticklabels=True,
            tick0=0,
            dtick=20,
            range=wait_time_xrange,
        )
    )
    empty_rows[0].children[2].children.figure["layout"].update(
        dict(margin=dict(t=-70, b=50, l=0, r=0, pad=0))
    )

    empty_rows[0].children[2].children.config["staticPlot"] = True

    empty_rows[0].children[2].children.figure["layout"]["xaxis"].update(
        dict(showline=True, showticklabels=True, tick0=0, dtick=0.5, range=score_xrange)
    )

    header.extend(rows)
    header.extend(empty_rows)
    return header


def create_table_figure(
    department, filtered_df, category, category_xrange, selected_index
):
    """Create figures.

    :param department: Name of department.
    :param filtered_df: Filtered dataframe.
    :param category: Defining category of figure, either 'wait time' or 'care score'.
    :param category_xrange: x axis range for this figure.
    :param selected_index: selected point index.
    :return: Plotly figure dictionary.
    """
    aggregation = {
        "Wait Time Min": "mean",
        "Care Score": "mean",
        "Days of Wk": "first",
        "Check-In Time": "first",
        "Check-In Hour": "first",
    }

    df_by_department = filtered_df[
        filtered_df["Department"] == department
    ].reset_index()
    grouped = (
        df_by_department.groupby("Encounter Number").agg(aggregation).reset_index()
    )
    patient_id_list = grouped["Encounter Number"]

    x = grouped[category]
    y = list(department for _ in range(len(x)))

    f = lambda x_val: dt.strftime(x_val, "%Y-%m-%d")
    check_in = (
        grouped["Check-In Time"].apply(f)
        + " "
        + grouped["Days of Wk"]
        + " "
        + grouped["Check-In Hour"].map(str)
    )

    text_wait_time = (
        "Patient # : "
        + patient_id_list
        + "<br>Check-in Time: "
        + check_in
        + "<br>Wait Time: "
        + grouped["Wait Time Min"].round(decimals=1).map(str)
        + " Minutes,  Care Score : "
        + grouped["Care Score"].round(decimals=1).map(str)
    )

    layout = dict(
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        clickmode="event+select",
        hovermode="closest",
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            range=category_xrange,
        ),
        yaxis=dict(
            showgrid=False, showline=False, showticklabels=False, zeroline=False
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    trace = dict(
        x=x,
        y=y,
        mode="markers",
        marker=dict(size=14, line=dict(width=1, color="#ffffff")),
        color="#2c82ff",
        selected=dict(marker=dict(color="#ff6347", opacity=1)),
        unselected=dict(marker=dict(opacity=0.1)),
        selectedpoints=selected_index,
        hoverinfo="text",
        customdata=patient_id_list,
        text=text_wait_time,
    )

    return {"data": [trace], "layout": layout}


WAIT_BINS = [-np.inf, 30, 60, 90, 120, np.inf]
WAIT_LABELS = ["Under 30", "30-60", "60-90", "90-120", "120+"]


def filter_records(start, end, clinic, admit_type):
    """Filter the dataset by clinic, admit source and check-in date range."""
    out = df[(df["Clinic Name"] == clinic) & (df["Admit Source"].isin(admit_type))]
    return out[(out["Check-In Time"] >= start) & (out["Check-In Time"] <= end)]


def kpi_tile(label, value, sub=""):
    return html.Div(
        className="kpi-tile",
        children=[
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(sub, className="kpi-sub"),
        ],
    )


def build_findings(d, threshold):
    """Plain-English findings for the current selection."""
    if d.empty:
        return [html.Li("No patient records in this selection.")]
    items = []
    slot = d.groupby(["Days of Wk", "Check-In Hour"])["Number of Records"].sum()
    (day, hour), n = slot.idxmax(), int(slot.max())
    items.append(html.Li(f"Busiest check-in slot: {day} at {hour} ({n:,} records)."))

    by_dept = d.groupby("Department")["Wait Time Min"].agg(["mean", "size"])
    by_dept = by_dept[by_dept["size"] >= 30]
    if len(by_dept) >= 2:
        worst = by_dept["mean"].idxmax()
        items.append(
            html.Li(
                f"Longest average wait: {worst} ({by_dept['mean'].max():.0f} min, "
                f"vs {d['Wait Time Min'].mean():.0f} min overall)."
            )
        )

    if len(d) > 30 and d["Wait Time Min"].std() > 0 and d["Care Score"].std() > 0:
        slope = np.polyfit(d["Wait Time Min"], d["Care Score"], 1)[0]
        r = d["Wait Time Min"].corr(d["Care Score"])
        items.append(
            html.Li(
                f"Each extra 10 minutes of waiting goes with a care score about "
                f"{abs(slope) * 10:.2f} points {'lower' if slope < 0 else 'higher'} "
                f"(correlation {r:.2f}). This is an association, not proof of cause."
            )
        )
    return items


def experience_figure(d):
    """Average care score by wait-time band."""
    bands = pd.cut(d["Wait Time Min"], WAIT_BINS, labels=WAIT_LABELS)
    g = (
        d.assign(band=bands)
        .groupby("band", observed=False)["Care Score"]
        .agg(["mean", "size"])
        .reindex(WAIT_LABELS)
    )
    fig = {
        "data": [
            dict(
                type="bar",
                x=WAIT_LABELS,
                y=g["mean"].round(2).tolist(),
                customdata=g["size"].tolist(),
                marker=dict(color="#2c82ff"),
                hovertemplate="Wait %{x} min<br>Avg care score %{y}<br>%{customdata:,} records<extra></extra>",
            )
        ],
        "layout": dict(
            margin=dict(l=50, r=10, t=10, b=50),
            xaxis=dict(title=dict(text="Wait time (minutes)")),
            yaxis=dict(title=dict(text="Average care score"), rangemode="tozero"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
        ),
    }
    return fig


def model_check_card():
    if not MODEL_CHECK:
        return html.Div()
    m = MODEL_CHECK
    fig = {
        "data": [
            dict(
                type="bar",
                x=["Always guess the average", "Gradient-boosted model"],
                y=[m["mae_baseline"], m["mae_model"]],
                marker=dict(color=["#b8c4d6", "#2c82ff"]),
                text=[m["mae_baseline"], m["mae_model"]],
                textposition="outside",
                hoverinfo="skip",
            )
        ],
        "layout": dict(
            margin=dict(l=50, r=10, t=10, b=40),
            yaxis=dict(title=dict(text="Mean absolute error (min)"), range=[0, 45]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=260,
        ),
    }
    return html.Div(
        id="model_check_card",
        children=[
            html.B("Model check: can we predict waiting time?"),
            html.Hr(),
            html.P(
                "I tested whether hour, weekday, month, clinic, department and admit source "
                "can predict how long a patient waits. A gradient-boosted model was trained on "
                f"January to September ({m['train_rows']:,} records) and tested on October to "
                f"December ({m['test_rows']:,} records)."
            ),
            html.P(
                f"Result: it does no better than guessing the average (error {m['mae_model']} vs "
                f"{m['mae_baseline']} minutes, R-squared {m['r2_test']}). In this dataset, waiting "
                "time is not explained by when or where a patient arrives. Something not recorded "
                "here, such as staffing or case complexity, would need to be added before a "
                "forecasting model is worth deploying. Reporting a null result like this "
                "keeps a model from being shipped on false hope."
            ),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            html.Small(
                f"Reproduce with scripts/model_check.py. Data note: {m['negative_wait_rows']} records "
                "have a negative wait time and are left as-is."
            ),
        ],
    )


def showcase_note():
    """Banner making clear this is a showcase on demonstration data, not a real client project."""
    link = html.A(
        "Discuss a similar project \u2192",
        href=(PORTFOLIO_URL.rstrip("/") + "/#contact") if PORTFOLIO_URL else "#",
        target="_blank",
        style={"marginLeft": "10px", "fontWeight": 600, "color": "#b0432d", "whiteSpace": "nowrap"},
    )
    return html.Div(
        id="showcase-note",
        children=[
            html.Strong("Showcase project. "),
            "Built to show what is possible, using demonstration data. This is not a real client project. "
            "The same approach works on your own data. ",
            link if PORTFOLIO_URL else "",
        ],
        style={
            "background": "#fff7ed", "color": "#7a4a12", "borderBottom": "1px solid #f3d9b8",
            "padding": "10px 24px", "fontSize": "14px", "lineHeight": "1.5", "textAlign": "center",
            "clear": "both",
        },
    )


app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[
                html.Div(
                    [
                        html.Span("Pawel Sogra", className="banner-name"),
                        html.Span(
                            "Data Science & AI | Live demo", className="banner-sub"
                        ),
                    ]
                ),
                html.A(
                    "Back to portfolio",
                    href=PORTFOLIO_URL,
                    className="banner-link",
                    style={} if PORTFOLIO_URL else {"display": "none"},
                ),
            ],
        ),
        showcase_note(),
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[
                description_card(),
                generate_control_card(),
                html.Div(
                    id="findings-card",
                    children=[
                        html.B("Key findings for this selection"),
                        html.Ul(id="findings-list"),
                    ],
                ),
            ]
            + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none"}
                )
            ],
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                html.Div(id="kpi-row"),
                # Patient Volume Heatmap
                html.Div(
                    id="patient_volume_card",
                    children=[
                        html.B("Patient Volume"),
                        html.Hr(),
                        dcc.Graph(id="patient_volume_hm"),
                        html.Div(
                            id="reset-btn-outer",
                            children=html.Button(id="reset-btn", children="Show All", n_clicks=0), 
                        ),
                    ],
                ),
                # Wait time vs patient experience
                html.Div(
                    id="experience_card",
                    children=[
                        html.B("Does waiting longer cost patient experience?"),
                        html.Hr(),
                        html.P("Set a wait-time target (minutes):", id="target-label"),
                        dcc.Slider(
                            id="wait-target",
                            min=30,
                            max=150,
                            step=10,
                            value=90,
                            marks={v: str(v) for v in range(30, 151, 30)},
                        ),
                        html.Div(id="target-summary"),
                        dcc.Graph(
                            id="experience_graph", config={"displayModeBar": False}
                        ),
                    ],
                ),
                # Patient Wait time by Department
                html.Div(
                    id="wait_time_card",
                    children=[
                        html.B("Patient Wait Time and Satisfactory Scores"),
                        html.Hr(),
                        html.Div(id="wait_time_table", children=initialize_table()),
                    ],
                ),
                model_check_card(),
            ],
        ),
        html.Div(
            id="footer",
            children=[
                "Demonstration dataset bundled with Plotly's sample app; treat as illustrative, "
                "not real clinical data. Adapted from Plotly's open-source ",
                html.A(
                    "dash-clinical-analytics",
                    href="https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-clinical-analytics",
                    target="_blank",
                ),
                " (MIT licence). Extended by Pawel Sogra: KPIs, findings, wait-vs-experience "
                "analysis, model check, faster data pipeline and current Dash/pandas support.",
            ],
        ),
    ],
)


@app.callback(
    Output("patient_volume_hm", "figure"),
    [
        Input("date-picker-select", "start_date"),
        Input("date-picker-select", "end_date"),
        Input("clinic-select", "value"),
        Input("patient_volume_hm", "clickData"),
        Input("admit-select", "value"),
        Input("reset-btn", "n_clicks"),
    ],
)
def update_heatmap(start, end, clinic, hm_click, admit_type, reset_click):
    start = start + " 00:00:00"
    end = end + " 23:59:59"

    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return generate_patient_volume_heatmap(
        start, end, clinic, hm_click, admit_type, reset
    )


@app.callback(
    [
        Output("kpi-row", "children"),
        Output("findings-list", "children"),
        Output("experience_graph", "figure"),
        Output("target-summary", "children"),
    ],
    [
        Input("date-picker-select", "start_date"),
        Input("date-picker-select", "end_date"),
        Input("clinic-select", "value"),
        Input("admit-select", "value"),
        Input("wait-target", "value"),
    ],
)
def update_insights(start, end, clinic, admit_type, threshold):
    start = start + " 00:00:00"
    end = end + " 23:59:59"
    d = filter_records(start, end, clinic, admit_type or [])

    if d.empty:
        empty = {"data": [], "layout": dict(height=300)}
        return (
            html.Div(className="kpi-row", children=[kpi_tile("Patient records", "0")]),
            build_findings(d, threshold),
            empty,
            html.P("No records in this selection."),
        )

    over = d["Wait Time Min"] > threshold
    pct_over = 100 * over.mean()
    kpis = html.Div(
        className="kpi-row",
        children=[
            kpi_tile("Patient records", f"{int(d['Number of Records'].sum()):,}"),
            kpi_tile("Average wait", f"{d['Wait Time Min'].mean():.0f} min"),
            kpi_tile("Average care score", f"{d['Care Score'].mean():.2f}"),
            kpi_tile(
                f"Waits over {threshold} min",
                f"{pct_over:.0f}%",
                f"{int(over.sum()):,} records",
            ),
        ],
    )
    if over.any() and (~over).any():
        summary = html.P(
            f"{pct_over:.0f}% of records waited longer than {threshold} minutes. "
            f"Their average care score was {d.loc[over, 'Care Score'].mean():.2f}, compared with "
            f"{d.loc[~over, 'Care Score'].mean():.2f} for patients seen within the target.",
            className="target-summary",
        )
    else:
        summary = html.P(
            "All records fall on one side of this target. Move the slider to compare.",
            className="target-summary",
        )
    return kpis, build_findings(d, threshold), experience_figure(d), summary


app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("wait_time_table", "children")] + wait_time_inputs + score_inputs,
)


@app.callback(
    Output("wait_time_table", "children"),
    [
        Input("date-picker-select", "start_date"),
        Input("date-picker-select", "end_date"),
        Input("clinic-select", "value"),
        Input("admit-select", "value"),
        Input("patient_volume_hm", "clickData"),
        Input("reset-btn", "n_clicks"),
    ]
    + wait_time_inputs
    + score_inputs,
)
def update_table(start, end, clinic, admit_type, heatmap_click, reset_click, *args):
    start = start + " 00:00:00"
    end = end + " 23:59:59"

    # Find which one has been triggered
    ctx = dash.callback_context

    prop_id = ""
    prop_type = ""
    triggered_value = None
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        prop_type = ctx.triggered[0]["prop_id"].split(".")[1]
        triggered_value = ctx.triggered[0]["value"]

    # filter data
    filtered_df = df[
        (df["Clinic Name"] == clinic) & (df["Admit Source"].isin(admit_type))
    ]
    filtered_df = filtered_df.sort_values("Check-In Time").set_index("Check-In Time")[
        start:end
    ]
    departments = filtered_df["Department"].unique()

    # Highlight click data's patients in this table
    if heatmap_click is not None and prop_id != "reset-btn":
        hour_of_day = heatmap_click["points"][0]["x"]
        weekday = heatmap_click["points"][0]["y"]
        clicked_df = filtered_df[
            (filtered_df["Days of Wk"] == weekday)
            & (filtered_df["Check-In Hour"] == hour_of_day)
        ]  # slice based on clicked weekday and hour
        departments = clicked_df["Department"].unique()
        filtered_df = clicked_df

    # range_x for all plots
    wait_time_xrange = [
        filtered_df["Wait Time Min"].min() - 2,
        filtered_df["Wait Time Min"].max() + 2,
    ]
    score_xrange = [
        filtered_df["Care Score"].min() - 0.5,
        filtered_df["Care Score"].max() + 0.5,
    ]

    figure_list = []

    if prop_type != "selectedData" or (
        prop_type == "selectedData" and triggered_value is None
    ):  # Default condition, all ""

        for department in departments:
            department_wait_time_figure = create_table_figure(
                department, filtered_df, "Wait Time Min", wait_time_xrange, ""
            )
            figure_list.append(department_wait_time_figure)

        for department in departments:
            department_score_figure = create_table_figure(
                department, filtered_df, "Care Score", score_xrange, ""
            )
            figure_list.append(department_score_figure)

    elif prop_type == "selectedData":
        selected_patient = ctx.triggered[0]["value"]["points"][0]["customdata"]
        selected_index = [ctx.triggered[0]["value"]["points"][0]["pointIndex"]]

        # [] turn on un-selection for all other plots, [index] for this department
        for department in departments:
            wait_selected_index = []
            if prop_id.split("_")[0] == department:
                wait_selected_index = selected_index

            department_wait_time_figure = create_table_figure(
                department,
                filtered_df,
                "Wait Time Min",
                wait_time_xrange,
                wait_selected_index,
            )
            figure_list.append(department_wait_time_figure)

        for department in departments:
            score_selected_index = []
            if department == prop_id.split("_")[0]:
                score_selected_index = selected_index

            department_score_figure = create_table_figure(
                department,
                filtered_df,
                "Care Score",
                score_xrange,
                score_selected_index,
            )
            figure_list.append(department_score_figure)
    # print('BREAKBREAKBREAKBREAK')
    # departments = np.insert(departments, 0, None)
    # figure_list[0]['layout']['margin']['b'] = 1
    # print(figure_list[0]['data'][0])
    # print(figure_list[0]['layout']['margin']['t'])
    # Put figures in table
    table = generate_patient_table(
        figure_list, departments, wait_time_xrange, score_xrange
    )
    return table


# Run the server
if __name__ == "__main__":
    app.run(debug=os.environ.get("DASH_DEBUG", "0") == "1", port=int(os.environ.get("PORT", 8050)))