import os
import pathlib

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import plotly.graph_objs as go
import dash_daq as daq

import pandas as pd
from ml_model import get_act_true_act_pred, get_atac1_true_atac1_pred, get_atac2_true_atac2_pred, get_atac3_true_atac3_pred


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server
app.config["suppress_callback_exceptions"] = True

APP_PATH = str(pathlib.Path(__file__).parent.resolve())

df_ts = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "spc_data.csv")))
params = list(df_ts)
max_length = len(df_ts)


current_activity, pred_activity = get_act_true_act_pred(None)
current_atac1, pred_atac1 = get_atac1_true_atac1_pred(None)

current_activity, pred_activity = "{:.0f}".format(current_activity), "{:.0f}".format(pred_activity)
current_atac1, pred_atac1 = "{:.0f}".format(current_atac1), "{:.0f}".format(pred_atac1)


current_atac2, pred_atac2 = get_atac2_true_atac2_pred(None)
current_atac2, pred_atac2 = "{:.0f}".format(current_atac2), "{:.0f}".format(pred_atac2)


current_atac3, pred_atac3 = get_atac3_true_atac3_pred(None)
current_atac3, pred_atac3 = "{:.0f}".format(current_atac3), "{:.0f}".format(pred_atac3)


suffix_row = "_row"
suffix_button_id = "_button"
suffix_sparkline_graph = "_sparkline_graph"
suffix_count = "_count"
suffix_ooc_n = "_OOC_number"
suffix_ooc_g = "_OOC_graph"
suffix_indicator = "_indicator"

df_gl = pd.read_csv('data/File_target.csv')
df_gl['date'] = pd.to_datetime(df_gl['date'].values)
df_gl.index = df_gl['date']
df_table = df_gl.tail(10).iloc[:, :10].round(3)


df_main_timeseries = pd.read_csv('data/prev_score.csv')
df_main_timeseries['date'] = pd.to_datetime(df_main_timeseries['date'])


x_activity_true = pd.to_datetime(df_main_timeseries['date'].values)
y_activity_true = df_main_timeseries['activity_true'].values

x_activity_pred = pd.to_datetime(df_main_timeseries['date'].values)
y_activity_pred = df_main_timeseries['activity_pred'].values

main_figure1 = html.Div(
    className='six columns',
    children=dcc.Graph(
        id='main_figure1',
        figure={
            'data': [
                {'x': x_activity_true, 'y': y_activity_true, 'type': 'scatter', 'name': 'activity_true'},
                {'x': x_activity_pred, 'y': y_activity_pred, 'type': 'scatter', 'name': 'activity_pred'},
            ],
            'layout': {
                'title': 'Activity',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)',
            }
        }
    ),
    style={"height": "25%", "width": "100%"}
)


def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H5("МАК"),
                    html.H6("Моделирование активности катализатора"),
                ],
            ),
            html.Div(
                id="banner-logo",
                children=[
                    html.Img(id="logo", src=app.get_asset_url("dsbear_ams.png"),
                             # width="40px", height="70px"
                             ),
                ],
            ),
        ],
    )


def build_tabs():
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="tab2",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="Control-chart-tab",
                        label="Оперативная статистика",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Specs-tab",
                        label="Технологическое моделирование",
                        value="tab1",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                ],
            )
        ],
    )


def init_df():
    ret = {}

    for col in list(df_ts[1:3]):
        data = df_ts[col]
        stats = data.describe()

        std = stats["std"].tolist()
        ucl = (stats["mean"] + 3 * stats["std"]).tolist()
        lcl = (stats["mean"] - 3 * stats["std"]).tolist()
        usl = (stats["mean"] + stats["std"]).tolist()
        lsl = (stats["mean"] - stats["std"]).tolist()

        ret.update(
            {
                col: {
                    "count": stats["count"].tolist(),
                    "data": data,
                    "mean": stats["mean"].tolist(),
                    "std": std,
                    "ucl": round(ucl, 3),
                    "lcl": round(lcl, 3),
                    "usl": round(usl, 3),
                    "lsl": round(lsl, 3),
                    "min": stats["min"].tolist(),
                    "max": stats["max"].tolist(),
                    "ooc": populate_ooc(data, ucl, lcl),
                }
            }
        )

    return ret


def populate_ooc(data, ucl, lcl):
    ooc_count = 0
    ret = []
    for i in range(len(data)):
        if data[i] >= ucl or data[i] <= lcl:
            ooc_count += 1
            ret.append(ooc_count / (i + 1))
        else:
            ret.append(ooc_count / (i + 1))
    return ret


def init_value_setter_store():
    # Initialize store data
    state_dict = init_df()
    return state_dict


def build_tab_1():
    return main_figure1


ud_usl_input = daq.NumericInput(
    id="ud_usl_input", className="setting-input", size=200, max=9999999
)
ud_lsl_input = daq.NumericInput(
    id="ud_lsl_input", className="setting-input", size=200, max=9999999
)
ud_ucl_input = daq.NumericInput(
    id="ud_ucl_input", className="setting-input", size=200, max=9999999
)
ud_lcl_input = daq.NumericInput(
    id="ud_lcl_input", className="setting-input", size=200, max=9999999
)


def build_value_setter_line(line_num, label, value, col3):
    return html.Div(
        id=line_num,
        children=[
            html.Label(label, className="four columns"),
            html.Label(value, className="four columns"),
            html.Div(col3, className="four columns"),
        ],
        className="row",
    )


def generate_modal():
    return html.Div(
        id="markdown",
        className="modal",
        children=(
            html.Div(
                id="markdown-container",
                className="markdown-container",
                children=[
                    html.Div(
                        className="close-container",
                        children=html.Button(
                            "Close",
                            id="markdown_close",
                            n_clicks=0,
                            className="closeButton",
                        ),
                    ),
                    html.Div(
                        className="markdown-text",
                        children=dcc.Markdown(
                            children=(
                                """
                        ###### What is this mock app about?

                        This is a dashboard for monitoring real-time process quality along manufacture production line. 

                        ###### What does this app shows

                        Click on buttons in `Parameter` column to visualize details of measurement trendlines on the bottom panel.

                        The sparkline on top panel and control chart on bottom panel show Shewhart process monitor using mock data. 
                        The trend is updated every other second to simulate real-time measurements. Data falling outside of six-sigma control limit are signals indicating 'Out of Control(OOC)', and will 
                        trigger alerts instantly for a detailed checkup. 
                        
                        Operators may stop measurement by clicking on `Stop` button, and edit specification parameters by clicking specification tab.

                    """
                            )
                        ),
                    ),
                ],
            )
        ),
    )


def build_quick_stats_panel():
    return html.Div(
        id="quick-stats",
        # id="card-1",
        className="four columns",
        children=[
            generate_section_banner("Текущие показатели"),
            html.P("Aктивность"),
            daq.LEDDisplay(
                id="operator-led",
                value=current_activity,
                color="#92e0d3",
                backgroundColor="#1e2130",
                size=50,
            ),
            html.P("Атактика 1"),
            daq.LEDDisplay(
                id="atactica-led",
                value=current_atac1,
                color="#92e0d3",
                backgroundColor="#1e2130",
                size=50,
            ),
            html.P("Атактика 2"),
            daq.LEDDisplay(
                id="atactica-led",
                value=current_atac2,
                color="#92e0d3",
                backgroundColor="#1e2130",
                size=50,
            ),
            html.P("Атактика 3"),
            daq.LEDDisplay(
                id="atactica-led",
                value=current_atac3,
                color="#92e0d3",
                backgroundColor="#1e2130",
                size=50,
            ),
            html.Div(
                id="utility-card",
                children=[daq.StopButton(id="stop-button", size=160, n_clicks=0)],
            ),
        ],
    )


def generate_section_banner(title):
    return html.Div(className="section-banner", children=title)


def generate_section_future():
    return html.Div(
        id="future-stats",
        className="row",
        children=[
            html.Div(
                id="future_card-1",
                children=[
                    html.P("Активность"),
                    daq.LEDDisplay(
                        id="future_activity-led",
                        value=pred_activity,
                        color="#92e0d3",
                        backgroundColor="#1e2130",
                        size=50,
                    ),
                    html.P("Атактика 1"),
                    daq.LEDDisplay(
                        id="future_atactic-led",
                        value=pred_atac1,
                        color="#92e0d3",
                        backgroundColor="#1e2130",
                        size=50,
                    ),
                    html.P("Атактика 2"),
                    daq.LEDDisplay(
                        id="future_atactic-led",
                        value=pred_atac2,
                        color="#92e0d3",
                        backgroundColor="#1e2130",
                        size=50,
                    ),
                    html.P("Атактика 3"),
                    daq.LEDDisplay(
                        id="future_atactic-led",
                        value=pred_atac3,
                        color="#92e0d3",
                        backgroundColor="#1e2130",
                        size=50,
                    ),
                ],
            ),
            html.Div(
                id="future_utility-card",
                children=[daq.StopButton(id="stop-button", size=160, n_clicks=0)],
            ),
        ],
    )


def generate_children_metrics_rows(stopped_interval):
    return [
        generate_metric_list_header(),
        html.Div(
            id="metric-rows",
            children=[
                generate_metric_row_helper(stopped_interval, 1),
                generate_metric_row_helper(stopped_interval, 2),
                generate_metric_row_helper(stopped_interval, 3),
                generate_metric_row_helper(stopped_interval, 4),
                generate_metric_row_helper(stopped_interval, 5),
                generate_metric_row_helper(stopped_interval, 6),
                generate_metric_row_helper(stopped_interval, 7),
            ],
        ),
    ]


def build_top_panel(stopped_interval):
    return html.Div(
        id="top-section-container",
        className="row",
        children=[
            # Metrics summary
            html.Div(
                id="metric-summary-session",
                className="eight columns",
                children=[
                    generate_section_banner("Изменения показателей"),
                    html.Div(
                        id="metric-div",
                        children=generate_children_metrics_rows(stopped_interval),
                    ),
                ],
            ),
            html.Div(
                id="ooc-piechart-outer",
                className="four columns",
                children=[
                    generate_section_banner("Прогноз 6 часов"),
                    generate_section_future()
                    # generate_piechart(),
                ],
            ),
        ],
    )


def generate_piechart():
    return dcc.Graph(
        id="piechart",
        figure={
            "data": [
                {
                    "labels": [],
                    "values": [],
                    "type": "pie",
                    "marker": {"line": {"color": "white", "width": 1}},
                    "hoverinfo": "label",
                    "textinfo": "label",
                }
            ],
            "layout": {
                "margin": dict(l=20, r=20, t=20, b=20),
                "showlegend": True,
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
                "font": {"color": "white"},
                "autosize": True,
            },
        },
    )


# Build header
def generate_metric_list_header():
    return generate_metric_row(
        "metric_header",
        {"height": "3rem", "margin": "1rem 0", "textAlign": "center"},
        {"id": "m_header_1", "children": html.Div("Параметр")},
        {"id": "m_header_2", "children": html.Div("Изменение")},
        {"id": "m_header_3", "children": html.Div("%")},
        # {"id": "m_header_5", "children": html.Div("%OOC")},
        # {"id": "m_header_6", "children": "Pass/Fail"},
    )


df_values = df_main_timeseries.tail(50)


col_values_gl = {
        'diameter': df_gl['activity'].values[:100], # 'Активность',
        'etch1': df_gl['act1'].values[:100],  # 'Атактика 1',
        'film-thickness': df_gl['act1'].values[:100],  # 'Атактика 2',
        'etch2': df_gl['f2'].values[:100],  # 'Пропилен',
        'line-width': df_gl['f3'].values[:100],  # 'Водород',
        'overlay': df_gl['f4'].values[:100],  # 'Катализатор',
        'volume': df_gl['f5'].values[:100],  #'Донор'
}


def generate_metric_row_helper(stopped_interval, index):
    col_names = {
        'diameter': 'Активность',
        'etch1': 'Атактика 1',
        'film-thickness': 'Атактика 2',
        'etch2': 'Пропилен',
        'line - width': 'Водород',
        'overlay': 'Катализатор',
        'volume': 'Донор'
    }
    item = params[index]
    div_id = item + suffix_row
    button_id = item + suffix_button_id
    sparkline_graph_id = item + suffix_sparkline_graph
    count_id = item + suffix_count
    ooc_percentage_id = item + suffix_ooc_n
    ooc_graph_id = item + suffix_ooc_g
    indicator_id = item + suffix_indicator
    state_dict = init_df()

    cur_x = df_gl['date'].values[:100]
    cur_y = col_values_gl[item.lower()][:100]

    return generate_metric_row(
        div_id,
        None,
        {
            "id": item,
            "className": "metric-row-button-text",
            "children": html.Button(
                id=button_id,
                className="metric-row-button",
                children=col_names.get(item.lower(), 'nan'),
                title="Click to visualize live SPC chart",
                n_clicks=0,
            ),
        },
        {
            "id": item + "_sparkline",
            "children": dcc.Graph(
                id=sparkline_graph_id,
                style={"width": "100%", "height": "95%"},
                config={
                    "staticPlot": False,
                    "editable": False,
                    "displayModeBar": False,
                },
                figure=go.Figure(
                    {
                        "data": [
                            {
                                "x": cur_x,
                                "y": cur_y,
                                # "x": state_dict["Batch"]["data"].tolist()[
                                #      :stopped_interval
                                #      ],
                                # "y": state_dict[item]["data"][:stopped_interval],

                                "mode": "lines+markers",
                                "name": item,
                                "line": {"color": "#f4d44d"},
                            }
                        ],
                        "layout": {
                            "uirevision": True,
                            "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                            "xaxis": dict(
                                showline=False,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=True,
                            ),
                            "yaxis": dict(
                                showline=False,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=True,
                            ),
                            "paper_bgcolor": "rgba(0,0,0,0)",
                            "plot_bgcolor": "rgba(0,0,0,0)",
                        },
                    }
                ),
            ),
        },
        {"id": ooc_percentage_id, "children": "0.00%"},
    )


def generate_metric_row(id, style, col1, col2, col3,
                        ):
    if style is None:
        style = {"height": "8rem", "width": "100%"}

    return html.Div(
        id=id,
        className="row metric-row",
        style=style,
        children=[
            html.Div(
                id=col1["id"],
                className="one column",
                style={"margin-right": "2.5rem", "minWidth": "50px"},
                children=col1["children"],
            ),
            html.Div(
                id=col2["id"],
                style={"height": "100%"},
                className="four columns",
                children=col2["children"],
            ),
            html.Div(
                id=col3["id"],
                style={"height": "100%"},
                className="four columns",
                children=col3["children"],
            ),
        ],
    )


def build_chart_panel():
    return html.Div(
        id="control-chart-container",
        className="twelve columns",
        children=[
            generate_section_banner("Оптимизация параметров"),
            generate_parameters_optmization()
        ],
    )


def generate_parameters_optmization():
    # parameters optimization from predict model

    local_data = [[0.00124305, 0.00135405, 0.00135714, 0.00210512],
                  [0.00096221, 0.00248905, 0.00191732, 0.00176184],
                  [0.00108628, 0.00135405, 0.00142174, 0.00810175]]

    columns = ['Пропилен', 'Водород', 'Катализатор', 'Донор']
    df_tmp = pd.DataFrame(local_data, columns=columns)
    dash_df = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df_tmp.columns],
        data=df_tmp.to_dict('records'),
        editable=True,
        active_cell={"row": 0, "column": 0},
        selected_cells=[{"row": 0, "column": 0}],

        style_header={'backgroundColor': 'rgb(30, 30, 30)'},
        style_cell={
            'backgroundColor': 'rgba(0,0,0,0)',
            'color': 'white',
            'hover': 'blue',
            'width': '50%'
        },
    )
    return html.Div(
        id="data-table",
        className="eight columns",
        children=[
            # dash_df_prev,
            dash_df
        ],
    )


app.title = 'fast_tactic'

app.layout = html.Div(
    id="big-app-container",
    children=[
        build_banner(),
        dcc.Interval(
            id="interval-component",
            interval=2 * 1000,  # in milliseconds
            n_intervals=50,  # start at batch 50
            disabled=True,
        ),
        html.Div(
            id="app-container",
            children=[
                build_tabs(),
                # Main app
                html.Div(id="app-content"),
            ],
        ),
        dcc.Store(id="value-setter-store", data=init_value_setter_store()),
        dcc.Store(id="n-interval-stage", data=50),
        generate_modal(),
    ],
)


@app.callback(
    [Output("app-content", "children"), Output("interval-component", "n_intervals")],
    [Input("app-tabs", "value")],
    [State("n-interval-stage", "data")],
)
def render_tab_content(tab_switch, stopped_interval):
    if tab_switch == "tab1":
        return build_tab_1(), stopped_interval
    return (
        html.Div(
            id="status-container",
            children=[
                build_quick_stats_panel(),
                html.Div(
                    id="graphs-container",
                    children=[
                        build_top_panel(stopped_interval),
                        build_chart_panel()],
                ),
            ],
        ),
        stopped_interval,
    )


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
