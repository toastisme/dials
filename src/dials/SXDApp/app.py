from __future__ import annotations

import json

import dash_bootstrap_components as dbc
import experiment_params
from algorithm_types import AlgorithmType
from app_import_tab import ImportTab
from app_reflection_table import reflection_table
from dash import (
    ALL,
    Dash,
    Input,
    Output,
    State,
    callback_context,
    dash_table,
    dcc,
    html,
)
from display_manager import DisplayManager
from open_file_manager import OpenFileManager

app = Dash(external_stylesheets=[dbc.themes.DARKLY])

file_manager = OpenFileManager()
display_manager = DisplayManager()

## Algorithm tabs

import_tab = ImportTab()

generic_tab = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Button("Run", n_clicks=0), width=8),
                    dbc.Col(
                        dcc.Link(
                            dbc.Button("Documentation", color="secondary"),
                            href="https://dials.github.io/documentation/programs/dials_find_spots.html",
                            target="_blank",
                        ),
                        width=1,
                        align="end",
                    ),
                ]
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.DropdownMenu(
                            label="Algorithm",
                            children=[
                                dbc.DropdownMenuItem("Item 1"),
                                dbc.DropdownMenuItem("Item 2"),
                                dbc.DropdownMenuItem("Item 3"),
                            ],
                        ),
                        dbc.Accordion(
                            [dbc.AccordionItem([], title="Advanced")],
                            start_collapsed=True,
                        ),
                    ]
                )
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H6("Output"),
                        dbc.Card(html.P()),
                    ]
                )
            ),
        ]
    ),
)

experiment_summary = dbc.Card(
    [
        dbc.Row(
            dbc.Card(
                [
                    dbc.CardHeader("Detector"),
                    dbc.CardBody(
                        dbc.ListGroup(
                            [
                                dash_table.DataTable(
                                    experiment_params.detector_values,
                                    experiment_params.detector_headers,
                                    id="detector-params",
                                    style_header=experiment_params.style_header[0],
                                    style_data=experiment_params.style_data[0],
                                ),
                            ],
                        )
                    ),
                ]
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Beam"),
                            dbc.CardBody(
                                dbc.ListGroup(
                                    [
                                        dash_table.DataTable(
                                            experiment_params.beam_values,
                                            experiment_params.beam_headers,
                                            id="beam-params",
                                            style_header=experiment_params.style_header[
                                                0
                                            ],
                                            style_data=experiment_params.style_data[0],
                                        ),
                                    ],
                                )
                            ),
                        ]
                    )
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Sequence"),
                            dbc.CardBody(
                                dbc.ListGroup(
                                    [
                                        dash_table.DataTable(
                                            experiment_params.sequence_values,
                                            experiment_params.sequence_headers,
                                            id="sequence-params",
                                            style_header=experiment_params.style_header[
                                                0
                                            ],
                                            style_data=experiment_params.style_data[0],
                                        ),
                                    ],
                                )
                            ),
                        ]
                    ),
                ),
            ]
        ),
        dbc.Card(
            [
                dbc.CardHeader("Goniometer"),
                dbc.CardBody(
                    dbc.ListGroup(
                        [
                            dash_table.DataTable(
                                experiment_params.goniometer_values,
                                experiment_params.goniometer_headers,
                                id="goniometer-params",
                                style_header=experiment_params.style_header[0],
                                style_data=experiment_params.style_data[0],
                            ),
                        ]
                    )
                ),
            ]
        ),
        dbc.Card(
            [
                dbc.CardHeader("Crystal"),
                dbc.CardBody(
                    dbc.ListGroup(
                        [
                            dash_table.DataTable(
                                experiment_params.crystal_values,
                                experiment_params.crystal_headers,
                                id="crystal-params",
                                style_header=experiment_params.style_header[0],
                                style_data=experiment_params.style_data[0],
                            ),
                        ]
                    )
                ),
            ]
        ),
    ],
    style={
        "height": "85.5vh",
        "maxHeight": "85.5vh",
        "overflow": "scroll",
    },
    body=True,
    id="experiment-summary",
)

find_spots_tab = [
    dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button("Run", n_clicks=0, id="dials-find-spots"),
                            width=8,
                        ),
                        dbc.Col(
                            html.P(
                                dcc.Link(
                                    dbc.Button("Documentation", color="secondary"),
                                    href="https://dials.github.io/documentation/programs/dials_find_spots.html",
                                    target="_blank",
                                ),
                                style={"margin-left": "65px"},
                            ),
                            width=1,
                            align="end",
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.P(
                                    [
                                        dbc.Label("Algorithm"),
                                        dcc.Dropdown(
                                            id="find-spots-threshold-algorithm",
                                            options=[
                                                "dispersion",
                                                "dispersion extended",
                                            ],
                                            value="dispersion extended",
                                            clearable=False,
                                        ),
                                        html.Div(
                                            id="find-spots-threshold-algorithm-placeholder",
                                            style={"display": "none"},
                                        ),
                                    ],
                                    style={"margin-top": "25px"},
                                ),
                            ]
                        ),
                        dbc.Col(
                            html.P(
                                [
                                    dbc.Label("Image Range"),
                                    dcc.RangeSlider(
                                        1,
                                        20,
                                        1,
                                        id="image-range",
                                        marks=None,
                                        allowCross=False,
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                        },
                                    ),
                                    html.Div(
                                        id="image-range-placeholder",
                                        style={"display": "none"},
                                    ),
                                ],
                                style={"margin-top": "25px", "color": "white"},
                            ),
                        ),
                        dbc.Label("Advanced Options"),
                        html.P(
                            dbc.Input(
                                id="input",
                                placeholder="See Documentation for full list of options",
                                type="text",
                            ),
                        ),
                    ]
                ),
            ],
        )
    ),
    dbc.Card(
        dbc.CardBody(
            [
                html.H6("Output"),
                dbc.Card(
                    dbc.CardBody(
                        dbc.Spinner(
                            html.Div(
                                id="dials-find-spots-log",
                                children=[],
                                style={
                                    "height": "46.5vh",
                                    "maxHeight": "46.5vh",
                                    "overflow": "scroll",
                                },
                            )
                        )
                    )
                ),
            ]
        )
    ),
]
image_viewer_tab = dbc.Card(
    dbc.CardImg(src=app.get_asset_url("image_viewer_with_line_plot.png"), top=True),
    style={
        "height": "85.5vh",
        "maxHeight": "85.5vh",
        "overflow": "scroll",
    },
)


app.layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H1(),
                ),
                dbc.Col(
                    html.A(
                        href="https://dials.github.io/",
                        children=[
                            html.Img(
                                alt="DIALS documentation",
                                src=app.get_asset_url("dials_logo.png"),
                                style={"height": "90%", "width": "90%"},
                            ),
                        ],
                        target="_blank",
                    ),
                    width=1,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("Files"),
                                    html.Div(
                                        dbc.ListGroup(id="open-files", children=[]),
                                        style={
                                            "height": "83.5vh",
                                            "maxHeight": "83.5vh",
                                            "overflow": "scroll",
                                        },
                                    ),
                                ]
                            )
                        )
                    ),
                    width=2,
                ),
                dbc.Col(
                    html.Div(
                        dbc.Tabs(
                            [
                                dbc.Tab(image_viewer_tab, label="Image Viewer"),
                                dbc.Tab(
                                    generic_tab,
                                    label="Reciprocal Lattice Viewer",
                                    disabled=True,
                                ),
                                dbc.Tab(experiment_summary, label="Experiment"),
                            ],
                        )
                    )
                ),
                dbc.Col(
                    html.Div(
                        [
                            dbc.Tabs(
                                children=[
                                    dbc.Tab(
                                        import_tab.content(),
                                        label="Import",
                                        tab_style={"color": "#9E9E9E"},
                                        active_tab_style={"color": "#9E9E9E"},
                                    ),
                                    dbc.Tab(
                                        find_spots_tab,
                                        label="Find Spots",
                                        disabled=True,
                                    ),
                                    dbc.Tab(generic_tab, label="Index", disabled=True),
                                    dbc.Tab(generic_tab, label="Refine", disabled=True),
                                    dbc.Tab(
                                        generic_tab, label="Integrate", disabled=True
                                    ),
                                    dbc.Tab(generic_tab, label="Scale", disabled=True),
                                ],
                                id="algorithm-tabs",
                            ),
                            dbc.Button(
                                id="open-reflection-table",
                                n_clicks=0,
                                style={
                                    "position": "absolute",
                                    "left": "11vw",
                                    "top": "88vh",
                                    "width": "20vw",
                                    "height": "2vh",
                                    "backgroundColor": "rgb(48,48,48)",
                                    "line-color": "rgb(48,48,48)",
                                },
                                children=[
                                    html.Img(
                                        src=app.get_asset_url("expand_less2.png"),
                                        style={
                                            "position": "relative",
                                            "left": "-9vw",
                                            "top": "-1.25vh",
                                        },
                                    ),
                                    html.P(
                                        "Reflection Table",
                                        style={
                                            "position": "relative",
                                            "left": "0vw",
                                            "top": "-3.75vh",
                                        },
                                    ),
                                ],
                            ),
                            reflection_table,
                        ],
                        style={"position": "relative"},
                    )
                ),
            ]
        ),
    ]
)


@app.callback(
    Output("reflection-table-window", "is_open"),
    Input("open-reflection-table", "n_clicks"),
    [State("reflection-table-window", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open


@app.callback(
    Output("find-spots-threshold-algorithm-placeholder", "children"),
    Input("find-spots-threshold-algorithm", "value"),
)
def update_find_spots_threshold_algorithm(threshold_algorithm):
    if file_manager.selected_file is not None:
        file_manager.update_selected_file_arg(
            AlgorithmType.dials_find_spots, "threshold.algorithm", threshold_algorithm
        )


@app.callback(
    Output("image-range-placeholder", "children"),
    Input("image-range", "value"),
)
def update_image_range(image_range):
    if file_manager.selected_file is None or image_range is None:
        return
    val = f"{image_range[0]},{image_range[1]}"
    file_manager.update_selected_file_arg(
        AlgorithmType.dials_find_spots, "scan_range", val
    )


@app.callback(
    [
        Output("algorithm-tabs", "children"),
        Output("open-files", "children"),
        Output("image-range", "min"),
        Output("image-range", "max"),
        Output("reflection-table", "data"),
        Output("beam-params", "data"),
        Output("detector-params", "data"),
        Output("sequence-params", "data"),
        Output("goniometer-params", "data"),
        Output("crystal-params", "data"),
        Output("dials-import-log", "children"),
        Output("dials-find-spots-log", "children"),
    ],
    [
        Input({"type": "open-file", "index": ALL}, "n_clicks"),
        Input("open-files", "children"),
        Input("algorithm-tabs", "children"),
        Input("dials-import", "filename"),
        Input("dials-import", "contents"),
        Input("dials-find-spots", "n_clicks"),
        Input("beam-params", "data"),
        Input("detector-params", "data"),
        Input("sequence-params", "data"),
        Input("goniometer-params", "data"),
        Input("crystal-params", "data"),
    ],
)
def event_handler(
    open_files_clicks_list,
    open_files,
    algorithm_tabs,
    import_filename,
    import_content,
    find_spots_n_clicks,
    *experiment_params,
):

    triggered_id = callback_context.triggered_id
    print(f"Triggered id : {triggered_id}")
    logs = file_manager.get_logs()
    reflection_table = file_manager.get_reflection_table()

    ## Nothing triggered
    if triggered_id is None:
        return (
            algorithm_tabs,
            open_files,
            1,
            20,
            reflection_table,
            *experiment_params,
            *logs,
        )

    ## Loading new file
    if triggered_id == "dials-import":
        file_manager.add_active_file(import_filename, import_content)
        logs[0] = file_manager.run(AlgorithmType.dials_import)

        open_files = display_manager.add_file(
            open_files,
            filename=file_manager.get_selected_filename(),
        )
        algorithm_tabs = display_manager.update_algorithm_tabs(
            algorithm_tabs, file_manager.selected_file
        )
        experiment_params = display_manager.get_experiment_params(
            file_manager.selected_file
        )
        min_image, max_image = file_manager.get_selected_file_image_range()
        return (
            algorithm_tabs,
            open_files,
            min_image,
            max_image,
            reflection_table,
            *experiment_params,
            *logs,
        )

    ## Running find spots
    if triggered_id == "dials-find-spots":
        logs[1] = file_manager.run(AlgorithmType.dials_find_spots)
        algorithm_tabs = display_manager.update_algorithm_tabs(
            algorithm_tabs, file_manager.selected_file
        )
        min_image, max_image = file_manager.get_selected_file_image_range()
        reflection_table = file_manager.get_reflection_table()
        return (
            algorithm_tabs,
            open_files,
            min_image,
            max_image,
            reflection_table,
            *experiment_params,
            *logs,
        )

    ## Clicked on file
    clicked_id = callback_context.triggered[0]["prop_id"]
    clicked_id = int(json.loads(clicked_id.split(".")[0])["index"])
    open_files = display_manager.select_file(open_files, clicked_id)

    # Update display
    file_manager.update_selected_file(clicked_id)
    logs = file_manager.get_logs()
    algorithm_tabs = display_manager.update_algorithm_tabs(
        algorithm_tabs, file_manager.selected_file
    )
    experiment_params = display_manager.get_experiment_params(
        file_manager.selected_file
    )
    min_image, max_image = file_manager.get_selected_file_image_range()
    reflection_table = file_manager.get_reflection_table()
    return (
        algorithm_tabs,
        open_files,
        min_image,
        max_image,
        reflection_table,
        *experiment_params,
        *logs,
    )


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
