from __future__ import annotations

import json

import dash_bootstrap_components as dbc
import experiment_params
import plotly.express as px
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
                                    style_table={"overflowX": "scroll"},
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
                                                "dispersion_extended",
                                            ],
                                            value="dispersion_extended",
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
                                id="find-spots-advanced-input",
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
                                    "backgroundColor": "rgb(34,34,34)",
                                },
                            )
                        ),
                    ),
                    style={"backgroundColor": "rgb(34,34,34)"},
                ),
            ]
        )
    ),
]

index_tab = [
    dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button("Run", n_clicks=0, id="dials-index"),
                            width=8,
                        ),
                        dbc.Col(
                            html.P(
                                dcc.Link(
                                    dbc.Button("Documentation", color="secondary"),
                                    href="https://dials.github.io/documentation/programs/dials_index.html",
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
                                            id="index-algorithm",
                                            options=[
                                                "fft3d",
                                                "fft1d",
                                                "real_space_grid_search",
                                            ],
                                            value="fft3d",
                                            clearable=False,
                                        ),
                                        html.Div(
                                            id="index-algorithm-placeholder",
                                            style={"display": "none"},
                                        ),
                                    ],
                                    style={"margin-top": "25px"},
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Label("Advanced Options"),
                        html.P(
                            dbc.Input(
                                id="index-advanced-input",
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
                                id="dials-index-log",
                                children=[],
                                style={
                                    "height": "46.5vh",
                                    "maxHeight": "46.5vh",
                                    "overflow": "scroll",
                                    "backgroundColor": "rgb(34,34,34)",
                                },
                            )
                        )
                    ),
                    style={"backgroundColor": "rgb(34,34,34)"},
                ),
            ]
        )
    ),
]

refine_bravais_panel = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Select Candidate Lattice")),
                dbc.ModalBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.ListGroup(
                                        [
                                            dash_table.DataTable(
                                                experiment_params.bravais_lattices_table_values,
                                                experiment_params.bravais_lattices_table_headers,
                                                id="dials-refine-bravais-table",
                                                style_header=experiment_params.style_header[
                                                    0
                                                ],
                                                style_data=experiment_params.style_data[
                                                    0
                                                ],
                                                row_selectable="single",
                                                selected_row_ids=[],
                                                page_size=50,
                                            ),
                                            html.Div(
                                                id="refine-bravais-table-placeholder",
                                                style={"display": "none"},
                                            ),
                                        ],
                                        className="dbc-row-selectable",
                                        style={
                                            "height": "78vh",
                                            "maxHeight": "78vh",
                                            "width": "78vh",
                                            "maxWidth": "78vh",
                                            "overflow": "scroll",
                                        },
                                    ),
                                    width=8,
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            dcc.Graph(id="lattice-rmsd-plot"),
                                            dcc.Graph(id="lattice-metric-fit-plot"),
                                        ],
                                        style={
                                            "display": "inline-block",
                                            "width": "120%",
                                            "height": "10%",
                                            "backgroundColor": "black",
                                        },
                                    ),
                                    width=4,
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.ModalFooter(dbc.Button("Refine", id="dials-refine", n_clicks=0)),
            ],
            id="dials-refine-bravais-panel",
            is_open=False,
            size="xl",
            backdrop="static",
            scrollable=True,
        )
    ]
)

refine_tab = [
    refine_bravais_panel,
    dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button("Run", n_clicks=0, id="dials-refine-bravais"),
                            width=8,
                        ),
                        dbc.Col(
                            html.P(
                                dcc.Link(
                                    dbc.Button("Documentation", color="secondary"),
                                    href="https://dials.github.io/documentation/programs/dials_refine.html",
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
                                        dbc.Label("Outlier Algorithm"),
                                        dcc.Dropdown(
                                            id="refine-outlier-algorithm",
                                            options=[
                                                "auto",
                                                "mcd",
                                                "tukey",
                                                "sauter_poon",
                                            ],
                                            value="auto",
                                            clearable=False,
                                        ),
                                        html.Div(
                                            id="refine-outlier-placeholder",
                                            style={"display": "none"},
                                        ),
                                    ],
                                    style={"margin-top": "25px"},
                                ),
                            ]
                        ),
                        dbc.Label("Advanced Options"),
                        html.P(
                            dbc.Input(
                                id="refine-advanced-input",
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
                                id="dials-refine-log",
                                children=[],
                                style={
                                    "height": "46.5vh",
                                    "maxHeight": "46.5vh",
                                    "overflow": "scroll",
                                    "backgroundColor": "rgb(34,34,34)",
                                },
                            )
                        )
                    ),
                    style={"backgroundColor": "rgb(34,34,34)"},
                ),
            ]
        )
    ),
]


image_viewer_tab = dbc.Card(
    dbc.CardImg(src=None, top=True, id="image-viewer-image"),
    style={
        "height": "85.5vh",
        "maxHeight": "85.5vh",
        "overflow": "scroll",
    },
)

reciprocal_lattice_viewer_tab = dbc.Card(
    dbc.CardImg(
        src=app.get_asset_url("reciprocal_lattice_viewer_before_indexing.png"),
        top=True,
        id="reciprocal-lattice-viewer-image",
    ),
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
                        dbc.Tabs(
                            children=[
                                dbc.Tab(
                                    dbc.Card(
                                        dbc.CardBody(
                                            html.Div(
                                                dbc.ListGroup(
                                                    id="open-files", children=[]
                                                ),
                                                style={
                                                    "height": "83.5vh",
                                                    "maxHeight": "83.5vh",
                                                    "overflow": "scroll",
                                                },
                                            ),
                                        ),
                                    ),
                                    label="Files",
                                    tab_style={"color": "rgb(48,48,48)"},
                                    active_tab_style={"color": "rgb(48,48,48)"},
                                ),
                            ]
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
                                    reciprocal_lattice_viewer_tab,
                                    label="Reciprocal Lattice Viewer",
                                    disabled=True,
                                ),
                                dbc.Tab(
                                    experiment_summary,
                                    label="Experiment",
                                    disabled=True,
                                ),
                            ],
                            id="state-tabs",
                        )
                    ),
                    width=5,
                ),
                dbc.Col(
                    html.Div(
                        [
                            dbc.Tabs(
                                children=[
                                    dbc.Tab(
                                        import_tab.content(),
                                        label="Import",
                                        tab_style={"color": "rgb(48,48,48)"},
                                        active_tab_style={"color": "rgb(48,48,48)"},
                                    ),
                                    dbc.Tab(
                                        find_spots_tab,
                                        label="Find Spots",
                                        disabled=True,
                                    ),
                                    dbc.Tab(index_tab, label="Index", disabled=True),
                                    dbc.Tab(refine_tab, label="Refine", disabled=True),
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
                                disabled=True,
                                outline=False,
                                style={
                                    "position": "absolute",
                                    "left": ".9vw",
                                    "top": "87vh",
                                    "width": "39.1vw",
                                    "height": "2vh",
                                    "backgroundColor": "rgb(34, 34, 34)",
                                    "lineColor": "rgb(34, 34, 34)",
                                },
                                children=[
                                    html.Img(
                                        src=app.get_asset_url("expand_less2.png"),
                                        hidden=True,
                                        style={
                                            "position": "relative",
                                            "left": "-18.5vw",
                                            "top": "-1.25vh",
                                        },
                                        id="reflection-table-arrow",
                                    ),
                                    html.P(
                                        "Reflection Table",
                                        hidden=True,
                                        style={
                                            "position": "relative",
                                            "left": "0vw",
                                            "top": "-3.75vh",
                                        },
                                        id="reflection-table-label",
                                    ),
                                ],
                            ),
                            reflection_table,
                        ],
                        style={"position": "relative"},
                    ),
                    width=5,
                ),
            ]
        ),
    ]
)


@app.callback(
    [
        Output("dials-refine-bravais-panel", "is_open"),
        Output("dials-refine-bravais-table", "data"),
        Output("lattice-rmsd-plot", "figure"),
        Output("lattice-metric-fit-plot", "figure"),
    ],
    [
        Input("dials-refine-bravais", "n_clicks"),
        Input("dials-refine", "n_clicks"),
        Input("dials-refine-bravais-table", "data"),
    ],
    [State("dials-refine-bravais-panel", "is_open")],
)
def show_refine_bravais_settings_panel(n1, n2, bravais_table, is_open):
    def get_plots(bravais_table):
        if bravais_table is not None:
            rmsd = [i["RMSD"] for i in bravais_table]
            mf = [i["Metric Fit"] for i in bravais_table]
            x = [idx + 1 for idx, i in enumerate(bravais_table)]
            rmsd = {"Candidate": x, "RMSD": rmsd}
            mf = {"Candidate": x, "Metric Fit": mf}
            rmsd_fig = px.scatter(rmsd, x="Candidate", y="RMSD", height=340)
            rmsd_fig.update_traces(
                mode="lines+markers", marker_color="white", textfont_color="white"
            )
            rmsd_fig.layout.plot_bgcolor = "rgb(48,48,48)"
            rmsd_fig.layout.paper_bgcolor = "rgb(48,48,48)"
            rmsd_fig.update_xaxes(showgrid=False, color="white")
            rmsd_fig.update_yaxes(showgrid=False, color="white")
            mf_fig = px.scatter(mf, x="Candidate", y="Metric Fit", height=340)
            mf_fig.update_traces(
                mode="lines+markers",
                marker_color="white",
            )
            mf_fig.layout.plot_bgcolor = "rgb(48,48,48)"
            mf_fig.layout.paper_bgcolor = "rgb(48,48,48)"
            mf_fig.update_xaxes(showgrid=False, color="white")
            mf_fig.update_yaxes(showgrid=False, color="white")

        else:
            rmsd = {"Candidate": [], "RMSD": []}
            mf = {"Candidate": [], "Metric Fit": []}
            rmsd_fig = px.scatter(rmsd, x="Candidate", y="RMSD")
            mf_fig = px.scatter(mf, x="Candidate", y="Metric Fit")

        return rmsd_fig, mf_fig

    triggered_id = callback_context.triggered_id
    rmsd_fig, mf_fig = get_plots(None)

    # Table updated
    if triggered_id == "dials-refine-bravais-table":
        return is_open, bravais_table, rmsd_fig, mf_fig

    # Refine_bravais_settings run button pressed
    if triggered_id == "dials-refine-bravais":
        _ = file_manager.run(AlgorithmType.dials_refine_bravais_settings)
        bravais_table = file_manager.get_bravais_lattices_table()
        rmsd_fig, mf_fig = get_plots(bravais_table)
        is_open = True
        return is_open, bravais_table, rmsd_fig, mf_fig

    # Refine run button pressed
    elif triggered_id == "dials-refine":
        is_open = False
        return is_open, bravais_table, rmsd_fig, mf_fig
    else:
        is_open = False
        return is_open, bravais_table, rmsd_fig, mf_fig


@app.callback(
    Output("refine-bravais-table-placeholder", "children"),
    Input("dials-refine-bravais-table", "selected_rows"),
)
def update_refine_selected_files(selected_rows):
    if selected_rows:

        # Update dials.reindex args
        basis = file_manager.get_change_of_basis(f"{selected_rows[0]+1}")
        file_manager.update_selected_file_arg(
            algorithm_type=AlgorithmType.dials_reindex,
            param_name="change_of_basis_op",
            param_value=basis,
        )

        # Update dials.refine input
        refine_expt_filename = f"bravais_setting_{selected_rows[0]+1}.expt"
        refine_refl_filename = "reindexed.refl"
        file_manager.set_selected_input_files(
            selected_files=[refine_expt_filename, refine_refl_filename],
            algorithm_type=AlgorithmType.dials_refine,
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
    Output("refine-outlier-placeholder", "children"),
    Input("refine-outlier-algorithm", "value"),
)
def update_refine_outlier_algorithm(outlier_algorithm):
    if file_manager.selected_file is not None:
        file_manager.update_selected_file_arg(
            AlgorithmType.dials_refine,
            "reflections.outlier.algorithm",
            outlier_algorithm,
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
    Output("index-algorithm-placeholder", "children"),
    Input("index-algorithm", "value"),
)
def update_index_algorithm(index_algorithm):
    if file_manager.selected_file is not None:
        file_manager.update_selected_file_arg(
            AlgorithmType.dials_index, "indexing.method", index_algorithm
        )


@app.callback(
    [Output("image-viewer-image", "src")],
    [
        Input("open-files", "children"),
    ],
)
def update_placeholder_images(open_files):
    if file_manager.can_run(AlgorithmType.dials_index):
        return [app.get_asset_url("image_viewer_after_find_spots.png")]
    if file_manager.can_run(AlgorithmType.dials_find_spots):
        return [app.get_asset_url("image_viewer_before_find_spots.png")]
    return [[]]


@app.callback(
    [
        Output("open-reflection-table", "disabled"),
        Output("reflection-table-label", "hidden"),
        Output("reflection-table-arrow", "hidden"),
        Output("open-reflection-table", "style"),
    ],
    [
        Input("open-files", "children"),
        Input("open-reflection-table", "style"),
    ],
)
def update_reflection_table_button(open_files, button_style):
    if file_manager.can_run(AlgorithmType.dials_index):
        button_style["backgroundColor"] = "rgb(55, 90, 127)"
        button_style["lineColor"] = "rgb(55, 90, 127)"
        return [False, False, False, button_style]
    button_style["backgroundColor"] = "rgb(34, 34, 34)"
    button_style["lineColor"] = "rgb(34, 34, 34)"
    return [True, True, True, button_style]


@app.callback(
    [
        Output("algorithm-tabs", "children"),
        Output("state-tabs", "children"),
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
        Output("dials-index-log", "children"),
        Output("dials-refine-log", "children"),
    ],
    [
        Input({"type": "open-file", "index": ALL}, "n_clicks"),
        Input("open-files", "children"),
        Input("algorithm-tabs", "children"),
        Input("state-tabs", "children"),
        Input("dials-import", "filename"),
        Input("dials-import", "contents"),
        Input("dials-find-spots", "n_clicks"),
        Input("dials-index", "n_clicks"),
        Input("dials-refine", "n_clicks"),
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
    state_tabs,
    import_filename,
    import_content,
    find_spots_n_clicks,
    index_n_clicks,
    refine_n_clicks,
    *experiment_params,
):

    triggered_id = callback_context.triggered_id
    print(f"Triggered id : {triggered_id}")
    logs = file_manager.get_logs()
    reflection_table = file_manager.get_reflection_table()
    min_image_range = 1
    max_image_range = 1

    ## Nothing triggered
    if triggered_id is None:
        return (
            algorithm_tabs,
            state_tabs,
            open_files,
            min_image_range,
            max_image_range,
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
        state_tabs = display_manager.update_state_tabs(
            state_tabs, file_manager.selected_file
        )
        experiment_params = display_manager.get_experiment_params(
            file_manager.selected_file
        )
        min_image_range, max_image_range = file_manager.get_selected_file_image_range()
        return (
            algorithm_tabs,
            state_tabs,
            open_files,
            min_image_range,
            max_image_range,
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
        state_tabs = display_manager.update_state_tabs(
            state_tabs, file_manager.selected_file
        )
        min_image_range, max_image_range = file_manager.get_selected_file_image_range()
        reflection_table = file_manager.get_reflection_table()
        experiment_params = display_manager.get_experiment_params(
            file_manager.selected_file
        )
        return (
            algorithm_tabs,
            state_tabs,
            open_files,
            min_image_range,
            max_image_range,
            reflection_table,
            *experiment_params,
            *logs,
        )

    ## Running index
    if triggered_id == "dials-index":
        logs[2] = file_manager.run(AlgorithmType.dials_index)
        algorithm_tabs = display_manager.update_algorithm_tabs(
            algorithm_tabs, file_manager.selected_file
        )
        state_tabs = display_manager.update_state_tabs(
            state_tabs, file_manager.selected_file
        )
        min_image_range, max_image_range = file_manager.get_selected_file_image_range()
        reflection_table = file_manager.get_reflection_table()
        experiment_params = display_manager.get_experiment_params(
            file_manager.selected_file
        )
        return (
            algorithm_tabs,
            state_tabs,
            open_files,
            min_image_range,
            max_image_range,
            reflection_table,
            *experiment_params,
            *logs,
        )

    ## Running refine
    if triggered_id == "dials-refine":
        if file_manager.has_selected_input_files(AlgorithmType.dials_refine):
            logs[3] = file_manager.run(AlgorithmType.dials_reindex)
        logs[3] = file_manager.run(AlgorithmType.dials_refine)
        algorithm_tabs = display_manager.update_algorithm_tabs(
            algorithm_tabs, file_manager.selected_file
        )
        state_tabs = display_manager.update_state_tabs(
            state_tabs, file_manager.selected_file
        )
        experiment_params = display_manager.get_experiment_params(
            file_manager.selected_file
        )
        min_image_range, max_image_range = file_manager.get_selected_file_image_range()
        return (
            algorithm_tabs,
            state_tabs,
            open_files,
            min_image_range,
            max_image_range,
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
    state_tabs = display_manager.update_state_tabs(
        state_tabs, file_manager.selected_file
    )
    experiment_params = display_manager.get_experiment_params(
        file_manager.selected_file
    )
    min_image, max_image = file_manager.get_selected_file_image_range()
    reflection_table = file_manager.get_reflection_table()
    return (
        algorithm_tabs,
        state_tabs,
        open_files,
        min_image,
        max_image,
        reflection_table,
        *experiment_params,
        *logs,
    )


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
