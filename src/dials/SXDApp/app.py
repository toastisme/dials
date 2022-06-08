from __future__ import annotations

import json

import dash_bootstrap_components as dbc
from algorithm_types import AlgorithmType
from app_import_tab import ImportTab
from dash import ALL, Dash, Input, Output, callback_context, dash_table, dcc, html
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

detector_table_header = [
    html.Thead(
        html.Tr(
            [
                html.Th("Panel Name"),
                html.Th("Origin (mm)"),
                html.Th("Fast Axis"),
                html.Th("Slow Axis"),
                html.Th("Pixels"),
                html.Th("Pixel Size (mm)"),
            ]
        )
    )
]

detector_table_row1 = html.Tr(
    [
        html.Td("1"),
        html.Td("(0, 0, 0)"),
        html.Td("(0, 0, 0)"),
        html.Td("(0, 0, 0)"),
        html.Td("(64, 64)"),
        html.Td("(3, 3)"),
    ]
)

detector_table_body = [html.Tbody([detector_table_row1])]

beam_table_header = [
    html.Thead(
        html.Tr(
            [
                html.Th("Sample to Source Direction"),
                html.Th("Sample to Moderator Distance (m)"),
                html.Th("Wavelength Range (A)"),
            ]
        )
    )
]

beam_table_row1 = html.Tr([html.Td("(0, 0, -1)"), html.Td("8.3"), html.Td("(0.2 - 8)")])

beam_table_body = [html.Tbody([beam_table_row1])]

scan_table_header = [
    html.Thead(
        html.Tr(
            [
                html.Th("Image Range"),
                html.Th("Sample to Moderator Distance (m)"),
                html.Th("Wavelength Range (A)"),
                html.Th("Time of Flight Channels"),
            ]
        )
    )
]

scan_table_row1 = html.Tr(
    [
        html.Td("(0, 0, -1)"),
        html.Td("8.3"),
        html.Td("(0.2 - 8)"),
        html.Td("1821"),
    ]
)

scan_table_body = [html.Tbody([scan_table_row1])]

goniometer_table_header = [
    html.Thead(
        html.Tr(
            [
                html.Th("Starting Orientation"),
            ]
        )
    )
]

goniometer_table_row1 = html.Tr(
    [
        html.Td("(0, 0, 0)"),
    ]
)

goniometer_table_body = [html.Tbody([goniometer_table_row1])]

crystal_table_header = [
    html.Thead(
        html.Tr(
            [
                html.Th("a"),
                html.Th("b"),
                html.Th("c"),
                html.Th("alpha"),
                html.Th("beta"),
                html.Th("gamma"),
                html.Th("Orientation"),
                html.Th("Space Group"),
            ]
        )
    )
]

crystal_table_row1 = html.Tr(
    [
        html.Td("-"),
        html.Td("-"),
        html.Td("-"),
        html.Td("-"),
        html.Td("-"),
        html.Td("-"),
        html.Td("-"),
        html.Td("-"),
    ]
)

crystal_table_body = [html.Tbody([crystal_table_row1])]

beam_values = [{"Wavelength": "-", "Sample to Source Direction": "-"}]

beam_headers = [
    {"name": "Wavelength", "id": "Wavelength"},
    {"name": "Sample to Source Direction", "id": "Sample to Source Direction"},
]

experiment_summary = dbc.Card(
    [
        dbc.Card(
            [
                dbc.CardHeader("Detector"),
                dbc.CardBody(
                    dbc.Table(
                        detector_table_header + detector_table_body,
                        bordered=True,
                        id="detector-params",
                    )
                ),
            ]
        ),
        dbc.Card(
            [
                dbc.CardHeader("Beam"),
                dbc.CardBody(
                    dbc.ListGroup(
                        [
                            dash_table.DataTable(
                                beam_values,
                                beam_headers,
                                id="beam-params",
                                style_header={
                                    "color": "white",
                                    "backgroundColor": "black",
                                },
                                style_data={
                                    "color": "white",
                                    "backgroundColor": "black",
                                },
                            ),
                        ],
                    )
                ),
            ]
        ),
        dbc.Card(
            [
                dbc.CardHeader("Scan"),
                dbc.CardBody(
                    dbc.ListGroup(
                        [
                            dbc.Table(
                                scan_table_header + scan_table_body,
                                bordered=True,
                                id="scan-params",
                            )
                        ]
                    )
                ),
            ]
        ),
        dbc.Card(
            [
                dbc.CardHeader("Goniometer"),
                dbc.CardBody(
                    dbc.ListGroup(
                        [
                            dbc.Table(
                                goniometer_table_header + goniometer_table_body,
                                bordered=True,
                                id="goniometer-params",
                            )
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
                            dbc.Table(
                                crystal_table_header + crystal_table_body,
                                bordered=True,
                                id="crystal-params",
                            )
                        ]
                    )
                ),
            ]
        ),
    ],
    body=True,
    id="experiment-summary",
)

find_spots_tab = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button("Run", n_clicks=0, id="dials-find-spots"), width=8
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
                                dbc.DropdownMenu(
                                    label="Threshold Algorithm",
                                    children=[
                                        dbc.DropdownMenuItem("dispersion"),
                                        dbc.DropdownMenuItem("dispersion extended"),
                                        dbc.DropdownMenuItem("radial profile"),
                                    ],
                                ),
                                style={"margin-top": "25px"},
                            ),
                            dbc.Accordion(
                                [dbc.AccordionItem([], title="Advanced")],
                                start_collapsed=True,
                            ),
                        ]
                    ),
                    dbc.Col(
                        html.P(
                            dcc.RangeSlider(
                                1,
                                20,
                                1,
                                marks=None,
                                allowCross=False,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                            style={"margin-top": "25px"},
                        )
                    ),
                ]
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
                                            "maxHeight": "500px",
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
    ),
)

image_viewer_tab = dbc.Card(
    dbc.CardImg(src=app.get_asset_url("image_viewer_with_line_plot.png"), top=True)
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
                                            "height": "84vh",
                                            "maxHeight": "84vh",
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
                                dbc.Tab(generic_tab, label="Integrate", disabled=True),
                                dbc.Tab(generic_tab, label="Scale", disabled=True),
                            ],
                            id="algorithm-tabs",
                            style={},
                        ),
                    )
                ),
            ]
        ),
    ]
)


@app.callback(
    [
        Output("algorithm-tabs", "children"),
        Output("open-files", "children"),
        Output("beam-params", "data"),
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
    ],
)
def event_handler(
    open_files_clicks_list,
    open_files,
    algorithm_tabs,
    import_filename,
    import_content,
    find_spots_n_clicks,
    beam_params,
):

    triggered_id = callback_context.triggered_id
    print(f"Triggered id : {triggered_id}")
    logs = file_manager.get_logs()

    ## Nothing triggered
    if triggered_id is None:
        return algorithm_tabs, open_files, beam_params, *logs

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
        beam_params = display_manager.update_beam_params(
            beam_params, file_manager.selected_file
        )
        return algorithm_tabs, open_files, beam_params, *logs

    ## Running find spots
    if triggered_id == "dials-find-spots":
        logs[1] = file_manager.run(AlgorithmType.dials_find_spots)
        algorithm_tabs = display_manager.update_algorithm_tabs(
            algorithm_tabs, file_manager.selected_file
        )
        return algorithm_tabs, open_files, beam_params, *logs

    ## Clicked on file
    clicked_id = callback_context.triggered[0]["prop_id"]
    clicked_id = int(json.loads(clicked_id.split(".")[0])["index"])
    open_files = display_manager.select_file(open_files, clicked_id)

    # Update logs
    file_manager.update_selected_file(clicked_id)
    logs = file_manager.get_logs()
    algorithm_tabs = display_manager.update_algorithm_tabs(
        algorithm_tabs, file_manager.selected_file
    )
    return algorithm_tabs, open_files, beam_params, *logs


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
