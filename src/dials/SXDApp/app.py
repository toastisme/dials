from __future__ import annotations

import base64
from os import mkdir
from os.path import isdir, splitext

import dash_bootstrap_components as dbc
import procrunner
from app_import_tab import ImportTab
from dash import Dash, Input, Output, dcc, html

app = Dash(external_stylesheets=[dbc.themes.DARKLY])

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
                                    html.H6("Active Files"),
                                    html.Div(
                                        dbc.ListGroup(id="active-files", children=[]),
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
                                dbc.Tab(generic_tab, label="Experiment Summary"),
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
        Output("dials-import-log", "children"),
        Output("active-files", "children"),
    ],
    [
        Input("dials-import", "filename"),
        Input("dials-import", "contents"),
        Input("active-files", "children"),
        Input("algorithm-tabs", "children"),
    ],
)
def run_dials_import(filenames, contents, active_files_list, algorithm_tabs):

    if filenames is None:
        return algorithm_tabs, None, active_files_list

    # Create local file
    filename, ext = splitext(filenames)
    file_dir = "tmp/" + filename
    if not isdir("tmp"):
        mkdir("tmp")
    if not isdir(file_dir):
        mkdir(file_dir)
    file_path = file_dir + "/" + filename + ext
    contents_type, contents_string = contents.split(",")
    decoded = base64.b64decode(contents_string)
    with open(file_path, "wb") as g:
        g.write(decoded)

    result = procrunner.run(("dials.import", file_path))

    # Get log text
    stdout = result.stdout.decode().split("\n")
    text = []
    for i in stdout:
        text.append(i)
        text.append(html.Br())

    # Update active file
    if active_files_list is None:
        active_files_list = []
    for i in active_files_list:
        i["props"]["active"] = False
    active_files_list.append(dbc.ListGroupItem(filename, active=True))

    # Update disabled algorithm tabs
    algorithm_tabs[1]["props"]["disabled"] = False

    return algorithm_tabs, text, active_files_list


@app.callback(
    Output("dials-find-spots-log", "children"), Input("dials-find-spots", "n_clicks")
)
def run_find_spots(n_clicks):

    if n_clicks == 0:
        return None

    result = procrunner.run(("dials.find_spots", "imported.expt"))

    # Get log text
    stdout = result.stdout.decode().split("\n")
    text = []
    for i in stdout:
        text.append(i)
        text.append(html.Br())

    # Update disabled algorithm tabs
    # algorithm_tabs[2]["props"]["disabled"] = False

    return text


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
