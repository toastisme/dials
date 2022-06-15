from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

backgroundColor = "#9E9E9E"


class ImportTab:
    def content(self):
        return [
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Upload(
                                        children=html.Div(
                                            [
                                                dbc.Button(
                                                    "Load",
                                                    n_clicks=0,
                                                ),
                                            ]
                                        ),
                                        id="dials-import",
                                    ),
                                    width=8,
                                ),
                                dbc.Col(
                                    html.P(
                                        dcc.Link(
                                            dbc.Button(
                                                "Documentation", color="secondary"
                                            ),
                                            href="https://dials.github.io/documentation/programs/dials_import.html",
                                            target="_blank",
                                        ),
                                        style={"margin-left": "65px"},
                                    ),
                                    width=1,
                                    align="end",
                                ),
                            ],
                        )
                    ]
                ),
                style={"backgroundColor": backgroundColor},
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H6("Output"),
                        dbc.Card(
                            dbc.CardBody(
                                dbc.Spinner(
                                    html.Div(
                                        id="dials-import-log",
                                        children=[],
                                        style={
                                            "height": "66.5vh",
                                            "maxHeight": "66.5vh",
                                            "overflow": "scroll",
                                        },
                                    )
                                )
                            ),
                        ),
                    ]
                ),
                style={"backgroundColor": backgroundColor},
            ),
        ]
