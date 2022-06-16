from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


class FindSpotsTab:
    def content(self):
        return [
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Button("Run"),
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
        ]
