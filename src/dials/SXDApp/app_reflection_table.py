from __future__ import annotations

import dash_bootstrap_components as dbc
import experiment_params
from dash import dash_table, html

reflection_table = html.Div(
    [
        dbc.Offcanvas(
            id="reflection-table-window",
            title="Reflection Table",
            is_open=False,
            placement="bottom",
            backdrop=False,
            style={
                "top": "13vh",
                "left": "59vw",
                "height": "86vh",
                "width": "41vw",
            },
            children=[
                dbc.ListGroup(
                    [
                        dash_table.DataTable(
                            experiment_params.reflection_table_values,
                            experiment_params.reflection_table_headers,
                            id="reflection-table",
                            style_header=experiment_params.style_header[0],
                            style_data=experiment_params.style_data[0],
                        ),
                    ],
                    style={
                        "height": "78vh",
                        "maxHeight": "78vh",
                        "width": "78vh",
                        "maxWidth": "78vh",
                        "overflow": "scroll",
                    },
                )
            ],
        ),
    ],
)
