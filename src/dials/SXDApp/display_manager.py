from __future__ import annotations

import dash_bootstrap_components as dbc


class DisplayManager:

    """
    Manages which files are displayed on the GUI
    """

    def add_file(self, active_files_list, filename):
        active_files_list.append(
            dbc.ListGroupItem(
                filename,
                id={"type": "open-file", "index": len(active_files_list)},
                action=True,
            )
        )
        return active_files_list
