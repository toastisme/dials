from __future__ import annotations

import pickle


def reflections(obj, outfile):
    """
    Dump the given object to file

    :param obj: The reflection list to dump
    :param outfile: The output file name or file object
    """
    if isinstance(outfile, str):
        with open(outfile, "wb") as outfile:
            pickle.dump(obj, outfile, pickle.HIGHEST_PROTOCOL)

    # Otherwise assume the input is a file and write to it
    else:
        pickle.dump(obj, outfile, pickle.HIGHEST_PROTOCOL)


def reference(obj, outfile):
    """
    Dump the given object to file

    :param obj: The reference list to dump
    :param outfile: The output file name or file object
    """

    if isinstance(outfile, str):
        with open(outfile, "wb") as outfile:
            pickle.dump(obj, outfile, pickle.HIGHEST_PROTOCOL)

    # Otherwise assume the input is a file and write to it
    else:
        pickle.dump(obj, outfile, pickle.HIGHEST_PROTOCOL)
