from __future__ import annotations

import logging

logging.getLogger("dials").addHandler(logging.NullHandler())

# Intercept easy_mp exceptions to extract stack traces before they are lost at
# the libtbx process boundary/the easy_mp API. In the case of a subprocess
# crash we print the subprocess stack trace, which will be most useful for
# debugging parallelized sections of DIALS code.
import libtbx.scheduling.stacktrace as _lss


def _stacktrace_tracer(error, trace, intercepted_call=_lss.set_last_exception):
    """Intercepts and prints ephemeral stacktraces."""
    if error and trace:
        logging.getLogger("dials").error(
            "\n\neasy_mp crash detected; subprocess trace: ----\n%s%s\n%s\n\n",
            "".join(trace),
            error,
            "-" * 46,
        )
    return intercepted_call(error, trace)


if _lss.set_last_exception.__doc__ != _stacktrace_tracer.__doc__:
    # ensure function is only redirected once
    _lss.set_last_exception = _stacktrace_tracer
