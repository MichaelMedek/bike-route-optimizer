"""Streamlit 3D bike-route viewer — entry shell.

`streamlit run app_webmap.py` needs a script path; this file's main() IS the page orchestration
(the top-level run body). Every reusable helper lives in bike_router.ui.app (mirror-tested under
the package); main() only wires them top → bottom.
"""

import streamlit as st

from bike_router.ui.app import (
    compute_button,
    configure_logging,
    download_graph_with_bar,
    render_controls,
    render_map,
    render_route_output,
    seed_state,
)


def main() -> None:
    st.set_page_config(page_title="Bike Route Optimizer", layout="centered")
    configure_logging()
    download_graph_with_bar()
    seed_state()
    st.title("🚲 Bike Route Optimizer")
    origin, destination = render_controls()
    compute_button(origin=origin, destination=destination)
    render_map(origin=origin, destination=destination)
    if st.session_state.result is not None:
        render_route_output(result=st.session_state.result)


if __name__ == "__main__":
    main()
