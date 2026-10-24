"""Embed the canvas pitch player (pitch_player.html) in a Streamlit page."""

import json
from pathlib import Path

import streamlit as st

TEMPLATE = Path(__file__).with_name("pitch_player.html")


def render(payload: dict, height: int = 820) -> None:
    """Draw one playback window from ui.payload.build_window.

    Playback runs entirely in the browser (requestAnimationFrame, interpolating
    between 25 Hz frames), so play, pause and scrubbing never round-trip to Python.
    With head data in the payload it also draws vision cones, yellow during head checks.
    """
    # Names come from data files: escape "</" so no value can close the <script> tag.
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    st.iframe(TEMPLATE.read_text().replace("__DATA__", data), height=height)
