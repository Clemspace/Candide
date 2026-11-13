"""Errors shared by the engines and the server (no PyTorch)."""


class ProbeError(ValueError):
    """A request the engine refuses, with a message meant for the user."""
