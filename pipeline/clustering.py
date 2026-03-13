"""Scene clustering stage (shot embeddings → scene groups).

Planned implementation: HDBSCAN over shot embeddings, with optional feature fusion
and constraints from actor presence.
"""

from __future__ import annotations


class SceneClusterer:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Clustering module will be implemented after shot embeddings.")

