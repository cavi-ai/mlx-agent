"""Ownership collaborator for tests focused on inference result handling."""
from unittest.mock import patch


def isolate(test):
    from mlx_agent.model_serving import _OwnedClaim
    guard = patch('mlx_agent.model_serving.acquire', return_value=_OwnedClaim())
    guard.start()
    test.addCleanup(guard.stop)
