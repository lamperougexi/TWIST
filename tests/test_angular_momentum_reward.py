from pathlib import Path
import importlib.util

import torch

_MODULE_PATH = Path(__file__).resolve().parents[1] / "legged_gym" / "legged_gym" / "envs" / "base" / "angular_momentum_utils.py"
_SPEC = importlib.util.spec_from_file_location("angular_momentum", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
compute_orbital_angular_momentum = _MODULE.compute_orbital_angular_momentum


def _assert_close(actual, expected, atol=1e-6):
    if not torch.allclose(actual, expected, atol=atol, rtol=0):
        raise AssertionError(f"Mismatch.\nactual={actual}\nexpected={expected}")


def test_known_two_body_case():
    # Two masses rotating around origin in opposite linear directions.
    # Expected Lz = 4.0 for this setup.
    body_pos = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]], dtype=torch.float32)
    body_vel = torch.tensor([[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]], dtype=torch.float32)
    body_masses = torch.tensor([[2.0, 2.0]], dtype=torch.float32)

    ang_mom = compute_orbital_angular_momentum(body_pos, body_vel, body_masses)
    expected = torch.tensor([[0.0, 0.0, 4.0]], dtype=torch.float32)
    _assert_close(ang_mom, expected)


def test_invariant_to_global_position_shift():
    body_pos = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]], dtype=torch.float32)
    body_vel = torch.tensor([[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]], dtype=torch.float32)
    body_masses = torch.tensor([[2.0, 2.0]], dtype=torch.float32)

    baseline = compute_orbital_angular_momentum(body_pos, body_vel, body_masses)
    shifted = compute_orbital_angular_momentum(
        body_pos + torch.tensor([[[3.0, -2.0, 5.0], [3.0, -2.0, 5.0]]], dtype=torch.float32),
        body_vel,
        body_masses,
    )
    _assert_close(shifted, baseline)


def test_invariant_to_uniform_velocity_bias():
    body_pos = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]], dtype=torch.float32)
    body_vel = torch.tensor([[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]], dtype=torch.float32)
    body_masses = torch.tensor([[2.0, 2.0]], dtype=torch.float32)

    baseline = compute_orbital_angular_momentum(body_pos, body_vel, body_masses)
    biased = compute_orbital_angular_momentum(
        body_pos,
        body_vel + torch.tensor([[[2.0, -3.0, 1.0], [2.0, -3.0, 1.0]]], dtype=torch.float32),
        body_masses,
    )
    _assert_close(biased, baseline)


if __name__ == "__main__":
    test_known_two_body_case()
    test_invariant_to_global_position_shift()
    test_invariant_to_uniform_velocity_bias()
    print("Angular momentum reward tests passed.")
