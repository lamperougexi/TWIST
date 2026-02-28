import torch


def compute_orbital_angular_momentum(body_pos, body_vel, body_masses):
    """Compute batched whole-body orbital angular momentum around the system COM.

    Args:
        body_pos: Tensor of shape (num_envs, num_bodies, 3)
        body_vel: Tensor of shape (num_envs, num_bodies, 3)
        body_masses: Tensor of shape (num_envs, num_bodies)

    Returns:
        Tensor of shape (num_envs, 3), orbital angular momentum around COM.
    """
    masses = body_masses.unsqueeze(-1)
    total_mass = torch.sum(masses, dim=1, keepdim=True).clamp(min=1e-6)

    com_pos = torch.sum(body_pos * masses, dim=1, keepdim=True) / total_mass
    com_vel = torch.sum(body_vel * masses, dim=1, keepdim=True) / total_mass

    rel_pos = body_pos - com_pos
    rel_momentum = masses * (body_vel - com_vel)
    ang_momentum = torch.cross(rel_pos, rel_momentum, dim=-1).sum(dim=1)
    return ang_momentum
