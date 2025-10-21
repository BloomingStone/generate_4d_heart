from generate_4d_heart.rotate_dsa.rotate_drr.torch_drr import TorchDRR
import torch

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    res = torch.eye(4)
    res[:3, :3] = torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))
    return res


def test_R_T():
    drr = TorchDRR()
    drr.rotate_cfg.alpha_start = 60 #degree
    drr.rotate_cfg.beta_start = 30
    
    alpha, beta, _ = drr.rotate_cfg.get_rotaiton_radian_at_frame(0)
    sod = drr.c_arm_cfg.sod
    
    # in DiffDrr, X Y Z coreesponding to R A S, the primary rotation is around Z axis 
    # and the secondary rotation is around X axis. And as the primary rotation is from 
    # Anterior(Y) to Right(X), alpha should be negative. Similarly, as the secondary 
    # rotation is from Anterior(Y) to Inferior(-Z), beta should be negative.
    R_z_alpha = _axis_angle_rotation("Z", torch.tensor(-alpha))
    R_x_beta = _axis_angle_rotation("X", torch.tensor(-beta))
    translate = torch.eye(4)
    translate[:3, 3] = torch.tensor([0.0, sod, 0.0])
    reorient = drr.reorient
    
    # internal rotation (zxy) can translate to external rotation with oppisite order (YXZ), so here first rotate around X_world, then Y_world
    M_c2w_gt = R_z_alpha @ R_x_beta @ translate @ reorient
    
    R, T = drr.get_R_T_at_frame(0)
    M_c2w = torch.eye(4)    # camera to world
    M_c2w[:3, :3] = R
    M_c2w[:3, 3] = T
    
    assert torch.allclose(M_c2w, M_c2w_gt)
    
    camera_direction = torch.tensor([0.0, 0.0, 1.0, 0.0])
    camera_direction = M_c2w @ camera_direction
    camera_pose = torch.tensor([0.0, 0.0, 0.0, 1.0])
    camera_pose = M_c2w @ camera_pose
    
    # ensure the direction point to the center of world
    unit_pose = camera_pose / torch.norm(camera_pose)
    assert torch.allclose(unit_pose[:3], -camera_direction[:3])
    
    
if __name__ == "__main__":
    test_R_T()