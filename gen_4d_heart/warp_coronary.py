from monai.networks.blocks.warp import Warp, DVF2DDF
from monai.transforms import AsDiscrete # type: ignore
import torch
import cupy as cp
from cupyx.scipy.ndimage import label, center_of_mass, distance_transform_edt
from torch.utils.dlpack import from_dlpack, to_dlpack


def separate_coronary(coronary: torch.Tensor) -> torch.Tensor:
    """
    separate coronary to LCA(1) and RCA(2)
    Args:
        coronary (torch.Tensor): coronary segmentation
    Returns:
        coronary (torch.Tensor): LCA and RCA, marked as 1 and 2
    """
    coronary = cp.from_dlpack(to_dlpack(coronary))
    # Find all connected components
    labeled_array, num_features = label(coronary)  # type: ignore
    
    # If only one component, return original (no separation needed)
    if num_features <= 1:
        return coronary
    
    # Calculate sizes of each component
    component_sizes = cp.bincount(labeled_array.ravel())[1:]  # Skip background (0)
    
    # Find two largest components
    largest_indices = cp.argsort(component_sizes)[-2:][::-1] + 1  # +1 to account for skipped 0
    
    region_0 = (labeled_array == largest_indices[0]).astype(cp.uint8)
    region_1 = (labeled_array == largest_indices[1]).astype(cp.uint8)
    
    center_0 = center_of_mass(region_0)
    center_1 = center_of_mass(region_1)
    
    if center_0[1] > center_1[1]:
        res = region_0 * 1 + region_1 * 2
    else:
        res = region_1 * 1 + region_0 * 2

    return from_dlpack(res.toDlpack())

def dvf_enhance_coronary_by_LV(
    dvf: torch.Tensor,            # (B=1, 3, W, H, D), float tensor on CUDA
    lv_mask: torch.Tensor,        # (W, H, D), binary mask on CUDA (0/1)
    coronary_mask: torch.Tensor,  # (W, H, D), binary mask on CUDA (0/1)
    # eps: float = 1e-3,
    # sigma: float = 5.0
) -> torch.Tensor:
    """
    基于 CuPy 的 GPU 实现：
      1) 计算 EDT 距离图 & 最近背景索引
      2) 对冠脉区域，按距离做 weighted average：
         w = exp(-distance / sigma)
         dvf_new = w * dvf_at_nearest_LV + (1-w) * dvf_orig
      3) DVF2DDF -> Warp 冠脉 mask
    
    Args:
        dvf:               (1,3,W,H,D) 输入 DVF，需在 CUDA 上
        LV_mask:          (W,H,D) 左心室二值 mask，CUDA bool/uint8
        coronary_mask:    (W,H,D) 冠脉二值 mask，CUDA bool/uint8
        eps:               距离为 0 时的最小偏置，防止除零
        sigma:             控制权重衰减的长度尺度

    Returns:
        
    """
    dvf_cp = cp.from_dlpack(to_dlpack(dvf))  # (1,3,W,H,D), float32
    lv_cp = cp.from_dlpack(to_dlpack(lv_mask)).astype(cp.bool_)  # (W,H,D), bool
    coronary_cp = cp.from_dlpack(to_dlpack(coronary_mask)).astype(cp.bool_)  # (W,H,D), bool
    
    (ix, iy, iz) = distance_transform_edt(
        ~lv_cp,
        return_distances=False,
        return_indices=True
    )
    
    jx, jy, jz = cp.where(coronary_cp)
    ix = ix[jx, jy, jz]
    iy = iy[jx, jy, jz]
    iz = iz[jx, jy, jz]
    
    dvf_cp[:, :, jx, jy, jz] += dvf_cp[:, :, ix, iy, iz]
    
    return from_dlpack(dvf_cp.toDlpack())


def test_once():
    from gen_4d_heart.roi import ROI
    import nibabel as nib
    from torch.nn import functional as  F
    
    roi_json = "/media/F/sj/Data/ASOCA/normal_gen_4d_output/roi_info/Normal_02.json"
    croped_dvf_path = "/media/F/sj/Data/ASOCA/normal_gen_4d_output/dvf/Normal_02/phase_10.nii.gz"
    cavity_path = "/media/F/sj/Data/ASOCA/normal_gen_4d/cavity/Normal_02.nii.gz"
    coronary_path = "/media/F/sj/Data/ASOCA/normal_gen_4d/coronary/Normal_02.nii.gz"
    
    roi = ROI.from_json(roi_json)
    zoom_rate = 1 / roi.get_zoom_rate().reshape(1, 3, 1, 1, 1)
    dvf = nib.loadsave.load(croped_dvf_path)
    dvf_data = dvf.get_fdata()
    dvf_tensor = torch.from_numpy(dvf_data).cuda()
    dvf_tensor = dvf_tensor.squeeze().permute(3, 0, 1, 2)[None] # (1,3,H,W,D)
    dvf_tensor = F.interpolate(dvf_tensor, scale_factor=zoom_rate.flatten().tolist(), mode='trilinear', align_corners=False)
    dvf_tensor = dvf_tensor * torch.from_numpy(zoom_rate).cuda()
    
    cavity = nib.loadsave.load(cavity_path)
    cavity = roi.crop(cavity)
    cavity_data = cavity.get_fdata()
    cavity_tensor = torch.from_numpy(cavity_data).cuda()
    lv_tensor = cavity_tensor == 2

    coronary = nib.loadsave.load(coronary_path)
    coronary = roi.crop(coronary)
    coronary_data = coronary.get_fdata()
    coronary_tensor = torch.from_numpy(coronary_data).cuda()
    
    new_dvf = dvf_enhance_coronary_by_LV(dvf_tensor, lv_tensor, coronary_tensor)
    new_dvf = new_dvf.squeeze().permute(1, 2, 3, 0).cpu().numpy()
    new_dvf = new_dvf * zoom_rate.flatten().tolist()
    new_dvf = new_dvf.astype(cp.float32)
    nib.save(nib.Nifti1Image(new_dvf, affine=cavity.affine), "test_new_dvf.nii.gz")

if __name__ == "__main__":
    test_once()