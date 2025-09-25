from pathlib import Path
import json
import logging

import nibabel as nib
from tqdm import tqdm

from .roi import ROI
from .ssm import SSM
from .shape_morph import ShapeMorphPredictor
from . import NUM_TOTAL_PHASE

logger = logging.getLogger(__name__)

def read_nifti(path: Path) -> nib.nifti1.Nifti1Image:
    image = nib.loadsave.load(path)
    assert isinstance(image, nib.nifti1.Nifti1Image)
    return image

# TODO 提取分离主要处理逻辑和save逻辑
def generate_4d_dvf(
    input_dir: Path,
    output_dir: Path,
    ckpt: Path = Path(__file__).parent / 'checkpoints' / 'ShapeMorph.pth',
    gen_cavity: bool = False,
    gen_gif: bool = False,
    device: int = 0
):
    """Generate dense velocity fields(dvf) that can warp a static heart CTA into 20 phases of a cardiac cycle. 
The static images should be stored as
-  cavity_dir: cavity label, like mmHWS (LV-myo=1, LV=2, RV=3, LA=4, RA=5)
    - [case_name].nii.gz

The generated images will be stored as
- output_dir
    - landmark: landmark points/label, stored as vtk file/nii.gz
        - [case_name].vtk
        - [case_name].nii.gz
    - roi_info:
        - [case_name].json
    - dvf: saved the zoomed ROI parts, shape=144x144x128, which can be recovered by corresponding ROI info
        - [case_name]
            - phase_00.nii.gz
            - ...
    - (cavity_gif, optional)
        - [case_name].gif
    - (4d_cavity, optional): cavity label for different phases, also zoomed to 144x144x128
        - [case_name].nii.gz

    Args:
        input_dir (Path): the path to cavity_dir which contains the cavity labels  (LV-myo=1, LV=2, RV=3, LA=4, RA=5)
        output_dir (Path): output dir contains landmark, roi_info, dvf, cavity_gif(optional), 4d_cavity(optional)
        ckpt (Path, optional): checkpoint path of Shape Morph. Defaults to Path(__file__).parent/'checkpoints'/'ShapeMorph.pth'.
        gen_cavity (bool, optional): whether to generate 4d cavity. Defaults to False.
        gen_gif (bool, optional): whether to generate gif. Defaults to False.
        device (int, optional): the id of gpu device to use. Defaults to 0.
    """
    (landmark_dir := output_dir / 'landmark').mkdir(exist_ok=True, parents=True)
    (roi_info_dir := output_dir / 'roi_info').mkdir(exist_ok=True, parents=True)
    dvf_dir = output_dir / 'dvf'

    ssm_dir = Path(__file__).parent / 'ssm'
    ssm = SSM(
        template_surface=ssm_dir / 'ssm_template.vtk',
        b_motion=ssm_dir / 'b_motion_mean_per_phase.npy',
        P_motion=ssm_dir / 'P_motion.npy',
    )
    shape_morpher_predictor = ShapeMorphPredictor(
        device_id=device, 
        checkpoint_path=ckpt,
    )

    output_dir.mkdir(exist_ok=True, parents=True)
    for cavity_path in tqdm(list(sorted((input_dir).glob('*.nii.gz'))), desc='Processing images'):
        # load cavity
        case_name = cavity_path.stem.split('.')[0]
        cavity = read_nifti(cavity_path)
        
        # calculate and save roi
        roi = ROI.get_from_cavity(cavity=cavity, padding=10)
        with open(roi_info_dir / f'{case_name}.json', 'w') as f:
            json.dump(roi.to_dict(), f)
        
        # cut roi and zoom cavity
        cavity_zoomed = roi.crop_zoom(cavity, is_label=True)
        
        # apply ssm no cavity, then save landmark
        try:
            ssm_res = ssm.apply(cavity_zoomed, device=device)
        except Exception as e:
            logger.error(f"case {cavity_path} fail to be apply ssm with error: {e}")
            continue
        ssm_res.landmark_vtk.save(landmark_dir / f'{case_name}.vtk')
        nib.loadsave.save(ssm_res.get_landmark_volume(), landmark_dir / f'{case_name}.nii.gz')
        
        # generate heart beating gif (optional)
        if gen_gif:
            (cavity_gif_dir := output_dir / 'cavity_gif').mkdir(exist_ok=True, parents=True)
            ssm_res.save_gif(cavity_gif_dir / f'{case_name}.gif')
        
        # setting shape morpher
        shape_morpher_predictor.set_shared_inputs(
            source_cavity_zoomed=cavity_zoomed,     # TODO 这里此前使用的是landmark，这里换用cavity尝试一下
        )
        
        # use shape morpher generate dvf at each heart phase
        cavity_list = []
        (dvf_case_dir := dvf_dir / case_name).mkdir(exist_ok=True, parents=True)
        for phase in tqdm(range(NUM_TOTAL_PHASE), desc='generate moving phases'):
            cavity_phase = ssm_res.get_motion_volume(phase)
            
            try:
                dvf = shape_morpher_predictor.predict(
                    target_cavity_zoomed=cavity_phase,
                )
            except Exception as e:
                logger.error(f"when shape morpher prediting, case {cavity_path} fail at {phase=} with error: {e}")
                continue
        
            nib.loadsave.save(dvf, dvf_case_dir / f'phase_{phase:02d}.nii.gz')
            
            if gen_cavity:
                cavity_list.append(cavity_phase)

        if gen_cavity:
            (cavity_4d_dir := output_dir / '4d_cavity').mkdir(exist_ok=True, parents=True)
            concat = nib.funcs.concat_images(cavity_list)
            nib.loadsave.save(concat, cavity_4d_dir / f'{case_name}.nii.gz')


if __name__ == '__main__':
    from jsonargparse import auto_cli
    auto_cli(generate_4d_dvf)