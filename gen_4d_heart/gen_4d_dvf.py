from pathlib import Path
from argparse import ArgumentParser
import json

import nibabel as nib
from tqdm import tqdm

from .roi import ROI
from .ssm import SSM, polydata_to_label_volume
from .shape_morph import ShapeMorphPredictor
from . import NUM_TOTAL_PHASE


def read_nifti(path: Path) -> nib.nifti1.Nifti1Image:
    image = nib.loadsave.load(path)
    assert isinstance(image, nib.nifti1.Nifti1Image)
    return image


def main():
    parser = ArgumentParser(description=""""
Generate dense velocity fields(dvf) that can warp a static heart CTA into 20 phases of a cardiac cycle. 
The static images should be stored as
-  cavity_dir: cavity label, like mmHWS (LV-myo=1, LV=2, RV=3, LA=4, RA=5)
    - [case_name].nii.gz

The generated images will be stored as
- output_dir
    - landmark: landmark points/label, stored as vtk file/nii.gz
        - [case_name].vtk
        - [case_name].nii.gz
    - cavity_gif
        - [case_name].gif
    - roi_info:
        - [case_name].json
    - dvf: saved the zoomed ROI parts, shape=144x144x128, which can be recovered by corresponding ROI info
        - [case_name]
            - phase_00.nii.gz
            - ...
    - (4d_cavity, optional): cavity label for different phases, also zoomed to 144x144x128
        - [case_name].nii.gz
""")
    parser.add_argument("-i", "--input", help="input directory of static cavity labels")
    parser.add_argument("-o", "--output", help="output directory of generated images")
    parser.add_argument("-d", "--device", help="cuda id to use for ShapeMorph", default=0)
    parser.add_argument("--gen_cavity", help="generate cavity label", action='store_true')
    parser.add_argument("--gen_gif", help="generate cavity gif", action='store_true')
    parser.add_argument("--ckpt", help="checkpoint path for ShapeMorph")

    args = parser.parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    (landmark_dir := output_dir / 'landmark').mkdir(exist_ok=True, parents=True)
    (roi_info_dir := output_dir / 'roi_info').mkdir(exist_ok=True, parents=True)
    dvf_dir = output_dir / 'dvf'

    device = args.device
    if args.ckpt is None:
        checkpoint = Path(__file__).parent / 'checkpoints' / 'ShapeMorph.pth'
    else:
        checkpoint = Path(args.ckpt)
    ssm_dir = Path(__file__).parent / 'ssm'
    ssm = SSM(
        template_surface=ssm_dir / 'ssm_template.vtk',
        b_motion=ssm_dir / 'b_motion_mean_per_phase.npy',
        P_motion=ssm_dir / 'P_motion.npy',
    )
    predictor = ShapeMorphPredictor(
        device_id=device, 
        checkpoint_path=checkpoint,
    )

    output_dir.mkdir(exist_ok=True, parents=True)
    for cavity_path in tqdm(list(sorted((input_dir).glob('*.nii.gz'))), desc='Processing images'):
        case_name = cavity_path.stem.split('.')[0]
        print(case_name)
        cavity = read_nifti(cavity_path)
        roi = ROI.get_from_cavity(cavity=cavity, padding=10)
        with open(roi_info_dir / f'{case_name}.json', 'w') as f:
            json.dump(roi.to_dict(), f)
        cavity_zoomed = roi.crop_zoom(cavity, is_label=True)
        ssm_res = ssm.apply(cavity_zoomed, device=device)
        ssm_res.landmark_vtk.save(landmark_dir / f'{case_name}.vtk')
        nib.loadsave.save(ssm_res.get_landmark_volume(), landmark_dir / f'{case_name}.nii.gz')
        if args.gen_gif:
            (cavity_gif_dir := output_dir / 'cavity_gif').mkdir(exist_ok=True, parents=True)
            ssm_res.save_gif(cavity_gif_dir / f'{case_name}.gif')
        (dvf_case_dir := dvf_dir / case_name).mkdir(exist_ok=True, parents=True)
        cavity_list = []
        predictor.set_shared_inputs(
            source_cavity_zoomed=cavity_zoomed,     # TODO 这里此前使用的是landmark，这里换用cavity尝试一下
        )
        for phase in tqdm(range(NUM_TOTAL_PHASE), desc='generate moving phases'):
            cavity_phase = ssm_res.get_motion_volume(phase)
            dvf = predictor.predict(
                target_cavity_zoomed=cavity_phase,
            )
            nib.loadsave.save(dvf, dvf_case_dir / f'phase_{phase:02d}.nii.gz')
            
            if args.gen_cavity:
                cavity_list.append(cavity_phase)

        if args.gen_cavity:
            (cavity_4d_dir := output_dir / '4d_cavity').mkdir(exist_ok=True, parents=True)
            concat = nib.funcs.concat_images(cavity_list)
            nib.loadsave.save(concat, cavity_4d_dir / f'{case_name}.nii.gz')



if __name__ == '__main__':
    main()