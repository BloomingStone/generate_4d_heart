from pathlib import Path
from argparse import ArgumentParser
import json

import nibabel as nib
from tqdm import tqdm

from .roi import ROI
from .ssm import SSM
from .shape_morph import ShapeMorphPredictor
from . import NUM_TOTAL_PHASE


def read_nifti(path: Path) -> nib.nifti1.Nifti1Image:
    image = nib.loadsave.load(path)
    assert isinstance(image, nib.nifti1.Nifti1Image)
    return image


def main():
    parser = ArgumentParser(description=""""
Generate dense displacement fields(ddf) that can warp a static heart CTA into 20 phases of a cardiac cycle. 
The static images should be stored as
- input_dir
    - image
        - [image_name].nii.gz
    - coronary: coronary artery label
        - [image_name].nii.gz   # name should be the same as images
    - cavity: cavity label, like mmHWS (LV-myo=1, LV=2, RV=3, LA=4, RA=5)
        - [image_name].nii.gz

The generated images will be stored as
- output_dir
    - landmark: landmark points/label, stored as vtk file/nii.gz
        - [image_name].vtk
        - [image_name].nii.gz
    - cavity_gif
        - [image_name].gif
    - roi_info:
        - [image_name].json
    - ddf (saved the zoomed ROI parts, which can be recovered by corresponding ROI info)
        - [image_name]
            - phase_00.nii.gz
            - ...
    - (image, optional)
        - image_name.nii.gz
            - phase_00.nii.gz
    - (4d_cavity, optional): cavity label for different phases
        - [image_name].nii.gz
    - (4d_coronary, optional)
        - [image_name].nii.gz
""")
    parser.add_argument("-i", "--input", help="input directory of static images")
    parser.add_argument("-o", "--output", help="output directory of generated images")
    parser.add_argument("-d", "--device", help="cuda id to use for ShapeMorph", default=0)
    parser.add_argument("--gen_cavity", help="generate cavity label", action='store_true')
    parser.add_argument("--gen_image", help="generate 4d image", action='store_true')
    parser.add_argument("--gen_coronary", help="generate 4d coronary label", action='store_true')


    args = parser.parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    (landmark_dir := output_dir / 'landmark').mkdir(exist_ok=True, parents=True)
    (cavity_gif_dir := output_dir / 'cavity_gif').mkdir(exist_ok=True, parents=True)
    (roi_info_dir := output_dir / 'roi_info').mkdir(exist_ok=True, parents=True)
    ddf_dir = output_dir / 'ddf'
    maybe_save_dirs = {}    # TODO 这些可能占内存较大，后续还是不存储4D形式的
    if args.gen_cavity:
        (cavity_4d_dir := output_dir / '4d_cavity').mkdir(exist_ok=True, parents=True)
        maybe_save_dirs['cavity'] = cavity_4d_dir
    if args.gen_coronary:
        (coronary_4d_dir := output_dir / '4d_coronary').mkdir(exist_ok=True, parents=True)
        maybe_save_dirs['coronary'] = coronary_4d_dir

    device = args.device
    checkpoint = Path(__file__).parent / 'checkpoints' / 'ShapeMorph.pth'
    ssm_dir = Path(__file__).parent / 'ssm'
    ssm = SSM(
        template_surface=ssm_dir / 'ssm_template_avg.vtk',
        b_motion=ssm_dir / 'b_motion_mean_per_phase.npy',
        P_motion=ssm_dir / 'P_motion.npy',
    )
    predictor = ShapeMorphPredictor(
        device_id=device, 
        checkpoint_path=checkpoint,
        return_wrapped_image=args.gen_image,
        return_wrapped_coronary=args.gen_coronary
    )

    output_dir.mkdir(exist_ok=True, parents=True)
    image_paths = list(sorted(input_dir.glob('image/*.nii.gz')))
    coronary_paths = list(sorted(input_dir.glob('coronary/*.nii.gz')))
    cavity_paths = list(sorted(input_dir.glob('cavity/*.nii.gz')))
    for image_path, cavity_path, coronary_path in tqdm(list(zip(image_paths, cavity_paths, coronary_paths)), desc='Processing images'):
        image_name = image_path.stem.split('.')[0]
        image = read_nifti(image_path)
        cavity = read_nifti(cavity_path)
        coronary = read_nifti(coronary_path)
        roi = ROI(cavity=cavity, padding=10)
        with open(roi_info_dir / f'{image_name}.json', 'w') as f:
            json.dump(roi.to_dict(), f)
        cavity_zoomed = roi.crop_zoom(cavity, is_label=True)
        ssm_res = ssm.apply(cavity_zoomed, device=device)
        ssm_res.landmark_vtk.save(landmark_dir / f'{image_name}.vtk')
        ssm_res.save_gif(cavity_gif_dir / f'{image_name}.gif')
        (ddf_case_dir := ddf_dir / image_name).mkdir(exist_ok=True, parents=True)
        if args.gen_image:
            (image_case_dir := output_dir / 'image' / image_name).mkdir(exist_ok=True, parents=True)
        else:
            image_case_dir = None
        temp_lists = {
            'cavity': [],
            'coronary': []
        }
        predictor.set_shared_inputs(
            source_cavity_zoomed=cavity_zoomed,
            coronary=coronary,
            original_image=image,
            roi = roi
        )
        for phase in tqdm(range(NUM_TOTAL_PHASE), desc='generate moving phases'):
            cavity_phase = ssm_res.get_motion_volume(phase)
            pred = predictor.predict(
                target_cavity_zoomed=cavity_phase,
            )
            nib.loadsave.save(pred['ddf'], ddf_case_dir / f'phase_{phase:02d}.nii.gz')
            if args.gen_image and image_case_dir is not None:
                nib.loadsave.save(pred['image'], image_case_dir / f'phase_{phase:02d}.nii.gz')

            maybe_need_save = {
                'cavity': cavity_phase,
                'coronary': pred['coronary'],
            }

            for key in maybe_save_dirs.keys():
                temp_lists[key].append(maybe_need_save[key])
        
        for key, dir in maybe_save_dirs.items():
            concat = nib.funcs.concat_images(temp_lists[key])
            nib.loadsave.save(concat, dir / f'{image_name}.nii.gz')



if __name__ == '__main__':
    main()