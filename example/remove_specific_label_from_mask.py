import click
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from pathlib import Path


@click.command()
@click.option("--input-dir", "-i", default="./annotations")
@click.option("--output-dir", "-o", default="./annotations_modified")
@click.option("--removal-target-label", "-r", default=2)
def main(input_dir, output_dir, removal_target_label):
    input_dir_pathlib = Path(input_dir)
    output_dir_pathlib = Path(output_dir)
    if output_dir_pathlib.exists():
        shutil.rmtree(output_dir)
    output_dir_pathlib.mkdir()

    seg_path_list = [path for path in input_dir_pathlib.glob("*.png")]
    for seg_path in tqdm(seg_path_list):
        seg_mask = cv2.imread(str(seg_path), cv2.IMREAD_ANYDEPTH)
        mask_name = seg_path.name
        seg_mask_modified = seg_mask
        seg_mask_modified[seg_mask_modified == removal_target_label] = 0
        output_mask_pathstr = str(output_dir_pathlib.joinpath(mask_name))
        cv2.imwrite(output_mask_pathstr, seg_mask_modified)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()