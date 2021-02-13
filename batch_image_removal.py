
import cv2
import click
import shutil
from pathlib import Path
from tqdm import tqdm
from object_removal import ObjectRemover

@click.command()
@click.option("--input-image-dir", "-i", default="./images")
@click.option("--input-mask-dir", "-m", default="./masks")
@click.option("--output_dir", "-o", default="./output")
@click.option("--hifill-pb", "-pb", default="./pb/hifill.pb")
def main(input_image_dir, input_mask_dir, output_dir, hifill_pb):
    input_image_dir_pathlib = Path(input_image_dir)
    input_mask_dir_pathlib = Path(input_mask_dir)
    output_dir_pathlib = Path(output_dir)
    if output_dir_pathlib.exists():
        shutil.rmtree(output_dir)
    output_dir_pathlib.mkdir()   

    input_image_list = [
        str(path) for path in input_image_dir_pathlib.glob("*") if path.suffix in [".jpg", ".png"]
    ]

    remover = ObjectRemover(hifill_pb)

    for input_image_pathstr in tqdm(input_image_list):
        image_name = Path(input_image_pathstr).name
        input_image = cv2.imread(input_image_pathstr)
        mask_image_path = input_mask_dir_pathlib.joinpath(image_name.replace(".jpg", ".png"))
        mask_image = cv2.imread(str(mask_image_path), cv2.IMREAD_ANYDEPTH)
        
        if not mask_image_path.exists():
            continue
        
        result = remover(input_image, mask_image)

        output_image_path = output_dir_pathlib.joinpath(image_name)
        cv2.imwrite(str(output_image_path), result)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()
