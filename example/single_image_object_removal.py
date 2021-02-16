import cv2
import click
from pathlib import Path
from object_removal.remover import ObjectRemover


SCRIPT_DIR = str(Path(__file__).parent)

@click.command()
@click.option("--input-image-path", "-i", default="{}/../data/color.jpg".format(SCRIPT_DIR))
@click.option("--input-mask-path", "-m", default="{}/../data/mask.png".format(SCRIPT_DIR))
@click.option("--output_image_path", "-o", default="{}/../out.png".format(SCRIPT_DIR))
@click.option("--hifill-pb", "-pb", default="{}/../pb/hifill.pb".format(SCRIPT_DIR))
def main(input_image_path, input_mask_path, output_image_path, hifill_pb):
    remover = ObjectRemover(hifill_pb)
    input_image = cv2.imread(input_image_path)
    mask_image = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)

    result = remover(input_image, mask_image)
    cv2.imwrite("result.png", result)
    cv2.waitKey(10)


if __name__ == "__main__":
    main()