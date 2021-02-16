import cv2
import click
import numpy as np


def main():
    rgb = cv2.imread("../data/rgb.jpg")

    bgrLower = np.array([10, 10, 80])
    bgrUpper = np.array([100, 100, 255])
    img_mask = cv2.inRange(rgb, bgrLower, bgrUpper)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, (15, 15))
    img_mask[:100, :] = 0
    img_mask = cv2.dilate(img_mask, (15, 15))
    img_mask = cv2.bitwise_not(img_mask)
    cv2.imwrite("../data/mask.png", img_mask)
    cv2.waitKey(10)


if __name__ == "__main__":
    main()