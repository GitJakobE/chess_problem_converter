import math
import cv2 as cv
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from util import Util

source_image = "data\Toft-00-0252.png" if len(sys.argv) < 2 else sys.argv[1]
debug = True


# Main function
def main():
    # import a pdf and convert its content to pngs
    Util.convert_pdf_to_pngs("D:\Skakproblemopgaver\\frasoren\chess\chess\data\data\Toft-00")

    # import, convert to grayscale and binarize
    image = cv.imread(source_image)
    blue_channel = image[:, :, 2]
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    if debug == True:
        cv.imwrite('blue_image.png', blue_channel)
        cv.imwrite('red_image.png', red_channel)
        cv.imwrite('green_image.png', green_channel)

    blue_img = np.zeros(image.shape)
    blue_img[:, :, 0] = blue_channel
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    retval, image_gray = cv.threshold(blue_channel, 128, 256, cv.THRESH_BINARY)
    if (debug == True):
        cv.imwrite('grayimage.png', image_gray)
    # fig, ax = plt.subplots()
    # im = ax.imshow(image)
    # # plt.show()
    cv.imshow('image', blue_img)
    # Determine lines using hough transform
    horizontal_lines, vertical_lines = Util.find_horizontal_vertical_lines(image_gray)
    # debug: draw_lines(image_gray, horizontal_lines);

    # Determine angle of lines and use to derotate image
    angle = Util.find_average_rotation(horizontal_lines, vertical_lines)
    angle_deg = 180.0 * angle / math.pi
    de_rotated_image_gray = Util.rotate_image(image_gray, angle_deg)

    # Locate the positions of the 'templates'
    template_locations = Util.locate_templates(de_rotated_image_gray)

    # Locate lines again on derotated image to find bounding box
    horizontal_lines, vertical_lines = Util.find_horizontal_vertical_lines(de_rotated_image_gray)
    bounding_box = Util.find_bounding_box(horizontal_lines, vertical_lines)

    bb_image_gray = image
    cv.rectangle(bb_image_gray, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0, 0, 255), 2)
    # cv.imshow('image', bb_image_gray)
    cv.imwrite('output.png', de_rotated_image_gray)
    fig, ax = plt.subplots()
    im = ax.imshow(bb_image_gray)
    plt.show()
    # Setup the 8x8 grid and find out which cells are occupied by which templates
    grid = Util.setup_grid()
    grid = Util.fill_templates_in_grid(grid, bounding_box, template_locations)
    # Output grid as JSON
    print(json.dumps(grid))

    # Draw output
    Util.draw_locations(de_rotated_image_gray, template_locations)
    cv.imwrite('output.png', de_rotated_image_gray)


# Run the program
main()
