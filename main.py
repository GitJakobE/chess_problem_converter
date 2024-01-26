import cv2 as cv
import sys
from util import Util
from scipy.ndimage import rotate

source_image = "data\Toft-00-0252.png" if len(sys.argv) < 2 else sys.argv[1]


# Main function
def main():
    # import, convert to grayscale and binarize
    line_profile_width = 300
    gaussian_sigma = 7
    approx_board = 500
    image = cv.imread(source_image)

    cv.imwrite('out/image_org.png', image)

    right_tilt_angle, left_tilt_angle = Util.find_edge(image, line_profile_width, gaussian_sigma, 10)
    rotated_image = rotate(image, (right_tilt_angle + left_tilt_angle) / 2, reshape=False)

    left_edge, right_edge = Util.find_board_edges(rotated_image, line_profile_width, gaussian_sigma, approx_board)
    image = Util.remove_black_writing(image)
    cv.imwrite(f'out/full_rotated.png', image)

    board_width = right_edge - left_edge
    tile_width = board_width // 8
    top_line = Util.find_top_line(image, left_edge, right_edge, tile_width)

    board = rotated_image[top_line:top_line + board_width, left_edge:right_edge]

    Util.print_board(board)

    Util.print_pieces(board)

# Run the program
main()
