from argparse import Namespace, ArgumentParser, ArgumentDefaultsHelpFormatter
import cv2 as cv
import sys

from loguru import logger
from config import BoardConfig, CLIArgs
import os

from util import BoardCalculations

# source_image = "data\Toft-00-0174.png" if len(sys.argv) < 2 else sys.argv[1]


# Main function
def main(args: CLIArgs):
    conf = BoardConfig()
    conf.read_predict_dict()
    logger.add("out/log.log")

    source_images = [filename.path for filename in os.scandir("data/") if filename.path.endswith(".png")]
    if args.input_file is not None:
        source_images = [filename.path for filename in os.scandir("data/") if filename.path.endswith(".png")]

    for source_image in source_images:
        conf.set_export(source_image)
        conf.set_source_image(source_image)

        image = cv.imread(source_image)

        logger.info(f"{source_image}: Printing board")

        try:
            board_image = BoardCalculations.find_board(image=image, conf=conf)
            logger.info(f"printing board and pieces for {source_image}")
            cv.imwrite(f'{conf.export.output_str}_board.png', board_image)

        except IndexError:
            logger.error(f"{conf.source_image}: board out of scope:")
            continue
        except Exception as e:
            logger.error(f"{conf.source_image} error in finding board: {e}")

        # models = [model for model in os.scandir("models/") if model.name.endswith(".pkl")]
        # for model in models:

        # with open("models/Nearest Neighbors.pkl", 'rb') as f:  # open a text file
        #     clf = pickle.load(f)
        # Util.print_board(board=board, conf=conf, classifier=clf)


def parse_arguments():
    parser = ArgumentParser(
        description="A converter of chess problems from images files to csv",
        formatter_class=ArgumentDefaultsHelpFormatter,
        prog="chess problem converter",
    )

    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        help="The input filename of the image",
        required=False,
    )

    return CLIArgs.from_namespace(parser.parse_args())

# Run the program
if __name__ == '__main__':
    main(parse_arguments())
