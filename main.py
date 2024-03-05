import pickle
import os

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import cv2 as cv
from loguru import logger

from config import BoardConfig, CLIArgs
from util import BoardCalculations, Util, constants
from model.pytorch_model import TouchDataModel


def main(args: CLIArgs):
    conf = BoardConfig()
    conf.read_predict_dict()
    logger.add("out/log.log")

    if args.convert_pdfs is not None:
        if not os.path.exists(args.convert_pdfs):
            logger.error(f"path for converting pdfs to pngs does not exsist: {args.convert_pdfs}")
            raise IndexError(f"path does not exist {args.convert_pdfs}")
        Util.convert_pdf_to_pngs(args.convert_pdfs)

    source_images = [filename.path for filename in os.scandir("data/") if filename.path.endswith(".png")]
    if args.input_file is not None:
        source_images = [filename.path for filename in os.scandir("data/") if filename.path.endswith(".png")]

    if args.verify:


    if args.model is not None:
        conf.model = args.model
        run_images(source_images=source_images, conf=conf)

    if args.train_model:
        dm = TouchDataModel()
        dm.init_torch_model()
        dm.train_torch_model(training_lib="trainingset")
        dm.save_model()


def run_images(source_images: list[str], conf: BoardConfig):
    try:
        with open(f"models/{conf.model}.pkl", 'rb') as f:  # open a text file
            clf = pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading the model: {e}")
        raise FileNotFoundError

    for source_image in source_images:
        conf.set_export(source_image)
        conf.set_source_image(source_image)
        conf.init_torch_model()

        image = cv.imread(source_image)
        if constants.standard_image_dim != image.shape[:-1]:
            logger.info(f"{source_image}: Resizing the image constants.standard_image_dim")
            image = cv.resize(image, constants.standard_image_dim, interpolation=cv.INTER_LINEAR)

        try:
            logger.info(f"{source_image}: Finding the board: ")
            board = BoardCalculations.find_board(image=image, conf=conf)
            cv.imwrite(f'{conf.export.output_str}_board.png', board.board_image)

            logger.info(f"{source_image}: Finding the pieces")
            Util.find_pieces(board=board, conf=conf)
            board.to_file(filename="out/setups.txt", board_name=conf.export.output_str)
        except IndexError:
            logger.error(f"{conf.source_image}: board out of scope:")
            continue
        except Exception as e:
            logger.error(f"{conf.source_image} error in finding board: {e}")


def verify(path:str, conf: BoardConfig):
    for
    try:
        with open(f"models/{conf.model}.pkl", 'rb') as f:  # open a text file
            clf = pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading the model: {e}")
        raise FileNotFoundError

    for source_image in source_images:
        conf.set_export(source_image)
        conf.set_source_image(source_image)
        conf.init_torch_model()

        image = cv.imread(source_image)




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

    parser.add_argument(
        "-t",
        "--train-model",
        help="this will train the model using the png files in the trainingset folder",
        action="store_true",
        required=False,
    )

    parser.add_argument(
        "-m",
        "--model",
        help="this will train the model using the png files in the trainingset folder",
        type=str,
        required=False,
    )

    parser.add_argument(
        "-c",
        "--convert-pdfs",
        type=str,
        help="used to convert the pdf files in a directory to png",
        required=False
    )

    parser.add_argument(
        "-v",
        "--verify",
        type=str,
        help="verify the output folders",
        required=False
    )

    return CLIArgs.from_namespace(parser.parse_args())


# Run the program
if __name__ == '__main__':
    main(parse_arguments())
