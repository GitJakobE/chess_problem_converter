from dataclasses import dataclass
import pickle


@dataclass
class Export:
    output_dir: str = "./out"
    output_with_tag: str = ""


@dataclass
class BoardConfig:
    export = Export()
    line_profile_width = 450
    gaussian_sigma = 7
    approx_board_width = 530
    approx_board_height = 530
    predict_dict = {}
    only_pieces: bool = True
    source_image = ""
    model = ""

    def set_source_image(self, filename: str):
        self.source_image = filename

    def set_export(self, filename: str):
        filename = filename.split('/')[-1].split('\\')[-1]
        self.export.output_str = f"./out/{filename.split('.')[0]}"

    def read_predict_dict(self, filename: str = "piece_lib_dict.pkl"):
        with open(filename, 'rb') as f:  # open a text file
            self.predict_dict = pickle.load(f)
