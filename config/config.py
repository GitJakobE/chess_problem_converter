from dataclasses import dataclass
from pathlib import Path

@dataclass
class Export:
    output_dir: str = "./out"
    output_with_tag: str = ""
    output_str: str = "./out"


@dataclass
class BoardConfig:
    export: Export = Export()
    line_profile_width = 300
    gaussian_sigma = 7
    approx_board = 510

    def set_export(self, filename: str):
        filename = filename.split('/')[-1].split('\\')[-1]
        self.export.output_str = f"./out/{filename.split('.')[0]}"
