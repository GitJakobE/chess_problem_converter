from typing import Optional
from argparse import Namespace
from dataclasses import dataclass

from dacite import from_dict

@dataclass
class CLIArgs:
    input_file: Optional[str]

    @staticmethod
    def from_namespace(name_space: Namespace) -> "CLIArgs":
        return from_dict(CLIArgs, name_space.__dict__)