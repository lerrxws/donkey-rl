import csv
import os
from typing import Any


class CSVLogger:
    def __init__(
        self,
        path: str,
        fieldnames: list[str],
    ):
        self.path = path
        self.fieldnames = fieldnames

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self.__file = open(
            path,
            mode="w",
            newline="",
            encoding="utf-8",
        )

        self.__writer = csv.DictWriter(
            self.__file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        self.__writer.writeheader()
        self.__file.flush()

    def write(self, row: dict[str, Any]) -> None:
        clean_row = {}

        for field in self.fieldnames:
            value = row.get(field, "")

            if value is None:
                value = ""

            clean_row[field] = value

        self.__writer.writerow(clean_row)
        self.__file.flush()

    def close(self) -> None:
        self.__file.close()