from unittest import TestCase
from src.processors.irr_processor import IRRProcessor, RawDataFileConfig

from pathlib import Path


class TestIRRProcessor(TestCase):
    """
    Test methods for `src/processors/irr_processor.py`.
    """

    def setUp(self) -> None:
        """
        Read in `labels.txt` for the labels used by C & A on first pass.

        """
        prj_root = Path(__file__).parent.parent.parent

        # The following filepath is where the REAL data is being stored
        self.filepath = prj_root / "data/csv/irr_first_pass/"
        self.folder_1 = prj_root / "data/csv/irr_first_pass/rater1/"
        self.folder_2 = prj_root / "data/csv/irr_first_pass/rater2/"

        # Fake data
        self.fake_filepath = prj_root / "tests/processors/irr_test_data/"
        self.fake_1 = prj_root / "tests/processors/irr_test_data/rater1/"
        self.fake_2 = prj_root / "tests/processors/irr_test_data/rater2/"

    def test_calculate_irr_real(self) -> None:
        _rater1_data = IRRProcessor.extract_rater_data(self.folder_1)
        _rater2_data = IRRProcessor.extract_rater_data(
            self.folder_2, RawDataFileConfig(labels_column_name="Code")
        )
        all_labels = IRRProcessor.process_all_labels_file(self.filepath / "labels.txt")

        irr_processor = IRRProcessor(all_labels)
        result = irr_processor.calculate_irr(_rater1_data, _rater2_data)

        print("Krippendorf's alpha:\t", result, end="\n\n")

    def test_calculate_irr_fake(self) -> None:
        _rater1_data = IRRProcessor.extract_rater_data(
            self.fake_1,
            RawDataFileConfig(
                labels_column_name="code", data_column_name="comment_body"
            ),
        )
        _rater2_data = IRRProcessor.extract_rater_data(
            self.fake_2,
            RawDataFileConfig(
                labels_column_name="code", data_column_name="comment_body"
            ),
        )
        all_labels = IRRProcessor.process_all_labels_file(
            self.fake_filepath / "labels.txt"
        )
        irr_processor = IRRProcessor(all_labels)
        result = irr_processor.calculate_irr(_rater1_data, _rater2_data)

        # The number is calculated in this spreadsheet: https://docs.google.com/spreadsheets/d/1AmJmlj7NAarpVKaVjcQSZJM8u6tBx64gnucWx-CLjQM/edit?usp=sharing
        self.assertEqual(round(result, 4), 0.5359)

        # This is the case where we set rbar = 2
        # self.assertEqual(round(result, 4), 0.6739)
        print(result)
