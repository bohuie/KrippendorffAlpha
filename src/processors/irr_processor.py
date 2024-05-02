from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame


@dataclass
class RaterDataRow:
    labels: str
    data: str


@dataclass
class RaterData:
    rows: List[RaterDataRow]


@dataclass
class RaterFolderMetadata:
    folder_path: Path
    rater_name: str


@dataclass
class RawDataFileConfig:
    data_column_name: str = "Data"
    labels_column_name: str = "Label"
    context_column_name: Optional[str] = None


@dataclass
class ProcessorConfig:
    rater_1_label_column_name: str = "label_1"
    rater_2_label_column_name: str = "label_2"
    data_column_name: str = "data"


class IRRProcessor:
    def __init__(
        self, available_labels: List[str], config: ProcessorConfig = ProcessorConfig()
    ):
        self.available_labels = available_labels
        self.config = config

    @staticmethod
    def _process_labels(labels: str) -> List[str]:
        return [_label.strip() for _label in labels.lower().split(", ")]

    def _get_agreement_table(
        self, rater1_data: RaterData, rater2_data: RaterData
    ) -> DataFrame:
        """
        Merge 2 rater data into 1 agreement table.

        This will one-hot encode the labels for each rater and merge them into a single table.

        Parameters
        ----------
        rater1_data : RaterData
            The first rater's data.
        rater2_data : RaterData
            The second rater's data.

        Returns
        -------
        agreement_table : pd.DataFrame
            The agreement table that contains the data and the one-hot encoded labels for each rater, the cells is the sum of the labels for each rater.

        """
        label_1 = self.config.rater_1_label_column_name
        label_2 = self.config.rater_2_label_column_name

        hash_map: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
        for rater_data in rater1_data.rows:
            cleaned_data = rater_data.data.strip()
            hash_map[cleaned_data][label_1] = self._process_labels(rater_data.labels)
            hash_map[cleaned_data][label_2] = []
        for rater_data in rater2_data.rows:
            cleaned_data = rater_data.data.strip()
            if cleaned_data not in hash_map:
                hash_map[cleaned_data] = {}
                hash_map[cleaned_data][label_1] = []
            hash_map[cleaned_data][label_2] = self._process_labels(
                rater_data.labels
            )

        for data, users_labels in hash_map.items():
            rater_1_labels = sorted(users_labels.get(label_1, []))
            rater_2_labels = sorted(users_labels.get(label_2, []))
            # compare the labels to see if they are strictly equal
            if rater_1_labels != rater_2_labels:
                print(f'{data}\nRater 1: {", ".join(rater_1_labels)}\nRater 2: {", ".join(rater_2_labels)}', end="\n----------------------------\n")

        df = DataFrame(
            {}, index=np.arange(len(hash_map)), columns=self.available_labels
        )
        df = df.replace(np.nan, 0)
        df.insert(
            loc=0, column=self.config.data_column_name, value=pd.Series(hash_map.keys())
        )

        df["num_rater"] = 0

        for data, users_labels in hash_map.items():
            row_idx = df[df[self.config.data_column_name] == data].index[0]
            if len(users_labels[label_1]) == 0 or len(users_labels[label_2]) == 0:
                print(data, end='\n--------MISSING DATA--------\n')

            if len(users_labels[label_1]) > 0:
                df.at[row_idx, "num_rater"] += 1
            if len(users_labels[label_2]) > 0:
                df.at[row_idx, "num_rater"] += 1

            for label in users_labels[label_1]:
                df.at[row_idx, label] += 1
            for label in users_labels[label_2]:
                df.at[row_idx, label] += 1

        return df

    @staticmethod
    def extract_rater_data(
        rater_data_folder_path: Path,
        file_config: RawDataFileConfig = RawDataFileConfig(),
    ) -> RaterData:
        rater_data_rows: List[RaterDataRow] = []
        for csv_file in rater_data_folder_path.rglob("*.csv"):
            df = pd.read_csv(csv_file)
            for row_idx, row_data in df.iterrows():
                data = row_data[file_config.data_column_name]
                labels = row_data[file_config.labels_column_name]
                if not pd.isna(labels) and not pd.isna(data):
                    rater_data_rows.append(RaterDataRow(labels=labels, data=data))

        return RaterData(rows=rater_data_rows)

    def calculate_irr(self, rater1_data: RaterData, rater2_data: RaterData) -> float:
        agreement_table = self._get_agreement_table(rater1_data, rater2_data)

        n = len(agreement_table)

        r_bar = sum([agreement_table.at[i, "num_rater"] for i in range(len(agreement_table))]) / n
        # drop num_rater after calculating r_bar
        agreement_table.drop(columns=["num_rater"], inplace=True)

        p_primea = 0
        total_num_ratings = 0
        for i in range(len(agreement_table)):
            r_i = agreement_table.loc[i, self.available_labels].sum()
            total_num_ratings += r_i
            for label in agreement_table.columns:
                if label == self.config.data_column_name:
                    continue
                r_ik = agreement_table[label].loc[agreement_table.index[i]]
                # TODO: Expand rbar_ik to factor in w_kl for more than just "nominal" weight function
                rbar_ik = agreement_table[label].loc[agreement_table.index[i]]

                p_aik = (r_ik * (rbar_ik - 1)) / (r_bar * (r_i - 1))
                p_primea += p_aik
        p_primea /= n
        p_a = (p_primea * (1 - 1 / (n * r_bar))) + (1 / (n * r_bar))

        # PI_ks: an array of PI_k, the percentage of ratings that fell into category k
        PI_ks = []
        for label in agreement_table.columns:
            if label == self.config.data_column_name:
                continue
            PI_k = agreement_table[label].sum() / total_num_ratings
            PI_ks.append(PI_k)

        # NB: Since we use nominal weights, w_kl is 0 unless k=l, so p_e is the sum of (PI_k)^2
        p_e = sum(PI_k * PI_k for PI_k in PI_ks)

        # Calculate krip_alpha
        krip_alpha = (p_a - p_e) / (1 - p_e)

        self.krip_alpha = krip_alpha
        return krip_alpha

    @staticmethod
    def process_all_labels_file(all_labels_path: Path) -> List[str]:
        with open(all_labels_path, "r") as f:
            return [_label.lower().strip() for _label in f.read().split(",")]


if __name__ == "__main__":
    # Assuming the project handles only 2 raters
    project_root = Path(__file__).parent.parent.parent
    rater1_folder = project_root / "data" / "rater1"
    rater2_folder = project_root / "data" / "rater2"
    all_labels_path = project_root / "data" / "labels.txt"

    _rater1_data = IRRProcessor.extract_rater_data(rater1_folder)
    _rater2_data = IRRProcessor.extract_rater_data(rater2_folder)
    _available_labels = IRRProcessor.process_all_labels_file(all_labels_path)

    irr_processor = IRRProcessor(_available_labels)
    print(irr_processor.calculate_irr(_rater1_data, _rater2_data))
