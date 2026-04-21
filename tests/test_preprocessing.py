"""Unit tests for preprocessing functions."""

import random
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from mm_cxr_diag.data.preprocessing import (
    convert_agestr_to_years,
    create_working_tabular_df,
    generate_image_labels,
    randomize_df,
    set_seed,
    train_test_split,
)


class TestPreprocessing(unittest.TestCase):
    """Unit tests for the preprocessing functions."""

    def test_single_labels(self):
        """Test detection of single labels."""
        conditions = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Effusion",
            "Emphysema",
            "Fibrosis",
            "Hernia",
            "Infiltration",
            "Mass",
            "No Finding",
            "Nodule",
            "Pleural_Thickening",
            "Pneumonia",
            "Pneumothorax",
        ]

        for idx, condition in enumerate(conditions):
            labels = generate_image_labels(condition)
            self.assertIsInstance(labels, torch.Tensor)
            self.assertEqual(labels.shape, (15,))
            self.assertEqual(labels.dtype, torch.float32)
            self.assertEqual(labels[idx], 1)
            zero_positions = list(range(15))
            zero_positions.remove(idx)
            for pos in zero_positions:
                self.assertEqual(labels[pos], 0)

    def test_multiple_labels(self):
        """Test detection of multiple conditions."""
        input_str = "Atelectasis|Edema|Mass"
        labels = generate_image_labels(input_str)

        self.assertEqual(labels[0], 1)  # Atelectasis
        self.assertEqual(labels[3], 1)  # Edema
        self.assertEqual(labels[9], 1)  # Mass
        self.assertEqual(sum(labels), 3)  # Only three positions should be 1

    def test_case_insensitivity(self):
        """Test that the function is case insensitive."""
        variations = ["ATELECTASIS", "atelectasis", "Atelectasis", "aTeLeCtAsIs"]

        for variant in variations:
            labels = generate_image_labels(variant)
            self.assertEqual(labels[0], 1)
            self.assertEqual(sum(labels), 1)

    def test_no_finding(self):
        """Test the 'No Finding' case."""
        labels = generate_image_labels("No Finding")
        self.assertEqual(labels[10], 1)
        self.assertEqual(sum(labels), 1)

    def test_empty_string(self):
        """Test empty string input."""
        with self.assertRaises(ValueError):
            generate_image_labels("")

    def test_invalid_label(self):
        """Test with invalid/non-existent condition."""
        with self.assertRaises(ValueError):
            generate_image_labels("NonExistentCondition")

    def test_mixed_valid_invalid(self):
        """Test mixture of valid and invalid labels."""
        with self.assertRaises(ValueError):
            generate_image_labels("Atelectasis|NonExistentCondition")

    def test_all_found_labels(self):
        """Test all labels are found."""
        input_str = "|".join(
            [
                "Atelectasis",
                "Cardiomegaly",
                "Consolidation",
                "Edema",
                "Effusion",
                "Emphysema",
                "Fibrosis",
                "Hernia",
                "Infiltration",
                "Mass",
                "Nodule",
                "Pleural_Thickening",
                "Pneumonia",
                "Pneumothorax",
            ]
        )
        labels = generate_image_labels(input_str)
        self.assertEqual(sum(labels), 14)  # All labels except "No Finding" should be 1
        self.assertEqual(labels[10], 0)

    def test_shape(self):
        """Test the shape of the output tensor."""
        input_str = "Atelectasis|Edema|Mass"
        labels = generate_image_labels(input_str)
        self.assertEqual(labels.shape, (15,))
        self.assertEqual(labels.dtype, torch.float32)


class TestAgeConversion(unittest.TestCase):
    """Unit tests for the age string conversion function."""

    def test_years(self):
        """Test conversion of year-based age strings."""
        self.assertEqual(convert_agestr_to_years("045y"), 45.0)
        self.assertEqual(convert_agestr_to_years("001y"), 1.0)
        self.assertEqual(convert_agestr_to_years("000y"), 0.0)

    def test_months(self):
        """Test conversion of month-based age strings."""
        self.assertEqual(convert_agestr_to_years("012m"), 1.0)
        self.assertEqual(convert_agestr_to_years("006m"), 0.5)
        self.assertEqual(convert_agestr_to_years("024m"), 2.0)

    def test_days(self):
        """Test conversion of day-based age strings."""
        self.assertAlmostEqual(convert_agestr_to_years("365d"), 1.0, places=6)
        self.assertAlmostEqual(convert_agestr_to_years("180d"), 0.493151, places=6)
        self.assertAlmostEqual(convert_agestr_to_years("030d"), 0.082192, places=6)

    def test_weeks(self):
        """Test conversion of week-based age strings."""
        self.assertAlmostEqual(convert_agestr_to_years("052w"), 1.0, places=6)
        self.assertAlmostEqual(convert_agestr_to_years("026w"), 0.5, places=6)
        self.assertAlmostEqual(convert_agestr_to_years("013w"), 0.25, places=6)

    def test_case_sensitivity(self):
        """Test that the function handles different cases properly."""
        self.assertEqual(convert_agestr_to_years("045Y"), 45.0)
        self.assertEqual(convert_agestr_to_years("012M"), 1.0)
        self.assertAlmostEqual(convert_agestr_to_years("365D"), 1.0, places=6)
        self.assertEqual(convert_agestr_to_years("052W"), 1.0)

    def test_empty_string(self):
        """Test that empty strings raise ValueError."""
        with self.assertRaises(ValueError):
            convert_agestr_to_years("")
        with self.assertRaises(ValueError):
            convert_agestr_to_years("    ")

    def test_invalid_length(self):
        """Test that strings of invalid length raise ValueError."""
        with self.assertRaises(ValueError):
            convert_agestr_to_years("1y")
        with self.assertRaises(ValueError):
            convert_agestr_to_years("45yr")
        with self.assertRaises(ValueError):
            convert_agestr_to_years("1000y")

    def test_invalid_format(self):
        """Test that strings with invalid format raise ValueError."""
        with self.assertRaises(ValueError):
            convert_agestr_to_years("045x")
        with self.assertRaises(ValueError):
            convert_agestr_to_years("abcy")
        with self.assertRaises(ValueError):
            convert_agestr_to_years("45.y")
        with self.assertRaises(ValueError):
            convert_agestr_to_years("-45y")


class TestTabularDataPreprocessing(unittest.TestCase):
    """Unit tests for the tabular data preprocessing function."""

    def setUp(self):
        """Set up test data."""
        self.test_df = pd.DataFrame(
            {
                "Image Index": ["00000001_000.png", "00000002_000.png"],
                "Finding Labels": ["Cardiomegaly", "No Finding"],
                "Follow-up #": [0, 1],
                "Patient Age": ["058Y", "012M"],
                "Patient Gender": ["M", "F"],
                "View Position": ["PA", "AP"],
            }
        )
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_column_renaming(self):
        """Test that columns are correctly renamed."""
        result_df = create_working_tabular_df(self.test_df)
        expected_columns = {
            "imageIndex",
            "followUpNumber",
            "patientAge",
            "patientGender",
            "viewPosition",
        }
        self.assertTrue(expected_columns.issubset(set(result_df.columns)))

    def test_age_conversion(self):
        """Test age string conversion."""
        result_df = create_working_tabular_df(self.test_df)
        self.assertEqual(result_df["patientAge"].iloc[0], 58.0)  # 058Y
        self.assertEqual(result_df["patientAge"].iloc[1], 1.0)  # 012M

    def test_gender_encoding(self):
        """Test gender binary encoding."""
        result_df = create_working_tabular_df(self.test_df)
        self.assertEqual(result_df["patientGender"].iloc[0], 0)  # M
        self.assertEqual(result_df["patientGender"].iloc[1], 1)  # F

    def test_view_position_encoding(self):
        """Test view position binary encoding."""
        result_df = create_working_tabular_df(self.test_df)
        self.assertEqual(result_df["viewPosition"].iloc[0], 0)  # PA
        self.assertEqual(result_df["viewPosition"].iloc[1], 1)  # AP

    def test_label_encoding(self):
        """Test one-hot encoding of finding labels."""
        result_df = create_working_tabular_df(self.test_df)

        self.assertEqual(result_df["label_cardiomegaly"].iloc[0], 1)
        self.assertEqual(result_df["label_no_finding"].iloc[0], 0)

        self.assertEqual(result_df["label_cardiomegaly"].iloc[1], 0)
        self.assertEqual(result_df["label_no_finding"].iloc[1], 1)

    def test_case_insensitivity(self):
        """Test case insensitive handling of categorical variables."""
        test_df = self.test_df.copy()
        test_df["Patient Gender"] = ["M", "f"]
        test_df["View Position"] = ["pA", "Ap"]

        result_df = create_working_tabular_df(test_df)
        self.assertEqual(result_df["patientGender"].iloc[0], 0)  # m
        self.assertEqual(result_df["patientGender"].iloc[1], 1)  # f
        self.assertEqual(result_df["viewPosition"].iloc[0], 0)  # pa
        self.assertEqual(result_df["viewPosition"].iloc[1], 1)  # ap


class TestRandomization(unittest.TestCase):
    """Unit tests for the DataFrame randomization function."""

    def setUp(self):
        """Set up test data."""
        self.test_df = pd.DataFrame(
            {"A": range(10), "B": range(10, 20), "C": list("abcdefghij")}
        )

    def test_row_count_preservation(self):
        """Test that the number of rows remains the same after randomization."""
        result_df = randomize_df(self.test_df)
        self.assertEqual(len(result_df), len(self.test_df))

    def test_column_preservation(self):
        """Test that all columns are preserved after randomization."""
        result_df = randomize_df(self.test_df)
        self.assertEqual(list(result_df.columns), list(self.test_df.columns))

    def test_content_preservation(self):
        """Test that all values are preserved after randomization."""
        result_df = randomize_df(self.test_df)
        for col in self.test_df.columns:
            self.assertEqual(
                sorted(result_df[col].tolist()), sorted(self.test_df[col].tolist())
            )

    def test_different_order(self):
        """Test that the order is actually randomized."""
        set_seed(42)
        result_df = randomize_df(self.test_df)
        # Check if at least one row is in a different position
        any_different = False
        for i in range(len(self.test_df)):
            if not (result_df.iloc[i] == self.test_df.iloc[i]).all():
                any_different = True
                break
        self.assertTrue(any_different)

    def test_index_reset(self):
        """Test that the index is reset after randomization."""
        result_df = randomize_df(self.test_df)
        self.assertEqual(list(result_df.index), list(range(len(self.test_df))))


class TestSeedSetting(unittest.TestCase):
    """Unit tests for the seed setting function."""

    def test_reproducible_torch_random(self):
        """Test that torch.random produces same numbers with same seed."""
        set_seed(42)
        random1 = torch.rand(5)
        set_seed(42)
        random2 = torch.rand(5)
        self.assertTrue(torch.equal(random1, random2))

    def test_reproducible_numpy_random(self):
        """Test that numpy.random produces same numbers with same seed."""
        set_seed(42)
        random1 = np.random.rand(5)
        set_seed(42)
        random2 = np.random.rand(5)
        np.testing.assert_array_equal(random1, random2)

    def test_reproducible_python_random(self):
        """Test that random produces same numbers with same seed."""
        set_seed(42)
        random1 = [random.random() for _ in range(5)]
        set_seed(42)
        random2 = [random.random() for _ in range(5)]
        self.assertEqual(random1, random2)

    def test_different_seeds(self):
        """Test that different seeds produce different random numbers."""
        set_seed(42)
        random1 = torch.rand(5)
        set_seed(43)
        random2 = torch.rand(5)
        self.assertFalse(torch.equal(random1, random2))


class TestTrainTestSplit(unittest.TestCase):
    """Unit tests for the train test split function."""

    def setUp(self):
        """Set up test data."""
        self.test_df = pd.DataFrame(
            {"A": range(100), "B": range(100, 200), "C": list("abcdefghij" * 10)}
        )

    def test_split_proportions(self):
        """Test that the split proportions are correct."""
        test_sizes = [0.2, 0.3, 0.5]
        for test_size in test_sizes:
            train_df, test_df = train_test_split(self.test_df, test_size=test_size)
            expected_test_size = int(len(self.test_df) * test_size)
            self.assertEqual(len(test_df), expected_test_size)
            self.assertEqual(len(train_df), len(self.test_df) - expected_test_size)

    def test_deterministic_with_seed(self):
        """Test that the split is deterministic with the same seed."""
        train1, test1 = train_test_split(self.test_df, test_size=0.2, seed=42)
        train2, test2 = train_test_split(self.test_df, test_size=0.2, seed=42)
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_different_seeds(self):
        """Test that different seeds produce different splits."""
        train1, test1 = train_test_split(self.test_df, test_size=0.2, seed=42)
        train2, test2 = train_test_split(self.test_df, test_size=0.2, seed=43)
        self.assertFalse(train1.equals(train2))
        self.assertFalse(test1.equals(test2))

    def test_row_integrity(self):
        """Test that rows are preserved and not modified after splitting."""
        train_df, test_df = train_test_split(self.test_df)
        combined_df = pd.concat([train_df, test_df])
        combined_df = combined_df.sort_index()
        self.test_df = self.test_df.sort_index()
        pd.testing.assert_frame_equal(combined_df, self.test_df)

    def test_mutually_exclusive_splits(self):
        """Test that train and test sets have no overlapping rows."""
        train_df, test_df = train_test_split(self.test_df)
        train_indices = set(train_df.index)
        test_indices = set(test_df.index)
        self.assertEqual(len(train_indices.intersection(test_indices)), 0)

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        train_df, test_df = train_test_split(empty_df)
        self.assertTrue(train_df.empty)
        self.assertTrue(test_df.empty)

    def test_invalid_test_size(self):
        """Test that invalid test_size raises ValueError."""
        invalid_sizes = [-0.1, 0, 1.0, 1.1]
        for size in invalid_sizes:
            with self.assertRaises(ValueError):
                train_test_split(self.test_df, test_size=size)


if __name__ == "__main__":
    unittest.main()
