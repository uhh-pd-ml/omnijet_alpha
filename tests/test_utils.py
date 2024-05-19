"""Tests for functions in gabbro/utils."""

import unittest

import awkward as ak
import numpy as np

from gabbro.utils.arrays import (
    ak_clip,
    ak_pad,
    ak_select_and_preprocess,
    ak_smear,
    ak_to_np_stack,
    count_appearances,
    np_to_ak,
    sort_by_pt,
)


class TestNpToAk(unittest.TestCase):
    def setUp(self):
        # np array of shape (2, 3, 2) (2 jets, 3 constituents, 2 features)
        self.np_array = np.array(
            [
                [[1, 2], [3, 3], [0, 0]],
                # also want to mask the "4", but checking if this is corrected by the function
                [[2, 2], [4, 0], [0, 0]],
            ]
        )
        self.np_mask = np.array(
            [
                [True, True, False],
                [True, False, False],
            ]
        )
        self.names = ["pt", "eta"]
        self.ak_arrary_expected = ak.Array(
            {
                "pt": [[1, 3], [2]],
                "eta": [[2, 3], [2]],
            }
        )
        self.ak_arrary_expected_without_mask = ak.Array(
            {
                "pt": [[1, 3, 0], [2, 4, 0]],
                "eta": [[2, 3, 0], [2, 0, 0]],
            }
        )

    def test_np_to_ak_with_mask(self):
        result = np_to_ak(self.np_array, mask=self.np_mask, names=self.names)
        for i, name in enumerate(self.names):
            self.assertTrue(ak.all(result[name] == self.ak_arrary_expected[name]))

    def test_np_to_ak_without_mask(self):
        result = np_to_ak(self.np_array, names=self.names)
        for i, name in enumerate(self.names):
            self.assertTrue(ak.all(result[name] == self.ak_arrary_expected_without_mask[name]))


class TestAkToNpStack(unittest.TestCase):
    def setUp(self):
        self.ak_array = ak.Array(
            {
                "pt": [[1, 2, 3], [2, 4]],
                "eta": [[0, 0, 0], [2, 2]],
                "phi": [[0, 0, 0], [3, 3]],
                "E": [[1, 1, 1], [4, 4]],
            }
        )
        # use as the arget array a version where the pt and eta are swapped
        # --> this check both that the order of the stacked fields is correct
        #     and that not all features have to be selected
        self.np_array_padded_len5_eta_pt = np.array(
            [
                [[0, 1], [0, 2], [0, 3], [0, 0], [0, 0]],
                [[2, 2], [2, 4], [0, 0], [0, 0], [0, 0]],
            ]
        )

    def test_ak_to_np_stack(self):
        input_data = ak_pad(self.ak_array, maxlen=5, axis=1)
        result = ak_to_np_stack(input_data, axis=2, names=["eta", "pt"])

        try:
            self.assertTrue(np.array_equal(result, self.np_array_padded_len5_eta_pt))
        except AssertionError:
            print("Arrays are not equal:")
            print("Expected:", self.np_array_padded_len5_eta_pt)
            print("Actual:", result)
            raise AssertionError


class TestSortByPt(unittest.TestCase):
    def test_error_raise(self):
        input_array_without_pt = ak.Array(
            {
                "eta": [[0, 0, 1], [2, 1]],
                "phi": [[0, 0, 1], [2, 1]],
            }
        )

        # check that AttributeError is raised
        with self.assertRaises(AttributeError):
            sort_by_pt(input_array_without_pt)

    def test_sorting_order(self):
        """Test the function sort_by_pt()"""

        input_array = ak.Array(
            {
                "pt": [[2, 1, 3], [2, 4]],
                "eta": [[0, 0, 1], [2, 1]],
            }
        )
        expected_sorted_array = ak.Array(
            {
                "pt": [[3, 2, 1], [4, 2]],
                "eta": [[1, 0, 0], [1, 2]],
            }
        )

        sorted_array = sort_by_pt(input_array)

        # compare array.pt and array.eta as lists
        self.assertEqual(sorted_array.pt.tolist(), expected_sorted_array.pt.tolist())
        self.assertEqual(sorted_array.eta.tolist(), expected_sorted_array.eta.tolist())


class TestAkSelectAndPreprocess(unittest.TestCase):
    def setUp(self):
        self.input_array = ak.Array(
            {
                "pt": [[2, 1], [2]],
                "eta": [[0, 1], [1]],
                "phi": [[0, 1], [2]],
            }
        )
        self.pp_dict = {
            "pt": {"subtract_by": 1, "multiply_by": 3, "func": "np.log", "inv_func": "np.exp"},
            "eta": {"subtract_by": 0, "multiply_by": 2},
            "phi": None,
        }
        self.expected_output = ak.Array(
            {
                "pt": [[(np.log(2) - 1) * 3, (np.log(1) - 1) * 3], [(np.log(2) - 1) * 3]],
                "eta": [[0 * 2, 1 * 2], [1 * 2]],
                "phi": [[0, 1], [2]],
            }
        )

    def test_ak_select_and_preprocess(self):
        result = ak_select_and_preprocess(self.input_array, pp_dict=self.pp_dict)
        for field in self.input_array.fields:
            self.assertEqual(result[field].tolist(), self.expected_output[field].tolist())

    def test_ak_select_and_preprocess_no_inv_func(self):
        """Test error raise if `func` is defined, but `inv_func` isn't."""

        pp_dict_wrong = {
            "pt": {"subtract_by": 1, "multiply_by": 3, "func": "np.log"},
        }
        with self.assertRaises(ValueError):
            ak_select_and_preprocess(self.input_array, pp_dict=pp_dict_wrong)

    def test_ak_select_and_preprocess_no_func(self):
        """Test error raise if `inv_func` is defined, but `func` isn't."""

        pp_dict_wrong = {
            "pt": {"subtract_by": 1, "multiply_by": 3, "inv_func": "np.exp"},
        }
        with self.assertRaises(ValueError):
            ak_select_and_preprocess(self.input_array, pp_dict=pp_dict_wrong)

    def test_inverse_sanity_check(self):
        """Test that applying the preprocessing once and then applying the inverse results in the
        same array again."""

        result = ak_select_and_preprocess(self.input_array, pp_dict=self.pp_dict)
        result = ak_select_and_preprocess(result, pp_dict=self.pp_dict, inverse=True)

        for field in self.pp_dict.keys():
            self.assertEqual(result[field].tolist(), self.input_array[field].tolist())

    def test_single_selection_cut(self):
        """Test that the function applies a single selection cut correctly."""

        arr = ak.Array(
            {
                "pt": [[1, 2, 3], [4, 5]],
                "eta": [[5, 6, 7], [8, 9]],
            }
        )
        pp_dict = {"pt": {"larger_than": 2}, "eta": None}
        arr_selected_expected = ak.Array(
            {
                "pt": [[3], [4, 5]],
                "eta": [[7], [8, 9]],
            }
        )
        arr_selected = ak_select_and_preprocess(arr, pp_dict)
        for field in pp_dict.keys():
            self.assertEqual(arr_selected[field].tolist(), arr_selected_expected[field].tolist())

    def test_multiple_selection_cuts(self):
        """Test that the function applies multiple selection cuts correctly."""
        arr = ak.Array(
            {
                "pt": [[1, 2, 3], [4, 5]],
                "eta": [[5, 6, 7], [8, 9]],
            }
        )
        pp_dict = {"pt": {"larger_than": 2}, "eta": {"smaller_than": 9}}
        arr_selected_expected = ak.Array(
            {
                "pt": [[3], [4]],
                "eta": [[7], [8]],
            }
        )
        arr_selected = ak_select_and_preprocess(arr, pp_dict)
        for field in pp_dict.keys():
            self.assertEqual(arr_selected[field].tolist(), arr_selected_expected[field].tolist())

    def test_selection_and_transform_combined(self):
        """Test that the function applies selection cuts and transforms the input array."""
        arr = ak.Array(
            {
                "pt": [[1, 2, 3], [4, 5]],
                "eta": [[5, 6, 7], [8, 9]],
            }
        )
        pp_dict = {
            "pt": {
                "larger_than": 2,
                "subtract_by": 1,
                "multiply_by": 3,
            },
            "eta": {"smaller_than": 9},
        }
        arr_selected_expected = ak.Array(
            {
                "pt": [[(3 - 1) * 3], [(4 - 1) * 3]],
                "eta": [[7], [8]],
            }
        )
        arr_selected = ak_select_and_preprocess(arr, pp_dict)
        for field in pp_dict.keys():
            self.assertEqual(arr_selected[field].tolist(), arr_selected_expected[field].tolist())


class TestAkSmearAndClip(unittest.TestCase):
    def setUp(self):
        self.input_array = ak.Array(
            {
                "pt": [[2, 1], [2]],
            }
        )

    def test_smear(self):
        """Test that the function smears the input array."""
        result = ak_smear(self.input_array["pt"], sigma=0.05, seed=101)
        expected_result = [
            [1.9604923750018493, 0.8982687259084063],
            [2.030165087346238],
        ]
        self.assertEqual(result.tolist(), expected_result)

    def test_clipmin(self):
        """Test that the function clips the input array to min value."""
        result = ak_clip(self.input_array["pt"], clip_min=1.5)
        expected_result = [
            [2, 1.5],
            [2],
        ]
        self.assertEqual(result.tolist(), expected_result)

    def test_clipmax(self):
        """Test that the function clips the input array to max value."""
        result = ak_clip(self.input_array["pt"], clip_max=1.5)
        expected_result = [
            [1.5, 1],
            [1.5],
        ]
        self.assertEqual(result.tolist(), expected_result)

    def test_clipminmax(self):
        """Test that the function clips the input array to min and max value."""
        result = ak_clip(self.input_array["pt"], clip_min=1.5, clip_max=1.8)
        expected_result = [
            [1.8, 1.5],
            [1.8],
        ]
        self.assertEqual(result.tolist(), expected_result)

    def test_smear_and_clip(self):
        """Test that the function smears and clips the input array."""
        result = ak_clip(
            ak_smear(
                self.input_array["pt"],
                sigma=0.05,
                seed=101,
            ),
            clip_min=0.9,
            clip_max=2.01,
        )
        expected_result = [
            [1.9604923750018493, 0.9],
            [2.01],
        ]
        self.assertEqual(result.tolist(), expected_result)


class TestTokenCounting(unittest.TestCase):
    def __init__(self):
        tokens_dev = np.array(
            [
                [1, 2, 4, 4, 0],
                [6, 6, 6, 6, 0],
            ]
        )
        mask_dev = np.array(
            [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
            ]
        )

        token_counts_expected = np.array(
            # each row corresponds to how often the token appears in the respective jet
            [
                [0, 1, 1, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 4],
            ]
        )
        n_token_appearance_expected = np.array(
            # each row corresponds to how many tokens are part of the group
            # of appearing i-times. E.g. here
            # in the first jet:
            # - 4 tokens appear 0 times
            # - 2 token appears 1 time,
            # - 1 token appears 2 times.
            # in the second jet:
            # - 6 tokens appear 0 times
            # - 1 tokens appear 4 times
            [
                [4, 2, 1, 0, 0],
                [6, 0, 0, 0, 1],
            ]
        )
        frac_token_appearance_expected = np.array(
            # each row corresponds to the fraction of tokens being part of the
            # group that appear i-times
            # I.e. here:
            # in the first jet:
            # - first one is a filler
            # - 50% of the tokens appear 1 time
            # - 50% of the tokens appear 2 times
            # in the second jet:
            # - first one is a filler
            # - 100% of the tokens appear 4 times
            [
                [
                    [0.0, 0.5, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            ]
        )

        token_counts, n_token_appearance, frac_token_appearance = count_appearances(
            tokens_dev,
            mask=mask_dev,
            count_up_to=4,
        )

        assert np.array_equal(token_counts, token_counts_expected)
        assert np.array_equal(n_token_appearance, n_token_appearance_expected)
        assert np.array_equal(frac_token_appearance, frac_token_appearance_expected)
