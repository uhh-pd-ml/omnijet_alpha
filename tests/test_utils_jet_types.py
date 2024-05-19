"""Tests for functions in gabbro/utils/jet_types."""

import unittest

from gabbro.utils.jet_types import (
    get_jet_type_from_file_prefix,
    get_numerical_label_from_file_prefix,
    get_tex_label_from_numerical_label,
)


class TestJetTypeHelperFunctions(unittest.TestCase):
    def test_get_numerical_from_fileprefix(self):
        self.assertEqual(get_numerical_label_from_file_prefix("ZJetsToNuNu_"), 0)
        self.assertEqual(get_numerical_label_from_file_prefix("HToBB_"), 1)
        self.assertEqual(get_numerical_label_from_file_prefix("HToCC_"), 2)
        self.assertEqual(get_numerical_label_from_file_prefix("HToGG_"), 3)
        self.assertEqual(get_numerical_label_from_file_prefix("HToWW4Q_"), 4)
        self.assertEqual(get_numerical_label_from_file_prefix("HToWW2Q1L_"), 5)
        self.assertEqual(get_numerical_label_from_file_prefix("ZToQQ_"), 6)
        self.assertEqual(get_numerical_label_from_file_prefix("WToQQ_"), 7)
        self.assertEqual(get_numerical_label_from_file_prefix("TTBar_"), 8)
        self.assertEqual(get_numerical_label_from_file_prefix("TTBarLep_"), 9)
        with self.assertRaises(ValueError):
            get_numerical_label_from_file_prefix("invalid_prefix")

    def test_get_tex_label_from_numerical_label(self):
        self.assertEqual(get_tex_label_from_numerical_label(0), "$q/g$")
        self.assertEqual(get_tex_label_from_numerical_label(1), "$H\\rightarrow b\\bar{b}$")
        self.assertEqual(get_tex_label_from_numerical_label(2), "$H\\rightarrow c\\bar{c}$")
        self.assertEqual(get_tex_label_from_numerical_label(3), "$H\\rightarrow gg$")
        self.assertEqual(get_tex_label_from_numerical_label(4), "$H\\rightarrow 4q$")
        self.assertEqual(get_tex_label_from_numerical_label(5), "$H\\rightarrow \\ell\\nu qq'$")
        self.assertEqual(get_tex_label_from_numerical_label(6), "$Z\\rightarrow q\\bar{q}$")
        self.assertEqual(get_tex_label_from_numerical_label(7), "$W\\rightarrow qq'$")
        self.assertEqual(get_tex_label_from_numerical_label(8), "$t\\rightarrow bqq'$")
        self.assertEqual(get_tex_label_from_numerical_label(9), "$t\\rightarrow b\\ell\\nu$")
        with self.assertRaises(ValueError):
            get_tex_label_from_numerical_label(10)

    def test_get_jet_type_from_file_prefix(self):
        self.assertEqual(get_jet_type_from_file_prefix("ZJetsToNuNu_"), "QCD")
        self.assertEqual(get_jet_type_from_file_prefix("HToBB_"), "Hbb")
        self.assertEqual(get_jet_type_from_file_prefix("HToCC_"), "Hcc")
        self.assertEqual(get_jet_type_from_file_prefix("HToGG_"), "Hgg")
        self.assertEqual(get_jet_type_from_file_prefix("HToWW4Q_"), "H4q")
        self.assertEqual(get_jet_type_from_file_prefix("HToWW2Q1L_"), "Hqql")
        self.assertEqual(get_jet_type_from_file_prefix("ZToQQ_"), "Zqq")
        self.assertEqual(get_jet_type_from_file_prefix("WToQQ_"), "Wqq")
        self.assertEqual(get_jet_type_from_file_prefix("TTBar_"), "Tbqq")
        self.assertEqual(get_jet_type_from_file_prefix("TTBarLep_"), "Tbl")
        with self.assertRaises(ValueError):
            get_jet_type_from_file_prefix("invalid_prefix")
