# -*- coding: utf-8 -*-
import os
import pandas as pd
import runpy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from pytorch_segmentation_models_trainer.tools.data_handlers.csv_to_parquet import (
    convert_csv_to_parquet,
    main,
)


class TestCsvToParquet(unittest.TestCase):
    def setUp(self):
        self.test_dir = TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_convert_single_file(self):
        csv_path = self.test_path / "test.csv"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_csv(csv_path, index=False)

        convert_csv_to_parquet(csv_path)

        parquet_path = self.test_path / "test.parquet"
        self.assertTrue(parquet_path.exists())
        df_parquet = pd.read_parquet(parquet_path)
        pd.testing.assert_frame_equal(df, df_parquet)

    def test_convert_directory(self):
        dir_path = self.test_path / "csv_dir"
        dir_path.mkdir()
        csv1 = dir_path / "test1.csv"
        csv2 = dir_path / "test2.csv"
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"b": [2]})
        df1.to_csv(csv1, index=False)
        df2.to_csv(csv2, index=False)

        convert_csv_to_parquet(dir_path)

        self.assertTrue((dir_path / "test1.parquet").exists())
        self.assertTrue((dir_path / "test2.parquet").exists())

    def test_convert_recursive(self):
        base_dir = self.test_path / "base"
        sub_dir = base_dir / "sub"
        sub_dir.mkdir(parents=True)
        csv1 = base_dir / "test1.csv"
        csv2 = sub_dir / "test2.csv"
        pd.DataFrame({"a": [1]}).to_csv(csv1, index=False)
        pd.DataFrame({"b": [2]}).to_csv(csv2, index=False)

        # Test non-recursive first
        convert_csv_to_parquet(base_dir, recursive=False)
        self.assertTrue((base_dir / "test1.parquet").exists())
        self.assertFalse((sub_dir / "test2.parquet").exists())

        # Test recursive
        convert_csv_to_parquet(base_dir, recursive=True)
        self.assertTrue((sub_dir / "test2.parquet").exists())

    def test_convert_with_output_path(self):
        csv_path = self.test_path / "test.csv"
        pd.DataFrame({"a": [1]}).to_csv(csv_path, index=False)

        output_dir = self.test_path / "output"
        output_dir.mkdir()

        convert_csv_to_parquet(csv_path, output_path=output_dir / "custom.parquet")
        self.assertTrue((output_dir / "custom.parquet").exists())

    def test_skips_non_csv_file(self):
        txt_path = self.test_path / "not_csv.txt"
        txt_path.write_text("a,b\n1,2\n")

        convert_csv_to_parquet(txt_path)

        self.assertFalse((self.test_path / "not_csv.parquet").exists())

    def test_empty_directory_prints_message(self):
        empty_dir = self.test_path / "empty"
        empty_dir.mkdir()

        convert_csv_to_parquet(empty_dir)

        self.assertEqual(list(empty_dir.glob("*.parquet")), [])

    def test_directory_output_path_preserves_relative_structure(self):
        base_dir = self.test_path / "base"
        sub_dir = base_dir / "sub"
        out_dir = self.test_path / "out"
        sub_dir.mkdir(parents=True)
        pd.DataFrame({"a": [1]}).to_csv(sub_dir / "nested.csv", index=False)

        convert_csv_to_parquet(base_dir, output_path=out_dir, recursive=True)

        self.assertTrue((out_dir / "sub" / "nested.parquet").exists())

    def test_directory_conversion_continues_after_read_error(self):
        csv_path = self.test_path / "bad.csv"
        csv_path.write_text("a\n1\n")

        with patch(
            "pytorch_segmentation_models_trainer.tools.data_handlers.csv_to_parquet.pd.read_csv",
            side_effect=RuntimeError("boom"),
        ):
            convert_csv_to_parquet(self.test_path)

        self.assertFalse((self.test_path / "bad.parquet").exists())

    def test_main_returns_for_missing_path(self):
        missing_path = self.test_path / "missing.csv"

        with patch("sys.argv", ["csv_to_parquet", str(missing_path)]):
            self.assertIsNone(main())

    def test_main_converts_existing_path(self):
        csv_path = self.test_path / "main.csv"
        out_path = self.test_path / "main.parquet"
        pd.DataFrame({"a": [1]}).to_csv(csv_path, index=False)

        with patch("sys.argv", ["csv_to_parquet", str(csv_path), "-o", str(out_path)]):
            main()

        self.assertTrue(out_path.exists())

    def test_module_main_guard_runs_main(self):
        csv_path = self.test_path / "guard.csv"
        out_path = self.test_path / "guard.parquet"
        pd.DataFrame({"a": [1]}).to_csv(csv_path, index=False)

        with patch("sys.argv", ["csv_to_parquet", str(csv_path), "-o", str(out_path)]):
            runpy.run_module(
                "pytorch_segmentation_models_trainer.tools.data_handlers.csv_to_parquet",
                run_name="__main__",
            )

        self.assertTrue(out_path.exists())


if __name__ == "__main__":
    unittest.main()
