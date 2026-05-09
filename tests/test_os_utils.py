# -*- coding: utf-8 -*-
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from pytorch_segmentation_models_trainer.utils.os_utils import (
    remove_folder,
    create_folder,
    hash_file,
    make_path_relative,
    import_module_from_cfg,
)


class TestOsUtils(unittest.TestCase):
    def setUp(self):
        self.test_dir = TemporaryDirectory()
        self.test_path = self.test_dir.name

    def tearDown(self):
        self.test_dir.cleanup()

    def test_create_and_remove_folder(self):
        folder_path = os.path.join(self.test_path, "new_folder")
        res = create_folder(folder_path)
        self.assertEqual(res, folder_path)
        self.assertTrue(os.path.exists(folder_path))

        # Test remove_folder
        self.assertTrue(remove_folder(folder_path))
        self.assertFalse(os.path.exists(folder_path))

        # Test remove non-existent
        self.assertTrue(remove_folder(os.path.join(self.test_path, "not_here")))

    def test_hash_file(self):
        file_path = os.path.join(self.test_path, "test.txt")
        with open(file_path, "w") as f:
            f.write("hello world")

        h = hash_file(file_path)
        # sha1 of "hello world" is 2aae6c35c94fcfb415dbe95f408b9ce91ee846ed
        self.assertEqual(h, "2aae6c35c94fcfb415dbe95f408b9ce91ee846ed")

    def test_make_path_relative(self):
        base = "/home/user/project"
        full = "/home/user/project/data/img.png"
        res = make_path_relative(full, base)
        self.assertEqual(res, os.path.join(base, "data/img.png"))

        # Test with None
        self.assertIsNone(make_path_relative(None, base))

    def test_import_module_from_cfg(self):
        from unittest.mock import MagicMock

        cfg = MagicMock()
        cfg._target_ = "os.path.join"
        func = import_module_from_cfg(cfg)
        self.assertEqual(func, os.path.join)


if __name__ == "__main__":
    unittest.main()
