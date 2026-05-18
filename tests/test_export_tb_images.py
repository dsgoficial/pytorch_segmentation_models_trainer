# -*- coding: utf-8 -*-
"""Tests for tools/export_tb_images.py and the export-tb-images CLI command."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from click.testing import CliRunner
from tensorboardX import SummaryWriter

from pytorch_segmentation_models_trainer.tools.cli import cli
from pytorch_segmentation_models_trainer.tools.export_tb_images import (
    export_images,
    find_event_files,
    list_image_tags,
    parse_steps_filter,
)


def _write_events(log_dir: Path, events: list) -> None:
    """Write image events via tensorboardX. events = [(tag, step, np_img)]."""
    writer = SummaryWriter(log_dir=str(log_dir))
    for tag, step, img in events:
        writer.add_image(tag, img, global_step=step)
    writer.close()


IMG = np.zeros((3, 8, 8), dtype=np.uint8)


class TestParseStepsFilter(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(parse_steps_filter(None))

    def test_single_integer(self):
        self.assertEqual(parse_steps_filter("5"), {5})

    def test_range(self):
        self.assertEqual(parse_steps_filter("0-3"), {0, 1, 2, 3})

    def test_comma_separated(self):
        self.assertEqual(parse_steps_filter("0,10,20"), {0, 10, 20})

    def test_mixed_range_and_values(self):
        result = parse_steps_filter("0-2,10")
        self.assertEqual(result, {0, 1, 2, 10})


class TestFindEventFiles(unittest.TestCase):
    def test_finds_tfevents_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_events(Path(tmp), [("tag", 0, IMG)])
            files = find_event_files(Path(tmp))
            self.assertEqual(len(files), 1)
            self.assertIn("tfevents", files[0].name)

    def test_empty_dir_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(find_event_files(Path(tmp)), [])


class TestListImageTags(unittest.TestCase):
    def test_returns_all_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_events(
                Path(tmp),
                [("alpha/tag", 1, IMG), ("beta/tag", 2, IMG)],
            )
            tags = list_image_tags(Path(tmp))
            self.assertIn("alpha/tag", tags)
            self.assertIn("beta/tag", tags)

    def test_empty_dir_returns_empty_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(list_image_tags(Path(tmp)), set())


class TestExportImages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.log_dir = Path(self.tmp) / "logs"
        self.out_dir = Path(self.tmp) / "out"
        self.log_dir.mkdir()
        _write_events(
            self.log_dir,
            [
                ("tile/A", 1, IMG),
                ("tile/A", 5, IMG),
                ("tile/B", 1, IMG),
            ],
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp)

    def test_export_all_returns_correct_count(self):
        count = export_images(self.log_dir, self.out_dir)
        self.assertEqual(count, 3)

    def test_export_creates_png_files(self):
        export_images(self.log_dir, self.out_dir)
        pngs = list(self.out_dir.glob("*.png"))
        self.assertEqual(len(pngs), 3)

    def test_tag_filter_restricts_output(self):
        count = export_images(self.log_dir, self.out_dir, tags=["tile/A"])
        self.assertEqual(count, 2)

    def test_steps_filter_restricts_output(self):
        count = export_images(self.log_dir, self.out_dir, steps="1")
        self.assertEqual(count, 2)

    def test_tag_and_steps_combined(self):
        count = export_images(self.log_dir, self.out_dir, tags=["tile/A"], steps="5")
        self.assertEqual(count, 1)

    def test_output_filename_format(self):
        export_images(self.log_dir, self.out_dir, tags=["tile/A"], steps="1")
        files = list(self.out_dir.glob("*.png"))
        self.assertEqual(len(files), 1)
        self.assertIn("tile_A_step000001", files[0].name)

    def test_output_dir_created_if_missing(self):
        new_out = Path(self.tmp) / "nonexistent" / "out"
        export_images(self.log_dir, new_out)
        self.assertTrue(new_out.exists())

    def test_unknown_tag_filter_exports_nothing(self):
        count = export_images(self.log_dir, self.out_dir, tags=["no_such_tag"])
        self.assertEqual(count, 0)

    def test_out_of_range_steps_exports_nothing(self):
        count = export_images(self.log_dir, self.out_dir, steps="99")
        self.assertEqual(count, 0)


class TestExportTbImagesCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.tmp = tempfile.mkdtemp()
        self.log_dir = Path(self.tmp) / "logs"
        self.log_dir.mkdir()
        _write_events(
            self.log_dir,
            [("alpha/tag", 1, IMG), ("beta/tag", 2, IMG)],
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp)

    def test_list_tags_shows_available_tags(self):
        result = self.runner.invoke(
            cli, ["export-tb-images", str(self.log_dir), "--list-tags"]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("alpha/tag", result.output)
        self.assertIn("beta/tag", result.output)

    def test_list_tags_empty_dir_reports_none(self):
        with tempfile.TemporaryDirectory() as empty:
            result = self.runner.invoke(cli, ["export-tb-images", empty, "--list-tags"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("No image tags found", result.output)

    def test_export_all_reports_count(self):
        out = str(Path(self.tmp) / "out")
        result = self.runner.invoke(
            cli, ["export-tb-images", str(self.log_dir), "-o", out]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Exported 2", result.output)

    def test_export_with_tag_filter(self):
        out = str(Path(self.tmp) / "out")
        result = self.runner.invoke(
            cli,
            [
                "export-tb-images",
                str(self.log_dir),
                "--tags",
                "alpha/tag",
                "-o",
                out,
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Exported 1", result.output)

    def test_export_with_steps_filter(self):
        out = str(Path(self.tmp) / "out")
        result = self.runner.invoke(
            cli,
            ["export-tb-images", str(self.log_dir), "--steps", "1", "-o", out],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Exported 1", result.output)
