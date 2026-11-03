"""Check ambiguous class proposals without removing overlapping real hands."""

from dataclasses import dataclass
import unittest

from egohand_live.detector import select_distinct_detections


@dataclass
class Detection:
    label_id: int
    score: float
    xyxy: tuple


class DetectionSelectionTests(unittest.TestCase):
    def test_same_box_two_classes_keeps_stronger_regardless_of_order(self):
        weak = Detection(1, 0.6, (10, 20, 100, 120))
        strong = Detection(2, 0.9, weak.xyxy)
        for detections in ([weak, strong], [strong, weak]):
            accepted, rejected = select_distinct_detections(detections)
            self.assertEqual(accepted, [strong])
            self.assertEqual(rejected, [weak])

    def test_distinct_overlapping_hands_remain(self):
        left = Detection(1, 0.9, (10, 20, 100, 120))
        right = Detection(2, 0.8, (11, 20, 101, 120))
        self.assertEqual(select_distinct_detections([left, right]), ([left, right], []))

    def test_empty_and_unknown_class(self):
        self.assertEqual(select_distinct_detections([]), ([], []))
        self.assertEqual(
            select_distinct_detections([Detection(0, 0.9, (0, 0, 1, 1))]), ([], [])
        )
