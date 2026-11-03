"""OpenCV preview with a drag-to-rotate, wrist-relative 3D mesh panel."""

import math
import cv2
import numpy as np


class Preview:
    def __init__(self, show_window=True):
        self.yaw = 0.0
        self.pitch = 0.0
        self.last = None
        if show_window:
            cv2.namedWindow("EgoHand Live", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("EgoHand Live", self.mouse)

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.last = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.last = None
        elif event == cv2.EVENT_MOUSEMOVE and self.last:
            self.yaw += (x - self.last[0]) * 0.01
            self.pitch += (y - self.last[1]) * 0.01
            self.last = (x, y)

    def render(self, image, results, faces, camera, label):
        canvas = image.copy()
        panel = np.full((image.shape[0], 640, 3), 28, np.uint8)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        rotation = np.array(
            [[cy, sy * sp, sy * cp], [0, cp, -sp], [-sy, cy * sp, cy * cp]]
        )
        for side, result in results.items():
            color = (80, 210, 110) if side == "left" else (230, 150, 70)
            points = result["pred_keypoints_3d"] + result["transl"]
            uv = camera.project(points)
            for pixel, point in zip(uv, points):
                if np.isfinite(pixel).all() and point[2] > 0:
                    if (np.abs(pixel) < 1e6).all():
                        cv2.circle(
                            canvas, tuple(np.rint(pixel).astype(int)), 3, color, -1
                        )
            vertices = result["pred_vertices"] @ rotation.T
            offset = 170 if side == "left" else 470
            projected = vertices[:, :2] * 1200 + [offset, panel.shape[0] / 2]
            triangles = np.asarray(faces[side], dtype=int)
            order = np.argsort(vertices[triangles, 2].mean(axis=1))[::-1]
            for triangle in triangles[order]:
                polygon = np.rint(projected[triangle]).astype(np.int32)
                cv2.fillConvexPoly(panel, polygon, color)
                cv2.polylines(panel, [polygon], True, (60, 65, 70), 1)
            joints = result["pred_keypoints_3d"] @ rotation.T
            joint_pixels = np.rint(
                joints[:, :2] * 1200 + [offset, panel.shape[0] / 2]
            ).astype(np.int32)
            for chain in (
                (0, 1, 2, 3, 4),
                (0, 5, 6, 7, 8, 9),
                (0, 10, 11, 12, 13, 14),
                (0, 15, 16, 17, 18, 19),
                (0, 20, 21, 22, 23, 24),
            ):
                for a, b in zip(chain, chain[1:]):
                    cv2.line(
                        panel,
                        tuple(joint_pixels[a]),
                        tuple(joint_pixels[b]),
                        (235, 235, 235),
                        2,
                    )
            for pixel in joint_pixels:
                cv2.circle(panel, tuple(pixel), 3, (30, 30, 30), -1)
        cv2.putText(
            panel,
            "Wrist-relative 3D | drag to rotate",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
        )
        view = np.hstack([canvas, panel])
        lines = label.splitlines()
        line_height = 32
        top = max(0, view.shape[0] - line_height * len(lines) - 14)
        cv2.rectangle(view, (0, top), (view.shape[1], view.shape[0]), (20, 20, 20), -1)
        for index, line in enumerate(lines):
            text_width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0][0]
            scale = min(0.75, 0.75 * (view.shape[1] - 20) / max(1, text_width))
            cv2.putText(
                view,
                line,
                (10, top + 27 + index * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return view

    def show(self, image, results, faces, camera, label):
        view = self.render(image, results, faces, camera, label)
        cv2.imshow("EgoHand Live", view)
        return cv2.waitKey(1) & 0xFF not in (27, ord("q"))
