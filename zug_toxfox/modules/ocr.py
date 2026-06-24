# type: ignore
import os
import warnings
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial import distance_matrix

from zug_toxfox import getLogger, pipeline_config

log = getLogger(__name__)

# Suppress warning related to torch.load in easy_ocr
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")


class BoundingBoxProcessor:
    def __init__(self, bounding_boxes):
        self.bounding_boxes = bounding_boxes
        self.threshold_factor = pipeline_config.ocr.bb_threshold_factor

        self.left_points = []
        self.right_points = []
        self.adjacency = defaultdict(list)
        self.used_indices = set()

    def get_vertical_middle_points(self, polygon):
        """Calculate the vertical middle points of the left and right ends of the bounding box."""
        top_left_y = polygon[0][1]
        bottom_left_y = polygon[3][1]
        top_right_y = polygon[1][1]
        bottom_right_y = polygon[2][1]

        left_middle_y = (top_left_y + bottom_left_y) / 2
        right_middle_y = (top_right_y + bottom_right_y) / 2

        left_middle_point = (int(polygon[0][0]), int(left_middle_y))
        right_middle_point = (int(polygon[1][0]), int(right_middle_y))

        return np.array(left_middle_point), np.array(right_middle_point)

    def connect_points(self):
        """Implements depth-first search (DFS) algorithm to cluster lines based on the adjacency matrix."""
        for box in self.bounding_boxes:
            left_middle_point, right_middle_point = self.get_vertical_middle_points(box)
            self.left_points.append(left_middle_point)
            self.right_points.append(right_middle_point)

        self.left_points = np.array(self.left_points)
        self.right_points = np.array(self.right_points)

        dist_matrix = distance_matrix(self.right_points, self.left_points, 2)
        nn_index = np.argmin(dist_matrix, axis=1)
        nn_distance = np.min(dist_matrix, axis=1)

        for n in range(len(nn_index)):
            sorted_distances = np.sort(dist_matrix[n])[:5]

            local_mean = np.mean(sorted_distances)
            local_std = np.std(sorted_distances)

            adaptive_threshold = (local_mean + local_std) * self.threshold_factor

            if nn_distance[n] <= adaptive_threshold:
                self.adjacency[n].append(nn_index[n])
                self.adjacency[nn_index[n]].append(n)

        lines = []
        for n in range(len(self.bounding_boxes)):
            if n not in self.used_indices:
                line = []
                self.depth_first_search(n, line)
                lines.append(line)

        return lines

    def depth_first_search(self, node, line):
        """Depth-first search (DFS) for clustering lines."""
        stack = [node]
        while stack:
            current = stack.pop()
            if current not in self.used_indices:
                self.used_indices.add(current)
                line.append(self.bounding_boxes[current])
                for neighbor in self.adjacency[current]:
                    if neighbor not in self.used_indices:
                        stack.append(neighbor)

    def sort_bounding_boxes(self):
        """Sort bounding boxes first vertically by lines and then horizontally within each line."""
        lines = self.connect_points()
        sorted_lines = [sorted(line, key=lambda box: self.get_center(box)[0]) for line in lines]
        sorted_lines = sorted(sorted_lines, key=lambda line: self.get_center(line[0])[1])
        return [box for line in sorted_lines for box in line]

    def get_center(self, polygon):
        """Calculate the center point of a bounding box."""
        x_coords = [point[0] for point in polygon]
        y_coords = [point[1] for point in polygon]

        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        return (int(center_x), int(center_y))


class OCR:
    """Text detection + recognition with a pluggable engine.

    The engine is chosen via the OCR_ENGINE env var (default "easyocr"). Every backend
    normalizes its output to a list of ``[polygon, text, confidence]`` triples, so the
    downstream reading-order clustering and token filtering are engine-agnostic. This lets
    us benchmark alternative engines (RapidOCR / PaddleOCR / ...) as a one-line config swap
    without rewriting the postprocessing that depends on bounding-box geometry.
    """

    def __init__(self, engine: str | None = None):
        # Precedence: explicit arg > OCR_ENGINE env > pipeline_config.ocr.engine > "easyocr".
        cfg_engine = pipeline_config.ocr.engine
        self.engine = (engine or os.environ.get("OCR_ENGINE") or (cfg_engine if isinstance(cfg_engine, str) else "easyocr")).lower()
        gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
        log.info("OCR engine: %s (gpu=%s)", self.engine, gpu)
        self._build_backend(self.engine, gpu)

        # Auto-orientation: many real ingredient panels are photographed sideways (the strip is
        # rotated 90/180/270). Detectors that assume straight pages then read ~nothing -> zero
        # predictions. When the straight read looks weak (or the crop is an extreme strip), re-OCR
        # at 90/180/270 and keep the orientation with the most legible recognized text. Costs extra
        # OCR passes ONLY on those weak/strip images, so upright photos are unaffected. AUTO_ORIENT
        # env toggles it (default on); thresholds tunable via ORIENT_* env.
        self.auto_orient = (os.environ.get("AUTO_ORIENT", "1").lower() not in ("0", "false", "no", "off"))
        self.orient_trigger = float(os.environ.get("ORIENT_TRIGGER", "120"))  # straight-read yield below this -> search
        self.orient_margin = float(os.environ.get("ORIENT_MARGIN", "1.15"))   # a rotation must beat straight by this factor

    def _build_backend(self, engine: str, gpu: bool) -> None:
        if engine == "easyocr":
            import easyocr

            self.processor = easyocr.Reader(["de", "en"], gpu=gpu)
        elif engine == "doctr":
            from zug_toxfox.modules.ocr_backends import DocTRBackend

            cfg_reco = pipeline_config.ocr.doctr_reco
            reco = os.environ.get("DOCTR_RECO") or (cfg_reco if isinstance(cfg_reco, str) else "crnn_vgg16_bn")
            self.processor = DocTRBackend(reco_arch=reco)
        elif engine == "rapidocr":
            from zug_toxfox.modules.ocr_backends import RapidOCRBackend

            self.processor = RapidOCRBackend()
        elif engine == "paddleocr":
            from zug_toxfox.modules.ocr_backends import PaddleOCRBackend

            self.processor = PaddleOCRBackend(gpu=gpu)
        else:
            raise ValueError(f"Unknown OCR_ENGINE '{engine}'")  # noqa: TRY003

    def _detect(self, image: np.ndarray) -> list[list[Any]]:
        """Run detection+recognition, normalized to [polygon, text, confidence] triples."""
        if self.engine == "easyocr":
            return self.processor.readtext(image, paragraph=False)
        return self.processor.readtext(image)

    @staticmethod
    def _yield_score(triples: list[list[Any]]) -> float:
        """Legibility proxy: total characters of recognized text weighted by confidence.

        Correctly-oriented text yields many high-confidence characters; rotated text yields a
        few low-confidence garbage tokens. Engine-agnostic (text at [1], confidence at [2]).
        """
        s = 0.0
        for t in triples:
            txt = (t[1] or "").strip()
            if len(txt) > 1:
                conf = t[2] if len(t) > 2 and t[2] is not None else 1.0
                s += len(txt) * float(conf)
        return s

    _ROT = ((90, cv2.ROTATE_90_CLOCKWISE), (270, cv2.ROTATE_90_COUNTERCLOCKWISE), (180, cv2.ROTATE_180))

    def _detect_oriented(self, image: np.ndarray) -> list[list[Any]]:
        """Detect at the image's best of {0,90,180,270} when the straight read looks weak.

        Only triggers the extra passes when the upright read is weak (yield < ORIENT_TRIGGER) or
        the crop is an extreme strip (aspect > ORIENT_ASPECT) -- the signature of a sideways panel
        -- so normal upright photos pay no extra OCR cost. A rotation must beat upright by
        ORIENT_MARGIN to be chosen (guards correctly-oriented but sparse panels).
        """
        triples0 = self._detect(image)
        if not self.auto_orient:
            return triples0
        base = self._yield_score(triples0)
        # A strong upright read is trusted regardless of shape (no extra passes). Only a weak read
        # -- the signature of a sideways/upside-down panel the detector couldn't parse -- triggers
        # the rotation search, so normal photos (even tall portrait ones) pay no extra OCR cost.
        if base >= self.orient_trigger:
            return triples0
        best_t, best_s = triples0, base
        for ang, code in self._ROT:
            t = self._detect(cv2.rotate(image, code))
            s = self._yield_score(t)
            if s > best_s * self.orient_margin and s > base * self.orient_margin:
                best_t, best_s = t, s
        if best_t is not triples0:
            log.info("Auto-orient: rotated (base yield %.0f -> %.0f)", base, best_s)
        return best_t

    def easyocr_to_dict(self, ocr_output: list[list[Any]], paragraph: bool = False) -> dict[str, Any]:
        """
        Converts EasyOCR output to a structured dictionary.

        Args:
            ocr_output (list[any]): List of the raw ocr outputs
            paragraph (bool, optional): If "paragraph" argument is true in the easy_ocr.readtext function, only a single box is returned without a confidence score. In this case, we skip the "level" key in the output dict.
            Defaults to "False".

        Returns:
            dict[str, any]: Dictionary with keys "text", "polygons", and optionally "level" (model confidence).
        """
        output_dict: dict[str, list[Any]] = {"text": [], "polygons": []}
        if not paragraph:
            output_dict["level"] = []

        # Detectors other than EasyOCR (docTR, RapidOCR) can legitimately return zero boxes on
        # tiny/blank crops; the reading-order clustering below assumes >=1 box, so short-circuit.
        if not ocr_output:
            return output_dict

        polygons: list[tuple[tuple[float, float], tuple[float, float]]] = []
        texts: list[str] = []
        levels: list[float] = []

        for output in ocr_output:
            polygons.append(output[0])
            texts.append(output[1])
            if not paragraph:
                levels.append(output[2])

        box_processor = BoundingBoxProcessor(polygons)
        sorted_polygons = box_processor.sort_bounding_boxes()

        sorted_texts = [texts[polygons.index(polygon)] for polygon in sorted_polygons]
        if not paragraph:
            sorted_levels = [levels[polygons.index(polygon)] for polygon in sorted_polygons]
            output_dict["level"] = sorted_levels

        output_dict["polygons"] = sorted_polygons
        output_dict["text"] = sorted_texts

        return output_dict

    def draw_bounding_box(self, image: np.ndarray, output: dict[str, Any]) -> np.ndarray:
        """Draws bounding boxes for a given image."""
        n_boxes = len(output["text"])
        for i in range(n_boxes):
            polygons = np.asarray(output["polygons"][i], dtype=np.int32)
            cv2.polylines(image, [polygons], True, (0, 255, 0), 1)
        return image

    def get_filtered_output(self, output: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Filters bounding box output to reduce the number of irrelevant boxes."""
        filtered_indices = {i for i, text in enumerate(output["text"]) if text.strip() and len(text.strip()) > 1}

        return {key: [output[key][i] for i in filtered_indices] for key in output}

    def process_easyocr(self, image: np.ndarray):
        output = self._detect_oriented(image)
        if self.engine == "doctr":
            # docTR emits whole-line boxes already, so the correct reading order is a simple
            # top-to-bottom (then left-to-right) sort. The DFS line-clustering below is built for
            # EasyOCR's sub-line word boxes and mis-orders multi-region labels (it interleaves the
            # marketing text with the ingredient panel), which both scrambles multi-word INCI names
            # and breaks the "Ingredients:" region cut. Sort by box top-y, then left-x.
            ordered = sorted(output, key=lambda t: (t[0][0][1], t[0][0][0])) if output else []
            return self.get_filtered_output({"polygons": [t[0] for t in ordered], "text": [t[1] for t in ordered]})
        output = self.easyocr_to_dict(ocr_output=output, paragraph=True)
        return self.get_filtered_output(output)

    def process_image(
        self, processed_image: np.ndarray, image: np.ndarray | None = None, debug: bool = False
    ) -> tuple[np.ndarray | None, list[str]]:
        """Executes ingredient detection and draws bounding boxes for a single image."""
        output = self.process_easyocr(processed_image)

        if debug:
            image_bb = self.draw_bounding_box(image, output)
            return image_bb, output["text"]
        else:
            return output["text"]
