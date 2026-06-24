# type: ignore
"""Alternative OCR engine backends.

Each backend exposes ``readtext(image) -> list[[polygon, text, confidence]]`` matching
EasyOCR's ``Reader.readtext(..., paragraph=False)`` output shape, so the engine-agnostic
reading-order clustering and token filtering in ``ocr.OCR`` work unchanged. Backends are
imported lazily by ``OCR._build_backend`` only when their engine is selected, so installing
one engine never forces the others' (heavy) dependencies.

Polygons are returned as plain Python lists (not numpy arrays): the downstream
``easyocr_to_dict`` does ``polygons.index(polygon)``, which needs value equality on lists.
"""
from typing import Any


class RapidOCRBackend:
    """RapidOCR (ONNXRuntime, PP-OCRv5 models). CPU-friendly, no torch dependency.

    Reads dense Latin/INCI text markedly better than EasyOCR (e.g. recovers
    TRIETHOXYCAPRYLYLSILANE / METHYLTRIMETHICONE where EasyOCR produces garbage).
    """

    def __init__(self):
        from rapidocr import RapidOCR

        self.engine = RapidOCR()

    def readtext(self, image) -> list[list[Any]]:
        res = self.engine(image)
        if res is None or res.boxes is None or res.txts is None:
            return []
        out: list[list[Any]] = []
        for box, txt, score in zip(res.boxes, res.txts, res.scores):  # noqa: B905
            poly = box.tolist() if hasattr(box, "tolist") else [list(p) for p in box]
            out.append([poly, txt, float(score)])
        return out


class DocTRBackend:
    """docTR (Mindee) two-stage det+recog on the already-installed torch+cpu.

    Strongest published evidence for this exact task: in the HalalBench food-packaging
    ingredient benchmark docTR had the highest German F1 (0.655 vs EasyOCR 0.621) at lower
    RAM. Has no notion of reading order itself, so per-word boxes are fed to the existing
    BoundingBoxProcessor line-clustering (kept engine-agnostic).
    """

    def __init__(self, det_arch: str | None = None, reco_arch: str | None = None):
        import os

        os.environ.setdefault("USE_TORCH", "1")  # only torch is installed; force its backend
        from doctr.models import ocr_predictor

        det_arch = det_arch or os.environ.get("DOCTR_DET", "db_resnet50")
        reco_arch = reco_arch or os.environ.get("DOCTR_RECO", "crnn_vgg16_bn")
        self.model = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True)
        # docTR resizes every page to a fixed square (default 1024) before detection, so small
        # text on dense ingredient panels falls below the recognizable scale. Raising the detection
        # resolution (DOCTR_DET_SIZE, e.g. 1536/2048) recovers small text at the cost of more
        # compute/RAM. Bilinear+aspect-preserving, so it never distorts the page.
        size = int(os.environ.get("DOCTR_DET_SIZE", "0"))
        if size:
            self.model.det_predictor.pre_processor.resize.size = (size, size)

    def readtext(self, image) -> list[list[Any]]:
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.model([rgb])
        page = result.pages[0]
        height, width = page.dimensions  # (H, W)
        out: list[list[Any]] = []
        # Emit ONE detection per docTR line (words joined in docTR's own reading order), mirroring
        # EasyOCR's line-level boxes. Emitting per-word and re-clustering with BoundingBoxProcessor
        # scrambles word order and shatters multi-word INCI names ("Ethylhexyl Salicylate"), which
        # collapses recall. docTR already orders blocks->lines->words correctly, so trust it.
        for block in page.blocks:
            for line in block.lines:
                words = line.words
                if not words:
                    continue
                xs: list[float] = []
                ys: list[float] = []
                for word in words:
                    geom = word.geometry
                    if len(geom) == 2:
                        (x0, y0), (x1, y1) = geom
                        xs += [x0, x1]
                        ys += [y0, y1]
                    else:
                        for px, py in geom:
                            xs.append(px)
                            ys.append(py)
                xmin, xmax = min(xs) * width, max(xs) * width
                ymin, ymax = min(ys) * height, max(ys) * height
                poly = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
                text = " ".join(w.value for w in words)
                conf = float(sum(w.confidence for w in words) / len(words))
                out.append([poly, text, conf])
        return out


class PaddleOCRBackend:
    """PaddleOCR (PP-OCRv4/v5). Heavier than RapidOCR; kept as a fallback candidate."""

    def __init__(self, gpu: bool = False):
        from paddleocr import PaddleOCR

        self.engine = PaddleOCR(use_angle_cls=True, lang="german", show_log=False, use_gpu=gpu)

    def readtext(self, image) -> list[list[Any]]:
        result = self.engine.ocr(image, cls=True)
        out: list[list[Any]] = []
        if not result or result[0] is None:
            return out
        for line in result[0]:
            box, (txt, score) = line
            poly = [list(map(float, p)) for p in box]
            out.append([poly, txt, float(score)])
        return out
