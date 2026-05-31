from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np


def write_yolo_txt(
        preds: np.ndarray, image_size: Tuple[int, int], out_path: Path
) -> None:
        """Write YOLO-format text file: class_id cx cy w h confidence per line.

        preds: numpy array Nx6: [class, x1, y1, x2, y2, conf]
        image_size: (width, height)
        out_path: full path to write (including filename)
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        w, h = image_size
        lines: List[str] = []
        if preds is None or preds.shape[0] == 0:
                out_path.write_text("")
                return

        for row in preds:
                cls = int(row[0])
                x1, y1, x2, y2 = [float(v) for v in row[1:5]]
                conf = float(row[5])
                cx = (x1 + x2) / 2.0 / float(w)
                cy = (y1 + y2) / 2.0 / float(h)
                width = (x2 - x1) / float(w)
                height = (y2 - y1) / float(h)
                lines.append(
                        f"{cls} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f} {conf:.6f}"
                )

        out_path.write_text("\n".join(lines) + "\n")


def write_voc_xml(
        preds: np.ndarray,
        image_size: Tuple[int, int],
        out_path: Path,
        image_filename: str,
        folder: Optional[str] = None,
        id2name: Optional[Dict[int, str]] = None,
) -> None:
        """Write a PASCAL VOC-like XML file for detections.

        Each <object> contains <name>, <confidence>, and <bndbox> (xmin,ymin,xmax,ymax).
        preds: Nx6 array as above.
        image_size: (width, height)
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        w, h = image_size

        annotation = ET.Element("annotation")
        folder_el = ET.SubElement(annotation, "folder")
        folder_el.text = folder or ""
        filename_el = ET.SubElement(annotation, "filename")
        filename_el.text = image_filename

        size_el = ET.SubElement(annotation, "size")
        ET.SubElement(size_el, "width").text = str(int(w))
        ET.SubElement(size_el, "height").text = str(int(h))
        ET.SubElement(size_el, "depth").text = "3"

        if preds is not None:
                for row in preds:
                        cls = int(row[0])
                        x1, y1, x2, y2 = [int(float(v)) for v in row[1:5]]
                        conf = float(row[5])

                        obj = ET.SubElement(annotation, "object")
                        name = (
                                id2name.get(cls)
                                if (id2name and cls in id2name)
                                else str(cls)
                        )
                        ET.SubElement(obj, "name").text = name
                        ET.SubElement(obj, "confidence").text = f"{conf:.6f}"
                        bnd = ET.SubElement(obj, "bndbox")
                        ET.SubElement(bnd, "xmin").text = str(max(0, x1))
                        ET.SubElement(bnd, "ymin").text = str(max(0, y1))
                        ET.SubElement(bnd, "xmax").text = str(max(0, x2))
                        ET.SubElement(bnd, "ymax").text = str(max(0, y2))

        tree = ET.ElementTree(annotation)
        tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
