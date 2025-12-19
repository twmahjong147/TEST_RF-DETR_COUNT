# test.py — Documentation

**File:** [test.py](test.py)

**Overview:**
- `test.py` runs inference on images in `samples/`, produces annotated images in `annotated_samples/`, and writes per-image class counts to `annotated_samples/<image>/class_counts.txt`.
- It uses a model from the local `inference` module (`get_model`) and the `supervision` package for detection/annotation utilities.

**Main behavior:**
- Iterate over image files in `samples/`.
- Run `model.infer(image, confidence=0.001)` to obtain predictions.
- Convert predictions to `supervision.Detections` and annotate/save images.
- For each detected class, the script filters detections by area outliers, removes contained detections (IoA-based), annotates per-class images, and writes class counts.

**Important functions (in `test.py`)**

- `filter_area_outliers(detections, labels, std_factor=2)`
  - Purpose: Remove detections whose bounding-box area is an outlier (outside mean ± std_factor * std).
  - Inputs:
    - `detections`: either a `supervision.Detections` object or a sequence/ndarray of boxes/detection objects that expose `.xyxy` or indexable [x1,y1,x2,y2].
    - `labels`: list of class names corresponding to `detections` order.
    - `std_factor`: float (default 2). Threshold for outlier exclusion.
  - Returns: `(filtered_detections, filtered_labels)` using the same detection container type when possible.
  - Notes: If there are no boxes or zero variance in areas, returns inputs unchanged or empty consistent types.

- `remove_contained_detections(detections, labels, ioa_thresh=1.0)`
  - Purpose: Remove detections that are (almost) entirely contained inside another detection using IoA (intersection over area of the candidate box).
  - Inputs:
    - `detections`: same accepted forms as above.
    - `labels`: list of class names corresponding to `detections` order.
    - `ioa_thresh`: float (default 1.0). A box `i` is considered contained in box `j` if IoA(i,j) >= `ioa_thresh`.
  - Behavior: When containment is detected, the function prefers keeping the larger-area box (ties favor the kept box) and removes the contained (smaller) box.
  - Returns: `(filtered_detections, filtered_labels)` in the same types when possible.
  - Notes: Use `ioa_thresh < 1.0` (e.g., `0.95`) to treat nearly-contained boxes as contained.

**Usage examples**

- Run the script (from repository root):

```bash
python test.py
```

- Import and use helpers interactively (Python):

```python
from test import filter_area_outliers, remove_contained_detections

# detections: supervision.Detections or list-like of boxes
filtered_dets, filtered_labels = filter_area_outliers(detections, labels, std_factor=2)
filtered_dets, filtered_labels = remove_contained_detections(filtered_dets, filtered_labels, ioa_thresh=1.0)
```

**Dependencies**
- Python packages: `supervision`, `numpy`, `Pillow` (PIL)
- Local module: `inference` (must provide `get_model(name)` and a `.infer(image, confidence)` method on the returned model)

Example install (if needed):

```bash
pip install numpy pillow
# install supervision per its docs (may require pip/conda and repo credentials)
```

**Configuration / Tuning**
- `std_factor` in `filter_area_outliers` controls how aggressively small/large areas are removed.
- `ioa_thresh` in `remove_contained_detections` controls containment sensitivity. `1.0` = strict full containment; `0.9-0.99` = near containment.
- The script uses a very low model `confidence=0.001` when calling `model.infer` so it collects many candidates before pruning; raise this to reduce false positives earlier.

**Where to modify**
- To change the output folder, edit the `OUTPUT_DIR` constant at the top of `test.py`.
- To change which model is used, edit the `get_model("rfdetr-large")` call.

**Notes & Caveats**
- The helper functions try to preserve the input detection container type (e.g., `supervision.Detections`) when possible; when the input is a Python list or other container, the returned `filtered_detections` will be of the same container semantics used in the script (typically a list or a `supervision.Detections` slice).
- The containment removal operates per-class in the script; it does not compare boxes across different classes.

---

Document created: `TEST_PY_DOC.md` (this file). For quick reference see the script itself: [test.py](test.py)
