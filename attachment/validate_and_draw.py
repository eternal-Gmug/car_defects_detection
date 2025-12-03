import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm


def read_label_polygons(label_path, img_w, img_h):
    """Read label file where each line is: class x1 y1 x2 y2 ...
    (normalized coords). Returns a combined binary mask (uint8)
    of shape ``(img_h, img_w)``.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not os.path.exists(label_path):
        return mask
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            # strip class id
            coords = parts[1:]
            try:
                coords = list(map(float, coords))
            except Exception:
                continue
            if len(coords) % 2 != 0:
                continue
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            # x is normalized by width, y by height
            pts[:, 0] = pts[:, 0] * img_w
            pts[:, 1] = pts[:, 1] * img_h
            pts_i = np.round(pts).astype(np.int32)
            if pts_i.shape[0] >= 3:
                cv2.fillPoly(mask, [pts_i], 1)
    return mask


def read_label_bboxes(label_path, img_w, img_h, fmt='yolo'):
    """Read bbox-style label file and return a binary mask.

    Supports common normalized YOLO format lines: class x_center y_center w h
    where coordinates are normalized to [0,1].

    Parameters
    - label_path: path to .txt label file
    - img_w, img_h: image width and height in pixels
    - fmt: currently only 'yolo' (x_center,y_center,w,h normalized)
      is supported

    Returns
    - mask: uint8 binary mask of shape (img_h, img_w)
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not os.path.exists(label_path):
        return mask
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Expect at least 5 parts for bbox: class x y w h
            if len(parts) < 5:
                continue
            vals = parts[1:5]
            try:
                x_c, y_c, bw, bh = map(float, vals)
            except Exception:
                continue

            # convert normalized center,w,h to pixel xyxy
            if fmt == 'yolo':
                x_c_px = x_c * img_w
                y_c_px = y_c * img_h
                bw_px = bw * img_w
                bh_px = bh * img_h
                x0 = int(round(x_c_px - bw_px / 2.0))
                y0 = int(round(y_c_px - bh_px / 2.0))
                x1 = int(round(x_c_px + bw_px / 2.0))
                y1 = int(round(y_c_px + bh_px / 2.0))
            else:
                # unsupported format: skip
                continue

            # clip to image
            x0 = max(0, min(img_w - 1, x0))
            x1 = max(0, min(img_w - 1, x1))
            y0 = max(0, min(img_h - 1, y0))
            y1 = max(0, min(img_h - 1, y1))

            if x1 > x0 and y1 > y0:
                cv2.rectangle(mask, (x0, y0), (x1, y1), 1, thickness=-1)

    return mask


def masks_from_result(result, img_h, img_w):
    """
    Extract a single combined binary mask from an ultralytics Results object.
    Tries multiple access patterns for different ultralytics versions.
    """
    pred_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not hasattr(result, 'masks') or result.masks is None:
        return pred_mask

    masks = result.masks
    # Try polygon representation (xy)
    try:
        polys = masks.xy
        if polys is not None:
            for poly in polys:
                pts = np.array(poly, dtype=np.int32)
                if pts.shape[0] >= 3:
                    cv2.fillPoly(pred_mask, [pts], 1)
            if pred_mask.sum() > 0:
                return pred_mask
    except Exception:
        pass

    # Try mask data tensor (n, h, w)
    try:
        data = masks.data
        # convert to numpy if tensor
        arr = np.array(data)
        if arr.ndim == 3:
            pred_mask = np.any(arr, axis=0).astype(np.uint8)
            return pred_mask
        if arr.ndim == 2:
            pred_mask = (arr > 0).astype(np.uint8)
            return pred_mask
    except Exception:
        pass

    return pred_mask


def masks_from_result_boxes(result, img_h, img_w):
    """Build a combined binary mask from detection boxes in a Results object.

    This is separate from `masks_from_result` so users can choose which to
    call depending on whether predictions include masks or only boxes.
    """
    pred_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if result is None:
        return pred_mask

    # Try boxes attribute (multiple ultralytics versions)
    boxes = None
    try:
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
    except Exception:
        boxes = None

    if boxes is None:
        # try results.boxes.xyxy style
        try:
            boxes = getattr(result, 'boxes', None)
        except Exception:
            boxes = None

    if boxes is None:
        return pred_mask

    # Extract numeric box coordinates robustly
    try:
        # ultralytics Boxes may have .xyxy or .xyxy.tolist() or be iterable
        coords = []
        if hasattr(boxes, 'xyxy'):
            arr = np.array(boxes.xyxy)
            if arr.size > 0:
                coords = arr
        else:
            # try iterating boxes
            for b in boxes:
                # each b may have .xyxy or be array-like
                if hasattr(b, 'xyxy'):
                    coords.append(np.array(b.xyxy).reshape(-1))
                else:
                    coords.append(np.array(b).reshape(-1))
            if len(coords) > 0:
                coords = np.vstack(coords)
            else:
                coords = np.zeros((0, 4))
    except Exception:
        coords = np.zeros((0, 4))

    if isinstance(coords, np.ndarray) and coords.size > 0:
        # assume coords are in xyxy pixel coordinates
        for box in coords:
            if len(box) < 4:
                continue
            x0, y0, x1, y1 = map(int, map(round, box[:4]))
            # clip
            x0 = max(0, min(img_w - 1, x0))
            x1 = max(0, min(img_w - 1, x1))
            y0 = max(0, min(img_h - 1, y0))
            y1 = max(0, min(img_h - 1, y1))
            if x1 > x0 and y1 > y0:
                cv2.rectangle(pred_mask, (x0, y0), (x1, y1), 1, thickness=-1)

    return pred_mask


def compute_iou(gt_mask, pred_mask):
    gt = gt_mask.astype(bool)
    pr = pred_mask.astype(bool)
    inter = np.logical_and(gt, pr).sum()
    union = np.logical_or(gt, pr).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def draw_contours_on_image(image, pred_mask, gt_mask=None):
    out = image.copy()
    # draw predicted contours in green
    pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        pred_mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    # optionally draw GT contours in red (thin)
    if gt_mask is not None:
        gt_mask_uint8 = (gt_mask * 255).astype(np.uint8)
        contours_g, _ = cv2.findContours(
            gt_mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(out, contours_g, -1, (0, 0, 255), 1)
    return out


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(repo_root, 'detection_data', 'valid')
    images_dir = os.path.join(data_root, 'images')
    labels_dir = os.path.join(data_root, 'labels')

    out_dir = os.path.join(repo_root, 'outputs', 'detection_visualizations')
    os.makedirs(out_dir, exist_ok=True)
    iou_csv = os.path.join(repo_root, 'outputs', 'detection_ious.csv')

    model_path = os.path.join(repo_root, 'yolov8n_model.pt')
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    print('Loading model...')
    model = YOLO(model_path)

    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = sorted(
        [
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith(image_extensions)
        ]
    )

    with open(iou_csv, 'w', encoding='utf-8') as outf:
        outf.write('image,iou\n')
        for img_name in tqdm(image_files, desc='Images'):
            img_path = os.path.join(images_dir, img_name)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_name)

            img = cv2.imread(img_path)
            if img is None:
                print('Could not read', img_path)
                continue
            h, w = img.shape[:2]

            # gt_mask = read_label_polygons(label_path, w, h)
            gt_mask = read_label_bboxes(label_path, w, h, fmt='yolo')

            # run model prediction for this image
            try:
                results = model.predict(img_path, conf=0.25, verbose=False)
            except Exception as e:
                print('Model prediction failed for', img_name, e)
                pred_mask = np.zeros((h, w), dtype=np.uint8)
            else:
                if len(results) == 0:
                    pred_mask = np.zeros((h, w), dtype=np.uint8)
                else:
                    # pred_mask = masks_from_result(results[0], h, w)
                    pred_mask = masks_from_result_boxes(results, h, w)

            iou = compute_iou(gt_mask, pred_mask)
            outf.write(f"{img_name},{iou:.6f}\n")

            vis = draw_contours_on_image(img, pred_mask, gt_mask=gt_mask)
            out_path = os.path.join(out_dir, img_name)
            cv2.imwrite(out_path, vis)

    print('Done. Visualizations saved to', out_dir)
    print('IoU CSV saved to', iou_csv)


if __name__ == '__main__':
    main()
