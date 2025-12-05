"""
批量图像增强脚本
此脚本遍历给定 `data` 目录下的所有 `**/images/*` 文件，
对每张图片应用同态滤波增强，并调用 `save_image` 保存对比图，
以及调用 `evaluate_image` 写入评估 CSV（由 `image_enhancement` 的 config 控制路径）。

用法示例（PowerShell）：
  $env:PYTHONPATH = 'D:\Temp\code\imgReso'; .\.venv\Scripts\Activate.ps1; python -m resolution.enhance.enhance --data-dir data --limit 10

此模块依赖 `resolution.enhance.image_enhancement` 中的函数：
  - homomorphic_filter(image, gammaH, gammaL, c, D0)
  - save_image(src, result, image_path, output_path=None)
  - evaluate_image(original, enhanced, filename)

"""
from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import csv
import math
import traceback
import tempfile
import cv2

from . import image_enhancement as ie


def find_image_files(data_dir: Path):
    """在 data_dir 下查找所有名为 images 的子目录，并返回图片路径列表。"""
    image_paths = []
    for root, dirs, files in os.walk(data_dir):
        if Path(root).name.lower() == 'images':
            for fn in files:
                if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                    image_paths.append(Path(root) / fn)
    return sorted(image_paths)


def process_image(p: Path, params: dict, dry_run=False, compare=False):
    try:
        img = Image.open(p)
    except Exception as e:
        print(f"无法打开图片 {p}: {e}")
        return False, None, None, None

    try:
        # Merge tuned enhance parameters from config with any CLI-provided homomorphic params.
        enhance_params = ie.CONFIG.get('enhance', {}).copy()
        # CLI `params` may contain homomorphic-specific keys (gammaH/gammaL/c/D0)
        if isinstance(params, dict):
            # Override or add any CLI-specified homomorphic params
            enhance_params.update(params)

        # 调用 auto_enhance（自动检测 + 自适应增强），返回 (enhanced, detection_dict)
        enhanced, det = ie.auto_enhance(img, params=enhance_params)

        # 根据原始图片路径推断 dataset 名称（期望结构： data/<dataset>/images/... ）
        dataset = 'default'
        try:
            # p.parent is images, p.parent.parent is dataset dir
            if p.parent.name.lower() == 'images' and p.parent.parent is not None:
                dataset = p.parent.parent.name
            else:
                dataset = p.parent.name
        except Exception:
            dataset = 'default'

        # 构造按 dataset 分流的输出路径，例如: results/<dataset>/images 和 results/<dataset>/csv
        images_dir_cfg = ie.CONFIG.get('results', {}).get('images_dir', 'results/images')
        csv_dir_cfg = ie.CONFIG.get('results', {}).get('csv_dir', 'results/csv')
        base_images_root = os.path.dirname(ie._resolve_path(images_dir_cfg))
        base_csv_root = os.path.dirname(ie._resolve_path(csv_dir_cfg))
        dataset_images_dir = os.path.join(base_images_root, dataset, 'images')
        dataset_csv_dir = os.path.join(base_csv_root, dataset, 'csv')

        # 后处理（可选）：降噪与小连通域去噪
        try:
            pp_cfg = ie.CONFIG.get('postprocess', {}).get('denoise', {})
        except Exception:
            pp_cfg = {}

        try:
            if pp_cfg.get('enable', True):
                enhanced = ie.post_denoise(enhanced, pp_cfg)
        except Exception:
            pass

        # 默认仅保存处理后的图像（save_processed_image），并写入到 dataset-specific images 目录
        if not dry_run:
            out_path = ie.save_processed_image(enhanced, str(p), output_path=dataset_images_dir)

        # 自动计算并记录 BRISQUE 分数（原始与增强后），将结果追加到 dataset-specific CSV
        try:
            os.makedirs(dataset_csv_dir, exist_ok=True)
            brisque_csv = os.path.join(dataset_csv_dir, 'brisque_results.csv')
            # compute_brisque_score accepts path or image; use path for original, tmp file for enhanced
            # default iou and enhanced brisque
            iou_v = float('nan')
            enh_b = float('nan')

            try:
                orig_b = ie.compute_brisque_score(str(p))
            except Exception:
                orig_b = float('nan')

            try:
                # Try direct scoring on the enhanced image object/array first
                # The BRISQUE wrapper accepts paths, PIL images or numpy arrays.
                enh_b = ie.compute_brisque_score(enhanced)
            except Exception as e:
                # Fallback: write a temporary file then score the file
                try:
                    fd, tmpf = tempfile.mkstemp(suffix='.png')
                    os.close(fd)
                    cv2.imwrite(tmpf, enhanced)
                    try:
                        enh_b = ie.compute_brisque_score(tmpf)
                    finally:
                        try:
                            os.remove(tmpf)
                        except Exception:
                            pass
                except Exception as e2:
                    print(f"Enhanced BRISQUE scoring failed for {p}: {e2}")
                    enh_b = float('nan')

            # Write header if new (增加 IoU 列)
            header = ['filename', 'image_path', 'original_brisque', 'enhanced_brisque', 'iou']
            write_header = not os.path.exists(brisque_csv)
            with open(brisque_csv, 'a', newline='', encoding='utf-8') as cf:
                writer = csv.writer(cf)
                if write_header:
                    writer.writerow(header)
                # compute IoU between masks from original and enhanced images (method configurable)
                try:
                    iou_cfg = ie.CONFIG.get('postprocess', {}).get('iou', {})
                    method = iou_cfg.get('method', 'otsu')
                    blur = int(iou_cfg.get('blur', 5))
                    min_area = int(iou_cfg.get('min_area', 20))
                    mask_orig = ie.mask_from_image(img, method=method, blur=blur, min_area=min_area)
                    mask_enh = ie.mask_from_image(enhanced, method=method, blur=blur, min_area=min_area)
                    iou_v = ie.compute_iou(mask_orig, mask_enh)
                except Exception:
                    iou_v = float('nan')

                writer.writerow([p.name, str(p), f"{float(orig_b) if not math.isnan(orig_b) else ''}", f"{float(enh_b) if not math.isnan(enh_b) else ''}", f"{float(iou_v) if not math.isnan(iou_v) else ''}"])
        except Exception:
            pass

        # 如果需要比较图与评估，可通过 compare=True 启用（可选）
        if compare and not dry_run:
            ie.save_image(img, enhanced, str(p), output_path=dataset_images_dir)
            ie.evaluate_image(img, enhanced, p.name, output_csv_dir=dataset_csv_dir)

        print(f"处理完成: {p}")
        return True, dataset, iou_v, enh_b
    except Exception as e:
        print(f"处理图片时出错: {p}\n{traceback.format_exc()}")
        return False, None, None, None


def main(argv=None):
    parser = argparse.ArgumentParser(description='批量增强 data 下的图像并输出（默认仅保存处理后图像），可选保存对比图与 CSV 评估')
    parser.add_argument('--data-dir', type=str, default='data', help='数据根目录（默认：data）')
    parser.add_argument('--limit', type=int, default=0, help='最多处理多少张图片（0 表示全部）')
    parser.add_argument('--dry-run', action='store_true', help='只列出将要处理的文件，不实际写文件')
    parser.add_argument('--compare', action='store_true', help='同时保存对比图并写入评估 CSV（默认不启用）')
    parser.add_argument('--gammaH', type=float, default=2.0, help='同态滤波 gammaH 参数')
    parser.add_argument('--gammaL', type=float, default=0.5, help='同态滤波 gammaL 参数')
    parser.add_argument('--c', type=float, default=1.0, help='同态滤波 c 参数')
    parser.add_argument('--D0', type=float, default=30.0, help='同态滤波 D0 参数')
    parser.add_argument('--skip-existing', action='store_true', help='若对比图已存在则跳过（基于文件名+时间戳策略可能无法精确检测）')

    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"指定的数据目录不存在: {data_dir}")
        sys.exit(1)

    files = find_image_files(data_dir)
    if not files:
        print(f"在 {data_dir} 未找到任何 images 子目录或图片文件。")
        return

    params = {
        'gammaH': args.gammaH,
        'gammaL': args.gammaL,
        'c': args.c,
        'D0': args.D0,
    }

    total = len(files)
    print(f"找到 {total} 张图片，开始处理（dry_run={args.dry_run}）。")

    # 可选：在每次运行前清理之前生成的处理后图像（默认启用）
    try:
        clear_prev = ie.CONFIG.get('results', {}).get('clear_previous', True)
    except Exception:
        clear_prev = True

    if clear_prev:
        try:
            images_dir_cfg = ie.CONFIG.get('results', {}).get('images_dir', 'results/images')
            base_images_root = os.path.dirname(ie._resolve_path(images_dir_cfg))
            if os.path.exists(base_images_root):
                removed = 0
                # 仅清理名为 'images' 的子目录中的文件，避免误删 csv 或其他数据
                for root, dirs, files_in in os.walk(base_images_root):
                    if Path(root).name.lower() == 'images':
                        for fn in files_in:
                            fp = os.path.join(root, fn)
                            try:
                                os.remove(fp)
                                removed += 1
                            except Exception:
                                pass
                print(f"已删除之前生成的 {removed} 个文件 (位于 {base_images_root} 下的 images 子目录)。")
        except Exception as e:
            print(f"清理先前生成的图像失败: {e}")

    # 如果配置要求，在开始 run 时清除旧的 brisque CSV 文件，避免追加到过期的结果
    try:
        reset_brisque = ie.CONFIG.get('results', {}).get('reset_brisque', True)
    except Exception:
        reset_brisque = True

    if reset_brisque:
        try:
            csv_dir_cfg = ie.CONFIG.get('results', {}).get('csv_dir', 'results/csv')
            base_csv_root = os.path.dirname(ie._resolve_path(csv_dir_cfg))
            removed_csv = 0
            if os.path.exists(base_csv_root):
                for root, dirs, files_in in os.walk(base_csv_root):
                    for fn in files_in:
                        if fn.lower() == 'brisque_results.csv':
                            fp = os.path.join(root, fn)
                            try:
                                os.remove(fp)
                                removed_csv += 1
                            except Exception:
                                pass
            if removed_csv:
                print(f"已移除旧的 brisque CSV 文件: {removed_csv} 个 (位于 {base_csv_root})。")
        except Exception as e:
            print(f"移除旧的 brisque CSV 时出现错误: {e}")
    n = 0
    brisque_stats = {}
    for p in files:
        if args.limit and n >= args.limit:
            break

        # 简单跳过检查：如果 skip_existing，且目标 images 目录已有同名文件（不能精确），则跳过
        if args.skip_existing:
            target_dir = Path(ie._resolve_path(ie.CONFIG.get('results', {}).get('images_dir', 'results/images')))
            # 目标文件名包含时间戳，所以无法直接判断，故这里不做复杂判断
            pass

        ok, ds, iou, enh_b = process_image(p, params, dry_run=args.dry_run, compare=args.compare)
        if ok:
            n += 1
            if ds:
                brisque_stats.setdefault(ds, []).append(enh_b)

    print(f"已完成：处理 {n} 张图片（从 {total} 张中）。")

    # 写入每个 dataset 的平均 enhanced BRISQUE 到 summary CSV
    try:
        csv_dir_cfg = ie.CONFIG.get('results', {}).get('csv_dir', 'results/csv')
        results_csv_root = os.path.dirname(ie._resolve_path(csv_dir_cfg))
        for ds, vals in brisque_stats.items():
            # 过滤掉 nan
            valid = [v for v in vals if not (v is None or (isinstance(v, float) and math.isnan(v)))]
            avg = float(np.mean(valid)) if valid else float('nan')
            ds_csv_dir = os.path.join(results_csv_root, ds, 'csv')
            os.makedirs(ds_csv_dir, exist_ok=True)
            summary_path = os.path.join(ds_csv_dir, 'summary.csv')
            try:
                with open(summary_path, 'w', newline='', encoding='utf-8') as sf:
                    w = csv.writer(sf)
                    w.writerow(['dataset', 'processed_images', 'avg_enhanced_brisque'])
                    w.writerow([ds, len(vals), f"{avg if not math.isnan(avg) else ''}"])
                print(f"已写入 {ds} 的平均 BRISQUE 至 {summary_path}: avg_enhanced_brisque={avg}")
            except Exception as e:
                print(f"写入 {ds} summary CSV 失败: {e}")
    except Exception:
        pass


if __name__ == '__main__':
    main()
