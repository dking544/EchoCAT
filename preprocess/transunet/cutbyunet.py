#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import unicodedata
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------- 配置（只需修改这三项路径） ----------------
original_root_folder = r''   # 源图像（允许包含子文件夹）
segmented_root_folder = r''  # 掩码目录（允许包含子文件夹）
output_root_folder = r''  # 输出目录（将保留掩码的相对目录结构）
# -------------------------------------------------------

SUPPORTED_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
PADDING_PIXELS = 0   # 在掩码尺度上对 bbox 扩展的像素；这里你要最小包围框，默认0
COVERAGE = 0.70      # 要包含的白色像素比例（0.90 = 包含 90% 的白色像素，允许丢弃最多 10%）

# 创建输出根目录
os.makedirs(output_root_folder, exist_ok=True)

# ---------- 辅助函数 ----------
def decode_escaped_name(name: str) -> str:
    def repl_hex(m):
        hx = m.group(1)
        try:
            return chr(int(hx, 16))
        except:
            return m.group(0)
    s = name
    s = re.sub(r'#U([0-9A-Fa-f]{4,6})', repl_hex, s)
    s = re.sub(r'%u([0-9A-Fa-f]{4,6})', repl_hex, s)
    s = re.sub(r'\\u([0-9A-Fa-f]{4,6})', repl_hex, s)
    s = re.sub(r'U\+([0-9A-Fa-f]{4,6})', repl_hex, s)
    return s

def normalize_key(s: str) -> str:
    return unicodedata.normalize('NFKC', s).lower()

def safe_filename(fname: str) -> str:
    base = os.path.basename(fname)
    base = re.sub(r'[\\/]+', '_', base)
    base = re.sub(r'[:\*\?"<>\|]+', '_', base)
    base = base.strip()
    if not base:
        base = "unnamed"
    return base

def iter_image_files(root_folder: str):
    files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for fn in filenames:
            low = fn.lower()
            if any(low.endswith(ext) for ext in SUPPORTED_EXTS):
                files.append(os.path.join(dirpath, fn))
    return files

# ---------- 构建源图映射（递归） ----------
# 我们构建三个索引：
# 1) orig_rel_map: 相对于 original_root_folder 的“相对路径（无扩展名）” -> [paths]
# 2) orig_stem_map: 文件 stem -> [paths] （fallback）
# 3) orig_fullname_map: 完整文件名（含扩展） -> path（若唯一）

orig_rel_map = {}    # key: normalize_key(rel_path_noext) -> list(paths)
orig_stem_map = {}   # key: normalize_key(stem) -> list(paths)
orig_fullname_map = {}  # key: normalize_key(filename) -> list(paths)

if os.path.isdir(original_root_folder):
    all_orig_files = iter_image_files(original_root_folder)
    for fullpath in all_orig_files:
        rel = os.path.relpath(fullpath, original_root_folder)
        rel_noext = os.path.splitext(rel)[0]
        key_rel = normalize_key(rel_noext.replace('\\', '/'))
        orig_rel_map.setdefault(key_rel, []).append(fullpath)

        fname = os.path.basename(fullpath)
        stem = os.path.splitext(fname)[0]
        key_stem = normalize_key(stem)
        orig_stem_map.setdefault(key_stem, []).append(fullpath)

        key_full = normalize_key(fname)
        orig_fullname_map.setdefault(key_full, []).append(fullpath)
else:
    print(f"警告：original_root_folder 不是目录：{original_root_folder}")

def _path_component_score(seg_components, orig_components):
    # 简单评分：统计相同组件（不区分顺序），并加权末尾匹配
    s = 0
    seg_set = set(seg_components)
    orig_set = set(orig_components)
    s += len(seg_set & orig_set)
    # 末尾连续相同的组件给额外分数（路径层级相似性）
    rev_seg = list(reversed(seg_components))
    rev_orig = list(reversed(orig_components))
    bonus = 0
    for a, b in zip(rev_seg, rev_orig):
        if a == b:
            bonus += 2
        else:
            break
    s += bonus
    return s

def find_original_for_segment(segmented_full_path: str):
    """
    根据 segmented 文件路径尝试逐步匹配原图：
    1) 优先：相同的相对路径（不考虑扩展名），例如 seg: segroot/.../a/b.png -> try origroot/.../a/b.*
    2) 回退：按 stem 在 orig_stem_map 中查找；若有多候选，按路径成分相似度打分选择最优。
    3) 回退2：按完整文件名匹配 orig_fullname_map（若存在）
    """
    # 1. 准备相对路径 key
    try:
        rel_seg = os.path.relpath(segmented_full_path, segmented_root_folder)
    except Exception:
        rel_seg = os.path.basename(segmented_full_path)
    rel_seg_noext = os.path.splitext(rel_seg)[0].replace('\\', '/')
    key_rel_seg = normalize_key(rel_seg_noext)

    # 优先尝试：直接把相对路径替换到 original_root_folder（允许不同扩展）
    for ext in SUPPORTED_EXTS:
        cand = os.path.join(original_root_folder, rel_seg_noext + ext)
        if os.path.exists(cand):
            return cand
    # 也尝试任何存在于 orig_rel_map 的匹配（忽略扩展）
    if key_rel_seg in orig_rel_map:
        lst = orig_rel_map[key_rel_seg]
        if len(lst) == 1:
            return lst[0]
        # 多候选时按路径相似度选择
        seg_comps = [normalize_key(c) for c in rel_seg_noext.split('/')]
        best = None
        best_score = -1
        for p in lst:
            relp = os.path.relpath(p, original_root_folder)
            relp_noext = os.path.splitext(relp)[0].replace('\\', '/')
            orig_comps = [normalize_key(c) for c in relp_noext.split('/')]
            score = _path_component_score(seg_comps, orig_comps)
            if score > best_score:
                best_score = score
                best = p
        if best is not None:
            return best

    # 2. 按 stem 回退匹配
    seg_fname = os.path.basename(segmented_full_path)
    seg_stem = os.path.splitext(seg_fname)[0]
    key_stem = normalize_key(seg_stem)
    candidates = orig_stem_map.get(key_stem, [])
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # 用目录/路径成分相似度来选最佳候选
        seg_rel_comps = [normalize_key(c) for c in rel_seg_noext.split('/')]
        best = None
        best_score = -1
        for p in candidates:
            relp = os.path.relpath(p, original_root_folder)
            relp_noext = os.path.splitext(relp)[0].replace('\\', '/')
            orig_comps = [normalize_key(c) for c in relp_noext.split('/')]
            score = _path_component_score(seg_rel_comps, orig_comps)
            if score > best_score:
                best_score = score
                best = p
        # 如果评分过低（例如都为0）我们仍返回 best，但打印警告
        if best is not None:
            if best_score <= 0:
                print(f"[警告] stem 匹配到多文件，但路径相似度低，选第一个：stem={seg_stem}, seg={segmented_full_path}")
            return best

    # 3. 按完整文件名回退（极少用）
    key_full = normalize_key(seg_fname)
    lst_full = orig_fullname_map.get(key_full, [])
    if len(lst_full) == 1:
        return lst_full[0]
    elif len(lst_full) > 1:
        # 同样按路径相似度选
        seg_rel_comps = [normalize_key(c) for c in rel_seg_noext.split('/')]
        best = None
        best_score = -1
        for p in lst_full:
            relp = os.path.relpath(p, original_root_folder)
            relp_noext = os.path.splitext(relp)[0].replace('\\', '/')
            orig_comps = [normalize_key(c) for c in relp_noext.split('/')]
            score = _path_component_score(seg_rel_comps, orig_comps)
            if score > best_score:
                best_score = score
                best = p
        if best is not None:
            print(f"[警告] 完整文件名匹配到多文件，选最佳匹配: {best} (score={best_score})")
            return best

    # 都没找到
    return None

def build_safe_output_path(segmented_full_path: str):
    rel = os.path.relpath(segmented_full_path, segmented_root_folder)
    comps = rel.split(os.sep)
    safe_comps = [safe_filename(c) for c in comps]
    out_path = os.path.join(output_root_folder, *safe_comps)
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    # print(f"[构建输出路径] {segmented_full_path} -> {out_path}")
    return out_path

# ---------- 主处理流程（以掩码中白色像素 255 为前景） ----------
summary = {'processed': 0, 'skipped_no_mask': 0, 'skipped_no_orig': 0, 'skipped_invalid_crop': 0, 'errors': 0}

seg_files = iter_image_files(segmented_root_folder)
if not seg_files:
    print("警告：segmented_root_folder 中未找到支持的掩码文件，请确认路径与扩展名。")

for segmented_path in tqdm(seg_files, desc="处理掩码"):
    segmented_filename = os.path.basename(segmented_path)
    stem = os.path.splitext(segmented_filename)[0]

    try:
        seg_img = Image.open(segmented_path).convert('L')
        seg_arr = np.array(seg_img)

        if 255 in np.unique(seg_arr):
            mask = (seg_arr == 255)
        else:
            mask = (seg_arr > 127)

        ys, xs = np.where(mask)
        if ys.size == 0:
            print(f"[跳过] 掩码中没有白色/前景像素: {segmented_path}")
            summary['skipped_no_mask'] += 1
            continue

        # ---------- 使用百分位数取中心 coverage 区间 ----------
        cov_pct = float(COVERAGE) * 100.0
        low_pct = (100.0 - cov_pct) / 2.0
        high_pct = 100.0 - low_pct

        MIN_PIXELS_FOR_PERCENTILE = 10
        if ys.size < MIN_PIXELS_FOR_PERCENTILE:
            left, upper, right, lower = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        else:
            lx = int(np.floor(np.percentile(xs, low_pct)))
            rx = int(np.ceil(np.percentile(xs, high_pct))) + 1
            uy = int(np.floor(np.percentile(ys, low_pct)))
            ly = int(np.ceil(np.percentile(ys, high_pct))) + 1

            if rx <= lx or ly <= uy:
                left, upper, right, lower = xs.min(), ys.min() + 0, xs.max() + 1, ys.max() + 1
            else:
                left, upper, right, lower = lx, uy, rx, ly

        left = max(0, left - PADDING_PIXELS)
        upper = max(0, upper - PADDING_PIXELS)
        right = min(seg_img.width, right + PADDING_PIXELS)
        lower = min(seg_img.height, lower + PADDING_PIXELS)

        if right <= left or lower <= upper:
            print(f"[跳过] 无效掩码裁剪区域: {segmented_path}")
            summary['skipped_invalid_crop'] += 1
            continue

        # ---------- 匹配原图并映射到原图坐标 ----------
        original_path = find_original_for_segment(segmented_path)
        if original_path is None or not os.path.exists(original_path):
            print(f"[跳过] 原始图像不存在: stem={stem}  (seg: {segmented_path})")
            summary['skipped_no_orig'] += 1
            continue

        orig_img = Image.open(original_path)
        orig_w, orig_h = orig_img.size
        seg_w, seg_h = seg_img.size
        sx = orig_w / seg_w
        sy = orig_h / seg_h

        crop_x1 = int(left * sx)
        crop_y1 = int(upper * sy)
        crop_x2 = int(right * sx)
        crop_y2 = int(lower * sy)

        crop_x1 = max(0, min(crop_x1, orig_w - 1))
        crop_y1 = max(0, min(crop_y1, orig_h - 1))
        crop_x2 = max(0, min(crop_x2, orig_w))
        crop_y2 = max(0, min(crop_y2, orig_h))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            print(f"[跳过] 无效裁剪区域（映射到原图后）: seg={segmented_path} -> orig={original_path}")
            summary['skipped_invalid_crop'] += 1
            continue

        cropped = orig_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # ---------- 准备输出并保存 ----------
        output_path = build_safe_output_path(segmented_path)
        # print(f"[保存] {segmented_path} -> {output_path}")
        ext = os.path.splitext(output_path)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            if cropped.mode in ('RGBA', 'LA') or ('transparency' in cropped.info):
                bg = Image.new('RGB', cropped.size, (255, 255, 255))
                if cropped.mode == 'RGBA':
                    bg.paste(cropped, mask=cropped.split()[3])
                elif cropped.mode == 'LA':
                    rgba = cropped.convert('RGBA')
                    bg.paste(rgba, mask=rgba.split()[3])
                else:
                    bg.paste(cropped.convert('RGBA'), mask=cropped.convert('RGBA').split()[3])
                bg.save(output_path, quality=95)
            else:
                cropped.convert('RGB').save(output_path, quality=95)
        else:
            cropped.save(output_path)

        summary['processed'] += 1

    except Exception as e:
        print(f"[错误] 处理失败: {segmented_path} -> {e}")
        summary['errors'] += 1

print("处理完成。统计：", summary)
