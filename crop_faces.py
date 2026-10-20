# -*- coding: utf-8 -*-
"""
recrop-from-video.py —— 从视频重裁「头肩像」训练集（LoRA 用）

为什么要有这个脚本：
  上一批数据是 512x512 的源，裁到 1024 后脸虽然够大，但细节是插值出来的（偏软）。
  如果手上还有**原始视频**，就该按原分辨率重新解帧 —— 这样才能拿到真高清。
  本脚本直接从视频里解帧、检测人脸、按"脸占画面比例"裁出头肩像并缩放到 1024。

用法（先用 --scan-only 看能产出多少，再决定跑全量）:

  # ① 先扫描 10 秒，估算"这个视频能产出多少张、脸多大"
  python recrop-from-video.py --video D:\\x.mp4 --scan-only

  # ② 满意了再全量导出
  python recrop-from-video.py --video D:\\x.mp4 --out F:\\DSH\\...\\face_video_crop

  # 多集/多个视频
  python recrop-from-video.py --dir D:\\新建文件夹 --out ... --every 15

关键参数:
  --every N        每 N 帧取一帧（视频 30fps 时，15 ≈ 每 0.5 秒一张）
  --target-face    脸占输出画面的高度比例（默认 0.36，落在理想区 25~45%）
  --size           输出边长（默认 1024）
  --min-face       脸高下限（源视频像素，默认 120）→ 太小的直接不要
  --max-dup        与上一张的感知哈希距离 <= 此值视为重复，跳过
  --sharp-min      清晰度下限（拉普拉斯方差），挡掉运动模糊
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, 'models', 'face_detection_yunet_2023mar.onnx')
DET = DEFAULT_MODEL


def resolve_model(path=''):
    """返回可用的 YuNet 模型路径；缺失时给出下载提示。"""
    cand = path or DET
    if os.path.isfile(cand):
        return cand
    raise SystemExit(
        '找不到人脸检测模型: %s\n'
        '请从 OpenCV Zoo 下载 face_detection_yunet_2023mar.onnx 放到 models/ 目录，'
        '或用 --model 指定路径。\n'
        'https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet' % cand)


def log(*a):
    print(*a, flush=True)


def open_capture(path):
    """打开视频，默认后端失败时依次尝试 FFMPEG / MSMF / DSHOW。

    某些环境下默认后端会把纯数字文件名当成图像序列 pattern（报
    VIDEOIO(CV_IMAGES) icvExtractPattern 断言失败），或缺少对应解码器；
    显式指定后端可以绕开这类自动推断问题。
    """
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        return cap
    cap.release()
    for flag_name in ('CAP_FFMPEG', 'CAP_MSMF', 'CAP_DSHOW'):
        flag = getattr(cv2, flag_name, None)
        if flag is None:
            continue
        cap = cv2.VideoCapture(path, flag)
        if cap.isOpened():
            log(f'  （默认后端打不开，已改用 {flag_name[4:]} 后端）')
            return cap
        cap.release()
    return cv2.VideoCapture()


def detect_face(img, min_ratio=0.004):
    """多尺度检测最大人脸。返回 (x,y,w,h) 或 None。"""
    h, w = img.shape[:2]
    for scale in (1.0, 0.7, 0.5, 0.35, 0.25):
        sw, sh = max(64, int(w * scale)), max(64, int(h * scale))
        small = img if scale == 1.0 else cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        d = cv2.FaceDetectorYN.create(DET, '', (sw, sh), 0.5, 0.3, 5000)
        d.setInputSize((sw, sh))
        _, faces = d.detect(small)
        if faces is None or not len(faces):
            continue
        fc = max(faces, key=lambda r: r[2] * r[3]).copy()
        if scale != 1.0:
            fc[0] /= scale; fc[1] /= scale; fc[2] /= scale; fc[3] /= scale
        if (fc[2] * fc[3]) / (w * h) < min_ratio:
            continue
        return float(fc[0]), float(fc[1]), float(fc[2]), float(fc[3])
    return None


def phash(img, size=16):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    d = cv2.dct(np.float32(g))
    low = d[:8, :8]
    return (low > np.median(low[1:])).flatten()


def crop_head_shoulder(img, box, size, target_face, out_face_min_px=None):
    """按目标「脸占画面高度比例」裁正方形。返回 (裁剪图, 脸在实际裁剪图里的高度px)。

    ⚠️ 两个实测踩过的坑（2026-09-17）：
      1) target_face 不能按"脸面积占比"理解 —— 它约束的是【脸高 / 裁剪边长】。
         人脸通常高>宽（约 1.3:1），所以裁剪边长必须 ≥ 脸高 × 1.3 才不会切脸。
         实测把 target_face 设 0.36 时，裁出来的脸只占画面 16%，且切到了下巴。
         → 现在同时取 max(脸高/target_face, 脸高×1.45, 脸宽×1.9)，保证有足够留白。
      2) 越界填充【不能用镜像/反射】—— 会在边上留下明显的镜像鬼影，
         被模型学成"画面边缘重复"这种特征。
         → 改成 BORDER_REPLICATE（用边缘像素延伸），观感自然得多。
    """
    fx, fy, fw, fh = box
    # 需要的裁剪边长（源像素）
    side = max(
        fh / max(0.05, target_face),   # 按目标比例
        fh * 1.45,                     # 至少给脸高 1.45 倍，避免切下巴/头顶
        fw * 1.90,                     # 脸部宽度的近两倍，横向留白
    )
    cx = fx + fw / 2.0
    cy = fy + fh / 2.0
    # 人像通常希望脸偏上（头顶留一点、下巴下面多点）—— 把中心略往下移。
    # 偏移不能太大，否则头顶会被切掉（实测 0.10 时头顶切了，改 0.06）。
    cy += fh * 0.06

    x0 = int(round(cx - side / 2)); y0 = int(round(cy - side / 2))
    x1 = int(round(cx + side / 2)); y1 = int(round(cy + side / 2))

    H, W = img.shape[:2]
    pad_l = max(0, -x0); pad_t = max(0, -y0)
    pad_r = max(0, x1 - W); pad_b = max(0, y1 - H)
    pad_total = pad_l + pad_t + pad_r + pad_b
    # 越界太多就放弃这张 —— 填充比例高说明脸已经贴边（多为已裁过的近景），
    # 硬裁会得到大片拉伸的条纹/糊边，那种图进数据集是负资产。
    if pad_total > side * 1.2:
        return None, 0
    if pad_l or pad_t or pad_r or pad_b:
        # 边界复制（不是镜像）—— 避免鬼影
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
        x0 += pad_l; x1 += pad_l; y0 += pad_t; y1 += pad_t
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None, 0
    interp = cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_CUBIC
    crop = cv2.resize(crop, (size, size), interpolation=interp)
    face_h_out = fh * (size / float(y1 - y0))
    return crop, face_h_out


def iter_videos(path):
    if os.path.isdir(path):
        exts = ('*.mp4', '*.mkv', '*.mov', '*.avi', '*.flv', '*.wmv', '*.webm', '*.ts', '*.m4v')
        out = []
        for e in exts:
            out += glob.glob(os.path.join(path, e))
            out += glob.glob(os.path.join(path, '**', e), recursive=True)
        return sorted(set(out))
    return [path]


def process_video(v, a, scan_only=False):
    cap = open_capture(v)
    if not cap.isOpened():
        log(f'[x] 打不开: {v}')
        log('    可能原因：视频编码不受支持（如 H.265/AV1），或当前 OpenCV 构建缺少 FFmpeg 后端。')
        log('    建议：换用官方 opencv-python wheel（自带 FFmpeg）；或先把视频转成 H.264/MP4 再试。')
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    dur = total / fps if fps else 0
    log('=' * 78)
    log(f'视频: {os.path.basename(v)}')
    log(f'  分辨率 {w}x{h}   {fps:.2f} fps   {total} 帧   约 {dur:.1f} 秒')
    log(f'  采样: 每 {a.every} 帧取一帧  → 最多 {total // max(1, a.every)} 个候选')

    if scan_only:
        # 只扫前 a.scan_seconds 秒，估算产出率
        cap.release()
        return dict(path=v, w=w, h=h, fps=fps, total=total, dur=dur)

    name = os.path.splitext(os.path.basename(v))[0]
    outdir = os.path.join(a.out, name)
    os.makedirs(outdir, exist_ok=True)

    idx = -1
    kept = skipped = 0
    face_heights = []
    meta = []
    last_ph = None

    while True:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        if idx % max(1, a.every) != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            continue

        box = detect_face(frame)
        if box is None:
            skipped += 1
            continue
        fw, fh = box[2], box[3]
        if fh < a.min_face:
            skipped += 1
            continue

        crop, face_h_out = crop_head_shoulder(frame, box, a.size, a.target_face)
        if crop is None:
            skipped += 1
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        bright = float(gray.mean())
        if sharp < a.sharp_min or bright < 40 or bright > 230:
            skipped += 1
            continue

        ph = phash(crop)
        if last_ph is not None and int(np.count_nonzero(ph != last_ph)) <= a.max_dup:
            skipped += 1
            continue
        last_ph = ph

        kept += 1
        face_heights.append(face_h_out)
        fn = f'{name}_f{idx:06d}.jpg'
        path = os.path.join(outdir, fn)
        cv2.imwrite(path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), a.jpeg_quality])
        meta.append(dict(file=fn, frame=idx, face_h=round(float(face_h_out), 1),
                         src_face_h=round(float(fh), 1), sharp=round(sharp, 1), bright=round(bright, 1)))

        if kept % 25 == 0:
            log(f'  ... 已导出 {kept} 张（跳过 {skipped}）  当前脸高 {face_h_out:.0f}px')

    cap.release()

    with open(os.path.join(outdir, '_meta.json'), 'w', encoding='utf-8') as fp:
        json.dump(dict(video=v, w=w, h=h, fps=fps, every=a.every, size=a.size,
                       target_face=a.target_face, kept=kept, skipped=skipped,
                       items=meta), fp, ensure_ascii=False, indent=2)

    log(f'  → 导出 {kept} 张，跳过 {skipped} 张')
    if face_heights:
        log(f'  → 输出脸高: 中位 {int(np.median(face_heights))}px  '
            f'范围 {int(min(face_heights))}~{int(max(face_heights))}px')
    log(f'  → 目录: {outdir}')
    return dict(path=v, w=w, h=h, kept=kept, skipped=skipped,
                face_median=float(np.median(face_heights)) if face_heights else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default='')
    ap.add_argument('--dir', default='')
    ap.add_argument('--out', default='output', help='导出目录（默认 ./output）')
    ap.add_argument('--model', default='', help='YuNet ONNX 模型路径（默认 models/face_detection_yunet_2023mar.onnx）')
    ap.add_argument('--every', type=int, default=15, help='每 N 帧取一帧')
    ap.add_argument('--target-face', type=float, default=0.55,
                    help='脸高占输出画面的比例。输出脸高 = size × 该值。'
                         '0.55 → 1024 图里脸高约 563px（头肩像，推荐）；'
                         '0.36 → 脸高约 368px（偏小，构图太宽）')
    ap.add_argument('--size', type=int, default=1024)
    ap.add_argument('--min-face', type=float, default=120, help='源视频里脸高下限(px)')
    ap.add_argument('--max-dup', type=int, default=5, help='感知哈希去重阈值')
    ap.add_argument('--sharp-min', type=float, default=40.0, help='清晰度下限')
    ap.add_argument('--jpeg-quality', type=int, default=95)
    ap.add_argument('--scan-only', action='store_true', help='只看视频信息，不解帧')
    ap.add_argument('--limit-videos', type=int, default=0)
    a = ap.parse_args()

    global DET
    DET = resolve_model(a.model)

    src = a.video or a.dir
    if not src:
        log('用法: --video <文件> 或 --dir <目录>')
        sys.exit(2)

    vids = iter_videos(src)
    if a.limit_videos:
        vids = vids[:a.limit_videos]
    if not vids:
        log('没找到视频文件:', src)
        sys.exit(1)

    log(f'找到 {len(vids)} 个视频')
    for v in vids:
        log('  ' + os.path.basename(v))
    log('')

    if not a.scan_only:
        os.makedirs(a.out, exist_ok=True)

    results = []
    for v in vids:
        r = process_video(v, a, scan_only=a.scan_only)
        if r:
            results.append(r)

    if a.scan_only:
        log('')
        log('=' * 78)
        log('扫描结果（估算，未解帧）:')
        tot = 0
        for r in results:
            est = r['total'] // max(1, a.every)
            tot += est
            log(f"  {os.path.basename(r['path'])}: {r['w']}x{r['h']}  {r['dur']:.0f}秒  "
                f"约 {est} 个候选帧")
        log('')
        log(f'合计候选帧约 {tot} 张（实际产出要看人脸检出率，通常 40%~80%）')
        log(f'建议：先跑一个视频试，确认脸高和数量满意再跑全部。')
        log(f'命令示例：')
        log(f'  python {os.path.basename(__file__)} --video "<第一个视频>" --out "{a.out}" --every {a.every}')
        return

    log('')
    log('=' * 78)
    log('汇总:')
    tot_kept = sum(r.get('kept', 0) for r in results)
    for r in results:
        if 'kept' in r:
            log(f"  {os.path.basename(r['path'])}: 导出 {r['kept']} 张，脸部高度中位 {r.get('face_median', 0):.0f}px")
    log(f'合计导出 {tot_kept} 张 → {a.out}')


if __name__ == '__main__':
    main()
