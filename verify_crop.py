# -*- coding: utf-8 -*-
"""verify_crop.py —— 直接验证裁剪数学（不经过视频，排除解码干扰）

为什么需要：用合成视频测试时，反复缩放会让人脸检测不稳，产出不可读，
            容易误判成"脚本有问题"。直接拿静态图片调裁剪函数、打印算出的框，
            才能确认逻辑本身对不对。

用法：
  python verify_crop.py [<图1> <图2> ...] [--target-face 0.55] [--size 1024]
  不给图则尝试读取同目录 samples/ 下的图片。
"""
import glob
import importlib.util
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))


def find_crop_script():
    """在同目录找裁剪脚本 —— 按关键字匹配，避免写死文件名。"""
    cands = []
    for f in os.listdir(HERE):
        if not f.endswith('.py') or f == os.path.basename(__file__):
            continue
        low = f.lower()
        if 'crop' in low or 'recrop' in low:
            cands.append(os.path.join(HERE, f))
    cands.sort(key=lambda p: (0 if p.endswith('crop_faces.py') else 1))
    return cands[0] if cands else None


def load_module(path, name='crop_mod'):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    target = 0.55
    size = 1024
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == '--target-face' and i + 1 < len(argv):
            target = float(argv[i + 1])
        if a == '--size' and i + 1 < len(argv):
            size = int(argv[i + 1])

    script = find_crop_script()
    if not script:
        print('❌ 同目录下没找到裁剪脚本（文件名应含 crop）')
        print('   当前目录:', HERE)
        sys.exit(1)
    print(f'使用脚本: {os.path.basename(script)}')
    m = load_module(script)

    files = [a for a in argv if not a.startswith('--') and os.path.exists(a)]
    if not files:
        d = os.path.join(HERE, 'samples')
        if os.path.isdir(d):
            files = sorted(glob.glob(os.path.join(d, '*.jpg')) + glob.glob(os.path.join(d, '*.png')))[:6]
    if not files:
        print('没有可用的样本图：传入图片路径，或把图片放进 samples/ 目录')
        sys.exit(1)

    print(f'目标: 输出 {size}x{size} · target-face={target}（脸高应 = {int(size * target)}px）\n')
    print(f"{'图':36s} {'源画面':>11s} {'源脸高':>7s} {'裁剪边长':>9s} {'输出脸高':>9s} {'脸高/画面':>9s}  判定")
    print('-' * 106)

    for f in files:
        img = cv2.imread(f)
        name = os.path.basename(f)[:36]
        if img is None:
            print(f'{name:36s}  读不了')
            continue
        box = m.detect_face(img)
        if box is None:
            print(f'{name:36s} {f"{img.shape[1]}x{img.shape[0]}":>11s}  未检出人脸')
            continue
        fx, fy, fw, fh = box
        crop, face_out = m.crop_head_shoulder(img, box, size, target)
        side = max(fh / target, fh * 1.45, fw * 1.90)
        if crop is None:
            print(f'{name:36s} {f"{img.shape[1]}x{img.shape[0]}":>11s} {int(fh):7d} {int(side):9d} '
                  f'{"—":>9s} {"—":>9s}  ⛔ 越界太多已拒绝')
            continue
        pct = 100.0 * face_out / size
        mark = '✅ 合格' if 25 <= pct <= 60 else ('⚠️ 偏小' if pct < 25 else '⚠️ 偏大')
        print(f'{name:36s} {f"{img.shape[1]}x{img.shape[0]}":>11s} {int(fh):7d} {int(side):9d} '
              f'{int(face_out):9d} {pct:8.1f}%  {mark}')

    print()
    print('判读：')
    print('  · 输出脸高应精确等于 size × target-face（脚本按定义保证）')
    print('  · 脸高/画面 落在 45~60% 是标准头肩像（对应脸面积占比约 20~30%）')
    print('  · 「越界太多已拒绝」表示人脸贴近画面边缘，裁出来会有大片填充 —— 跳过是对的')


if __name__ == '__main__':
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass
    main()
