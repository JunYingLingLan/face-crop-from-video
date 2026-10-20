# face-crop-from-video

> 从**视频**里解帧、检测人脸、按「脸占画面高度比例」裁出头肩像，输出 1024² 样本图。
> 用于构建人脸数据集（CV 训练/微调素材预处理）——不依赖 ffmpeg，只用 OpenCV。

## 为什么需要它

如果手头只有一堆 512×512 的小图，放大到 1024 只会得到插值出来的糊边细节。
**只要还有原始视频，按原分辨率重新解帧**就是最有效的一步：同一张脸，细节是真实采样出来的。

本脚本把这件事做成一条命令：解帧 → 检测 → 按构图规则裁剪 → 质量过滤 → 去重 → 缩放导出，并写出逐张元数据。

## 特性

| 能力 | 说明 |
|---|---|
| **零外部依赖解码** | 用 OpenCV 的 `VideoCapture` 解帧，**不需要 ffmpeg** |
| **多尺度人脸检测** | 内置 OpenCV YuNet（`FaceDetectorYN`），对 1.0/0.7/0.5/0.35/0.25 五档尺度分别检测，取最大人脸，小脸也能抓到 |
| **构图可控** | `--target-face` 直接控制「输出图里脸高占边长的比例」，而不是让人猜裁剪框 |
| **质量过滤** | 拉普拉斯方差挡运动模糊、亮度区间挡过暗过曝、最小脸高挡远景 |
| **去重** | DCT 感知哈希（pHash），与上一张距离 ≤ `--max-dup` 视为重复帧跳过 |
| **先看再跑** | `--scan-only` 只读视频头，估算能产出多少张，不用真解帧 |
| **可追溯** | 每批输出附 `_meta.json`：帧号、源脸高、输出脸高、清晰度、亮度 |

## 安装

```bash
pip install -r requirements.txt
```

- 需要 `opencv-python >= 4.8`（`cv2.FaceDetectorYN` 从 4.5.4 起提供，4.8 起接口稳定）
- 本仓库已附带检测模型 `models/face_detection_yunet_2023mar.onnx`（来自 [OpenCV Zoo](https://github.com/opencv/opencv_zoo)，Apache-2.0），开箱即用；也可用 `--model` 指定自己的模型

### 已验证

Windows 11 + Python 3.12 + OpenCV 5.0 实跑通过：用 MJPG/AVI 合成视频（60 帧、轻微平移与亮度扰动）跑完整流程，
导出 6–7 张样本，`verify_crop.py` 判定全部合格——输出脸高 563 px（= 1024 × 0.55，占比 55.0%），
`_meta.json` 逐帧记录源脸高、清晰度与亮度。

## 用法

### ① 先扫描，估算产出

```bash
python crop_faces.py --video D:/sample.mp4 --scan-only
```

```text
视频: sample.mp4
  分辨率 1920x1080   30.00 fps   9000 帧   约 300.0 秒
  采样: 每 15 帧取一帧  → 最多 600 个候选
```

### ② 满意再全量导出

```bash
python crop_faces.py --video D:/sample.mp4 --out output --every 15 --target-face 0.55
```

### ③ 批量处理一个目录里的所有视频

```bash
python crop_faces.py --dir D:/videos --out output --every 15
```

输出按视频名分目录：

```text
output/
└── sample/
    ├── sample_f000015.jpg
    ├── sample_f000030.jpg
    └── _meta.json
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--video` / `--dir` | — | 单个视频文件，或包含视频的目录（递归，支持 mp4/mkv/mov/avi/flv/wmv/webm/ts/m4v） |
| `--out` | `output` | 导出根目录，每个视频一个子目录 |
| `--every` | `15` | 每 N 帧取一帧；30 fps 下 15 ≈ 每 0.5 秒一张 |
| `--target-face` | `0.55` | 脸高占输出边长的比例（0.55 → 1024 图里脸高约 563 px，标准头肩像） |
| `--size` | `1024` | 输出边长 |
| `--min-face` | `120` | 源视频中的脸高下限（px），太小的直接丢弃 |
| `--max-dup` | `5` | 感知哈希去重阈值（越小越严格） |
| `--sharp-min` | `40` | 清晰度下限（Laplacian 方差） |
| `--jpeg-quality` | `95` | 导出 JPEG 质量 |
| `--model` | `models/face_detection_yunet_2023mar.onnx` | YuNet ONNX 模型路径 |
| `--scan-only` | 关闭 | 只看视频信息与候选数量，不解帧 |
| `--limit-videos` | `0` | 只处理前 N 个视频（调试用，0 表示不限） |

## 构图规则（为什么不是简单的「按人脸框放大」）

裁剪边长取三者的最大值：

```python
side = max(
    face_h / target_face,   # 按目标比例
    face_h * 1.45,          # 至少给脸高 1.45 倍，避免切下巴/头顶
    face_w * 1.90,          # 脸部宽度的近两倍，横向留白
)
```

两个踩过的坑，写进代码注释也写在这里：

1. **`target_face` 约束的是「脸高 / 裁剪边长」，不是「脸面积占比」。**
   人脸通常高 > 宽（约 1.3:1），只按面积比例算会让裁剪边长不足，实测会出现切下巴。
2. **越界填充不能用镜像（`BORDER_REFLECT`）。**
   镜像会在画面边缘留下鬼影，被模型学成「边缘重复」特征。这里用 `BORDER_REPLICATE`（延伸边缘像素），观感自然。
   越界总长度超过 `side × 1.2` 直接放弃该帧——说明脸已贴边，硬裁只会得到大片糊边。

人脸中心额外下移 `0.06 × 脸高`：头顶留一点、下巴下方留更多，符合常见人像构图。

## 自检工具

`verify_crop.py` 直接用静态图片验证裁剪数学（不经过视频，排除解码干扰）：

```bash
python verify_crop.py img/a.jpg img/b.jpg --target-face 0.55 --size 1024
```

会打印每张图的源脸高、裁剪边长、输出脸高与占比，并给出「合格 / 偏小 / 偏大」判定。判据：输出脸高应精确等于 `size × target-face`；占比落在 45–60% 属标准头肩像。

## 常见问题

**`[x] 打不开: xxx.mp4`，日志里出现 `VIDEOIO(CV_IMAGES) ... icvExtractPattern`**

这是视频后端没能接管该文件、回落成「图像序列」解析导致的。依次排查：

1. **文件名是不是纯数字？** 部分 OpenCV 版本会把纯数字文件名当成图像序列 pattern（`img_%03d.jpg` 那种）。本脚本已内置 `FFMPEG → MSMF → DSHOW` 三级后端回退来绕过这类自动推断；
2. **编码是不是 H.265 / AV1？** 多数 OpenCV 的 FFmpeg 构建只带 H.264 解码，先转成 H.264 MP4 再试；
3. **换官方 `opencv-python` wheel**（自带 FFmpeg），避免使用裁剪过的自定义构建。

**导出数量远少于候选数？**

正常。检出率通常 40%–80%，再叠加最小脸高、清晰度、亮度、去重四道过滤。先跑 `--scan-only` 看候选数，再调 `--every`。

**脸被切下巴，或脸太大/太小？**

调 `--target-face`：切下巴说明脸占比过大，调小（如 0.45）；脸太小则调大（如 0.65）。注意该参数约束的是「脸高 / 输出边长」，不是面积占比。

**多人画面为什么只出一张？**

只保留最大人脸，不做人脸聚类——这是为单人数据集设计的。

## 已知限制

- DHT 之类的时序问题与本项目无关；这里的检测只处理彩色帧（不支持灰度/深度视频）
- 多人画面只保留**最大人脸**，不做人脸聚类
- 侧脸、遮挡、强逆光会漏检（YuNet 的检测阈值固定 0.5）
- 逐帧 `grab()` + 抽样 `retrieve()`，长视频耗时主要花在解码上，可先用 `--every` 粗采样

## License

[MIT](LICENSE) · 内置模型 `face_detection_yunet_2023mar.onnx` 来自 OpenCV Zoo，遵循 Apache-2.0
