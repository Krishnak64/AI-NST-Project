# 🎨 Neural Style Transfer with AdaIN

Arbitrary neural style transfer in PyTorch using **Adaptive Instance Normalization (AdaIN)**, served through a Flask web app. Upload any content image and any style image, and get a stylized result in a single forward pass, with a slider to control how strongly the style is applied.

🔗 **Live demo:** [https://ai-nst-project-rydt.onrender.com/](https://ai-nst-project-rydt.onrender.com/)

---

## Features

- Style transfer for **arbitrary styles**, with no retraining per style
- Fast, feed-forward inference (VGG encoder → AdaIN → decoder)
- **Alpha control** to blend between the original content (`0.0`) and the full style (`1.0`)
- Simple Flask web UI for uploading images and viewing results
- Training script to train your own decoder on custom content and style datasets
- Runs on both CPU and GPU (CUDA is used automatically when available)

## How it works

The model follows the approach from *"Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"* (Huang & Belongie, 2017).

1. **Encoder:** a fixed, pre-trained VGG-19 (up to `relu4_1`) extracts features from the content and style images.
2. **AdaIN:** the content features are normalized and then re-scaled and shifted to match the channel-wise mean and standard deviation of the style features.
3. **Alpha blending:** `t = α · AdaIN(c, s) + (1 − α) · c`
4. **Decoder:** a trained decoder, which mirrors the encoder, maps `t` back to image space.

During training, only the decoder is optimized, using a content loss and a style loss:

```
L = λ_content · L_content + λ_style · L_style
```

## Project structure

```
.
├── app.py                      # Flask web application
├── train.py                    # Decoder training script
├── utils/
│   ├── models.py               # VGGEncoder and Decoder
│   └── utils.py                # AdaIN, mean/std, dataset and transforms
├── templates/
│   └── index.html              # Web UI template
├── static/uploads/             # Uploaded and generated images
├── examples/                   # Example content/style images
├── experiment/
│   └── final_exp/
│       └── decoder_final.pth   # Trained decoder weights
├── vgg_normalised.pth          # Pre-trained normalised VGG weights
└── requirements.txt
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add the model weights

Place the following files in the project:

| File | Location |
| ---- | -------- |
| Normalised VGG weights | `vgg_normalised.pth` (project root) |
| Trained decoder | `experiment/final_exp/decoder_final.pth` |

> The normalised VGG weights come from the [original AdaIN implementation](https://github.com/naoto0804/pytorch-AdaIN).

### 4. Run the app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

For production, use gunicorn:

```bash
gunicorn app:app
```

## Usage

1. Upload a **content image** (PNG, JPG or JPEG).
2. Upload a **style image**.
3. Set **alpha** between `0` and `1`:
   - `1.0` is full stylization
   - lower values keep more of the original content
4. Click **Transfer Style** to see the result.

## Training your own decoder

Prepare two folders of images, one for content (for example MS-COCO) and one for style (for example WikiArt), then run:

```bash
python train.py \
  --content_dir path/to/content_data \
  --style_dir path/to/style_data \
  --vgg vgg_normalised.pth \
  --experiment my_experiment \
  --epochs 10 \
  --batch_size 4 \
  --style_weight 5
```

Checkpoints and sample outputs are saved to `experiment/<experiment_name>/`.

### Training arguments

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--content_dir` | – | Folder of content images |
| `--style_dir` | – | Folder of style images |
| `--vgg` | `vgg_normalised.pth` | Pre-trained VGG weights |
| `--experiment` | `experiment1` | Output folder name |
| `--final_size` | `256` | Final crop size |
| `--content_size` / `--style_size` | `512` | Resize size before cropping |
| `--batch_size` | `4` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--lr_decay` | `5e-5` | Learning rate decay |
| `--epochs` | `1` | Number of epochs |
| `--content_weight` | `1.0` | Content loss weight |
| `--style_weight` | `5` | Style loss weight |
| `--save_interval` | `2` | Save a checkpoint every N epochs |
| `--resume` | `False` | Resume from a checkpoint (`--decoder_path`, `--optimizer_path`) |

## Tech stack

- Python
- PyTorch and torchvision
- Flask, Flask-WTF and Flask-Bootstrap
- Pillow
- Gunicorn
- Deployed on [Render](https://render.com/)

## Acknowledgements

- Huang, X. and Belongie, S., *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*, ICCV 2017.
- [naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) for the normalised VGG weights.
