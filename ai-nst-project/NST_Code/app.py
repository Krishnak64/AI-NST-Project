"""
Flask NST app tuned for Render's free tier (512 MB RAM, CPU only).

Start command (Render -> Settings -> Start Command):
    gunicorn app:app --workers 1 --threads 1 --timeout 180 --bind 0.0.0.0:$PORT

Rename this file to app.py before deploying.
"""
import gc
import logging
import os
import time
import uuid

import torch
from flask import Flask, render_template, request, send_from_directory
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from PIL import Image
from torchvision import transforms
from werkzeug.utils import secure_filename
from wtforms import FileField, FloatField, HiddenField, SubmitField
from wtforms.validators import InputRequired, NumberRange

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# -------------------- Configuration --------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

# Longest side of the images fed to the network. Keep small on the free tier:
# memory use grows with the square of this value.
MAX_SIDE = int(os.environ.get("MAX_SIDE", 256))

# Delete uploaded/generated files older than this (seconds).
FILE_MAX_AGE = 60 * 60

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-me-in-render-env")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
Bootstrap(app)

# CPU only, single thread: lowest memory and CPU contention.
device = torch.device("cpu")
torch.set_num_threads(1)
torch.set_grad_enabled(False)

logger.info("Loading AdaIN models on %s", device)

encoder = VGGEncoder(os.path.join(BASE_DIR, "vgg_normalised.pth")).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "experiment", "final_exp", "decoder_final.pth"),
        map_location=device,
    )
)
encoder.eval()
decoder.eval()
for p in list(encoder.parameters()) + list(decoder.parameters()):
    p.requires_grad = False

gc.collect()
logger.info("AdaIN models loaded successfully")

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()


# -------------------- Form --------------------

class UploadForm(FlaskForm):
    content = FileField("Content Image")
    style = FileField("Style Image")
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField(
        "Alpha",
        default=1.0,
        validators=[InputRequired(), NumberRange(min=0, max=1)],
    )
    submit = SubmitField("Transfer Style")


# -------------------- Helpers --------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def cleanup_old_files():
    """Remove old files so the disk doesn't fill up."""
    now = time.time()
    try:
        for name in os.listdir(UPLOAD_FOLDER):
            path = os.path.join(UPLOAD_FOLDER, name)
            if os.path.isfile(path) and now - os.path.getmtime(path) > FILE_MAX_AGE:
                os.remove(path)
    except OSError:
        logger.exception("Cleanup failed")


def save_uploaded_file(file):
    """Validate, downscale and save an upload. Returns the stored filename."""
    if not file or not file.filename:
        return None

    if not allowed_file(file.filename):
        raise ValueError("Only PNG, JPG and JPEG images are allowed.")

    # Opening with PIL also verifies the file really is an image.
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        raise ValueError("The uploaded file is not a valid image.")

    # Downscale right away: smaller file on disk and less RAM later.
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

    filename = f"{uuid.uuid4().hex}.jpg"
    image.save(os.path.join(UPLOAD_FOLDER, filename), "JPEG", quality=92)
    image.close()
    return filename


def load_image(filename):
    safe_name = secure_filename(filename)
    if not safe_name:
        raise ValueError("Invalid image filename.")

    path = os.path.join(UPLOAD_FOLDER, safe_name)
    if not os.path.isfile(path):
        raise ValueError("Uploaded image not found. Please upload it again.")

    image = Image.open(path).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return image


# -------------------- Style transfer --------------------

def style_transfer(content_image, style_image, alpha):
    content = to_tensor(content_image).unsqueeze(0).to(device)
    style = to_tensor(style_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        content_feats = encoder(content, is_test=True)
        style_feats = encoder(style, is_test=True)
        del content, style

        stylized = adaptive_instance_normalization(content_feats, style_feats)
        del style_feats

        stylized = alpha * stylized + (1 - alpha) * content_feats
        del content_feats

        output = decoder(stylized)
        del stylized

    return output.squeeze(0).clamp(0, 1).cpu()


# -------------------- Routes --------------------

@app.route("/healthz")
def healthz():
    return "ok", 200


@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if request.method == "POST":
        if form.validate_on_submit():
            try:
                cleanup_old_files()

                content_filename = save_uploaded_file(form.content.data) or secure_filename(
                    form.content_path.data or ""
                )
                style_filename = save_uploaded_file(form.style.data) or secure_filename(
                    form.style_path.data or ""
                )

                if not content_filename or not style_filename:
                    raise ValueError("Please upload both content and style images.")

                form.content_path.data = content_filename
                form.style_path.data = style_filename

                logger.info("Starting style transfer")

                content_image = load_image(content_filename)
                style_image = load_image(style_filename)

                output = style_transfer(content_image, style_image, float(form.alpha.data))

                result_filename = f"stylized_{uuid.uuid4().hex}.png"
                to_pil(output).save(os.path.join(UPLOAD_FOLDER, result_filename))
                result_image = result_filename

                logger.info("Style transfer completed successfully")

                del output, content_image, style_image

            except ValueError as exc:
                error = str(exc)
            except Exception:
                logger.exception("Style transfer failed")
                error = "Something went wrong while processing the images. Please try smaller images."
            finally:
                gc.collect()
        else:
            error = "Please check your uploaded images and alpha value."
            logger.warning("Form validation failed: %s", form.errors)

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename or form.content_path.data,
        style_image=style_filename or form.style_path.data,
        error=error,
    )


@app.route("/uploads/<filename>")
def send_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory(os.path.join(BASE_DIR, "examples"), filename)


@app.errorhandler(413)
def file_too_large(_):
    return "Image is too large. Maximum upload size is 5 MB.", 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
