
import os
import uuid
import logging

import torch
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired, NumberRange
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# -------------------- Configuration --------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

Bootstrap(app)

# Render CPU deployment
device = torch.device("cpu")
torch.set_num_threads(1)

logger.info("Loading AdaIN models on %s", device)

encoder_path = os.path.join(BASE_DIR, "vgg_normalised.pth")
decoder_path = os.path.join(
    BASE_DIR, "experiment", "final_exp", "decoder_final.pth"
)

encoder = VGGEncoder(encoder_path).to(device)
decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(decoder_path, map_location=device)
)

encoder.eval()
decoder.eval()

logger.info("AdaIN models loaded successfully")


# -------------------- Form --------------------

class UploadForm(FlaskForm):
    content = FileField("Content Image")
    style = FileField("Style Image")

    content_path = HiddenField()
    style_path = HiddenField()

    alpha = FloatField(
        "Alpha",
        default=1.0,
        validators=[InputRequired(), NumberRange(min=0, max=1)]
    )

    submit = SubmitField("Transfer Style")


# -------------------- Helpers --------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


def save_uploaded_file(file):
    if not file or not file.filename:
        return None

    if not allowed_file(file.filename):
        raise ValueError("Only PNG, JPG and JPEG images are allowed.")

    extension = secure_filename(file.filename).rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{extension}"

    file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

    return filename


def load_image(filename):
    safe_filename = secure_filename(filename)

    if not safe_filename:
        raise ValueError("Invalid image filename.")

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_filename)

    if not os.path.isfile(image_path):
        raise ValueError("Uploaded image not found. Please upload it again.")

    image = Image.open(image_path).convert("RGB")

    # Limit image dimensions to reduce RAM consumption.
    image.thumbnail((512, 512))

    return image


# -------------------- Style Transfer --------------------

def style_transfer(content_image, style_image, alpha):
    image_transform = transforms.Compose([
        transforms.Resize(256, max_size=512),
        transforms.ToTensor()
    ])

    content_tensor = image_transform(content_image).unsqueeze(0).to(device)
    style_tensor = image_transform(style_image).unsqueeze(0).to(device)

    try:
        with torch.inference_mode():
            content_features = encoder(content_tensor, is_test=True)
            style_features = encoder(style_tensor, is_test=True)

            stylized_features = adaptive_instance_normalization(
                content_features,
                style_features
            )

            stylized_features = (
                alpha * stylized_features
                + (1 - alpha) * content_features
            )

            stylized_image = decoder(stylized_features)

        return stylized_image

    finally:
        del content_tensor, style_tensor


def save_image(image, path):
    image = image.detach().cpu().squeeze(0).clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)


# -------------------- Routes --------------------

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
                # Save newly uploaded files or reuse existing selections.
                content_filename = save_uploaded_file(form.content.data)

                if not content_filename:
                    content_filename = secure_filename(form.content_path.data or "")

                style_filename = save_uploaded_file(form.style.data)

                if not style_filename:
                    style_filename = secure_filename(form.style_path.data or "")

                if not content_filename or not style_filename:
                    raise ValueError("Please upload both content and style images.")

                form.content_path.data = content_filename
                form.style_path.data = style_filename

                logger.info("Starting style transfer")

                content_image = load_image(content_filename)
                style_image = load_image(style_filename)

                alpha = form.alpha.data

                output = style_transfer(
                    content_image,
                    style_image,
                    alpha
                )

                result_filename = f"stylized_{uuid.uuid4().hex}.png"
                result_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    result_filename
                )

                save_image(output, result_path)

                result_image = result_filename

                logger.info("Style transfer completed successfully")

                del output, content_image, style_image

            except Exception as exc:
                logger.exception("Style transfer failed")
                error = str(exc)

        else:
            error = "Please check your uploaded images and alpha value."
            logger.warning("Form validation failed: %s", form.errors)

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename or form.content_path.data,
        style_image=style_filename or form.style_path.data,
        error=error
    )


@app.route("/uploads/<filename>")
def send_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory(
        os.path.join(BASE_DIR, "examples"),
        filename
    )


@app.errorhandler(413)
def file_too_large(error):
    return "Image is too large. Maximum upload size is 8 MB.", 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
