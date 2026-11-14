import os
import sys
import base64
import time
from io import BytesIO

# Dam bao console Windows ho tro hien thi UTF-8 khong bi loi charmap
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# Them source/DIRE va guided-diffusion vao sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIRE_DIR = os.path.join(BASE_DIR, "source", "DIRE")
GUIDED_DIFFUSION_DIR = os.path.join(DIRE_DIR, "guided-diffusion")

for p in (DIRE_DIR, GUIDED_DIFFUSION_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.utils import get_network
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # Toi da 32MB

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSIFIER_PATH = os.path.join(BASE_DIR, "checkpoints", "lsun_adm.pth")
DIFFUSION_PATH = os.path.join(BASE_DIR, "checkpoints", "256x256_diffusion_uncond.pt")

# ---------------------------------------------------------
# 1. Khoi tao Classifier (ResNet-50)
# ---------------------------------------------------------
print(f"[*] Khoi tao Classifier ResNet-50 tren thiet bi: {DEVICE}...")
if os.path.exists(CLASSIFIER_PATH):
    try:
        with torch.device("meta"):
            classifier_model = get_network("resnet50")
        state_dict = torch.load(CLASSIFIER_PATH, map_location="cpu", mmap=True, weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        classifier_model.load_state_dict(state_dict, assign=True)
    except Exception:
        classifier_model = get_network("resnet50")
        state_dict = torch.load(CLASSIFIER_PATH, map_location="cpu", mmap=True, weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        classifier_model.load_state_dict(state_dict)

    classifier_model.eval()
    if DEVICE == "cuda":
        classifier_model.cuda()
    print(f"[+] Tai Classifier thanh cong tu: {CLASSIFIER_PATH}")
else:
    classifier_model = get_network("resnet50")
    print(f"[!] Canh bao: Khong tim thay checkpoint Classifier tai {CLASSIFIER_PATH}")

# ---------------------------------------------------------
# 2. Khoi tao Diffusion Model (256x256 ADM) neu da co file
# ---------------------------------------------------------
diffusion_model = None
diffusion_engine = None
DIFFUSION_READY = False


def load_diffusion_if_ready():
    global diffusion_model, diffusion_engine, DIFFUSION_READY
    if DIFFUSION_READY:
        return True

    # Kiem tra file ton tai va du dung luong (~2.1 GB)
    if not os.path.exists(DIFFUSION_PATH):
        return False
    file_size = os.path.getsize(DIFFUSION_PATH)
    if file_size < 2_000_000_000:
        print(f"[!] File Diffusion chua tai xong ({file_size / (1024**2):.1f} MB / ~2109 MB).")
        return False

    try:
        print(f"[*] Dang nap Diffusion Model 256x256 (ddim20) len {DEVICE}...")
        defaults = model_and_diffusion_defaults()
        defaults.update(
            dict(
                attention_resolutions="32,16,8",
                class_cond=False,
                diffusion_steps=1000,
                dropout=0.1,
                image_size=256,
                learn_sigma=True,
                noise_schedule="linear",
                num_channels=256,
                num_head_channels=64,
                num_res_blocks=2,
                resblock_updown=True,
                use_fp16=(DEVICE == "cuda"),
                use_scale_shift_norm=True,
                timestep_respacing="ddim20",
            )
        )
        with torch.device("meta"):
            model, diff = create_model_and_diffusion(**defaults)
        sd = torch.load(DIFFUSION_PATH, map_location="cpu", mmap=True, weights_only=False)
        model.load_state_dict(sd, assign=True)
        if DEVICE == "cuda":
            model.to("cuda")
            model.convert_to_fp16()
        model.eval()

        diffusion_model = model
        diffusion_engine = diff
        DIFFUSION_READY = True
        print("[+] Nap Diffusion Model thanh cong! Pipeline 2 buoc DIRE da san sang.")
        return True
    except Exception as e:
        print(f"[!] Loi khi nap Diffusion Model: {e}")
        return False


load_diffusion_if_ready()

# ---------------------------------------------------------
# Ham tien xu ly & tinh DIRE
# ---------------------------------------------------------
transform_pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def center_crop_arr(pil_image: Image.Image, image_size: int = 256) -> np.ndarray:
    """Tien xu ly anh giong het guided_diffusion/image_datasets.py"""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def pil_to_base64(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def compute_dire_images(pil_image: Image.Image):
    """
    Buoc 1 cua DIRE:
    Anh goc x0 -> DDIM Inversion -> x_T -> DDIM Reconstruction -> x_hat_0
    DIRE = |x0 - x_hat_0|
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    arr = center_crop_arr(pil_image, 256)
    arr_norm = arr.astype(np.float32) / 127.5 - 1.0
    imgs = torch.from_numpy(np.transpose(arr_norm, [2, 0, 1])).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        latent = diffusion_engine.ddim_reverse_sample_loop(
            diffusion_model,
            (1, 3, 256, 256),
            noise=imgs,
            clip_denoised=True,
            model_kwargs={},
            real_step=0,
        )
        recons = diffusion_engine.ddim_sample_loop(
            diffusion_model,
            (1, 3, 256, 256),
            noise=latent,
            clip_denoised=True,
            model_kwargs={},
            real_step=0,
        )
        dire = torch.abs(imgs - recons)

    recons_uint8 = ((recons + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
    dire_uint8 = (dire * 255.0 / 2.0).clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()

    cropped_orig_pil = Image.fromarray(arr.astype(np.uint8))
    recons_pil = Image.fromarray(recons_uint8)
    dire_pil = Image.fromarray(dire_uint8)
    return cropped_orig_pil, recons_pil, dire_pil


def classify_pil_image(img: Image.Image):
    """Buoc 2: Dua anh (ban do DIRE) vao ResNet-50 de phan loai"""
    if img.mode != "RGB":
        img = img.convert("RGB")

    tensor = transform_pipeline(img)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = classifier_model(tensor)
        prob_synthetic = logits.sigmoid().item()

    is_synthetic = prob_synthetic >= 0.5
    confidence = prob_synthetic if is_synthetic else (1.0 - prob_synthetic)

    return {
        "prob_synthetic": round(prob_synthetic, 4),
        "prob_real": round(1.0 - prob_synthetic, 4),
        "prob_synthetic_pct": round(prob_synthetic * 100, 2),
        "prob_real_pct": round((1.0 - prob_synthetic) * 100, 2),
        "label": "SYNTHETIC (AI Generated)" if is_synthetic else "REAL (Authentic)",
        "is_synthetic": is_synthetic,
        "confidence_pct": round(confidence * 100, 2),
        "device": DEVICE.upper(),
        "arch": "ADM Diffusion (ddim20) + ResNet-50",
    }


@app.route("/")
def index():
    return render_template("index.html", diffusion_ready=load_diffusion_if_ready())


@app.route("/api/status")
def api_status():
    ready = load_diffusion_if_ready()
    size_mb = 0.0
    if os.path.exists(DIFFUSION_PATH):
        size_mb = round(os.path.getsize(DIFFUSION_PATH) / (1024 * 1024), 1)
    return jsonify({
        "diffusion_ready": ready,
        "diffusion_size_mb": size_mb,
        "device": DEVICE.upper(),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "Vui lòng chọn hoặc tải lên tệp ảnh hợp lệ"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Tên tệp không hợp lệ"}), 400

    mode = request.form.get("mode", "auto")  # auto | dire | direct

    try:
        start_time = time.time()
        image_bytes = file.read()
        pil_image = Image.open(BytesIO(image_bytes))

        use_diffusion = (mode == "dire") or (mode == "auto" and load_diffusion_if_ready())

        recons_b64 = None
        dire_b64 = None

        if use_diffusion:
            if not load_diffusion_if_ready():
                return jsonify({"error": "Mô hình Diffusion (256x256_diffusion_uncond.pt) chưa sẵn sàng."}), 503
            _, recons_pil, dire_pil = compute_dire_images(pil_image)
            recons_b64 = pil_to_base64(recons_pil)
            dire_b64 = pil_to_base64(dire_pil)
            result = classify_pil_image(dire_pil)
            result["pipeline_mode"] = "Full 2-Step DIRE (DDIM20 + ResNet-50)"
        else:
            result = classify_pil_image(pil_image)
            result["pipeline_mode"] = "Direct Classifier Only (Chưa qua Diffusion)"
            result["arch"] = "ResNet-50 (Direct)"

        result["inference_time_ms"] = round((time.time() - start_time) * 1000, 2)
        if recons_b64 and dire_b64:
            result["recons_image"] = recons_b64
            result["dire_image"] = dire_b64

        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"error": f"Lỗi trong quá trình xử lý ảnh: {str(e)}"}), 500


@app.route("/api/sample/<name>")
def api_sample(name):
    """Du doan truc tiep tren cac anh mau co san"""
    sample_map = {
        "real": os.path.join(BASE_DIR, "data", "test2.jpg"),
        "dire": os.path.join(BASE_DIR, "outputs", "dire_test", "test.png"),
        "original": os.path.join(BASE_DIR, "data", "test.png"),
    }

    if name not in sample_map or not os.path.exists(sample_map[name]):
        return jsonify({"error": "Ảnh mẫu không tồn tại"}), 404

    try:
        start_time = time.time()
        pil_image = Image.open(sample_map[name])

        # Neu la anh dire co san thi phan loai truc tiep, nguoc lai chay full DIRE neu da san sang
        if name != "dire" and load_diffusion_if_ready():
            _, recons_pil, dire_pil = compute_dire_images(pil_image)
            result = classify_pil_image(dire_pil)
            result["pipeline_mode"] = "Full 2-Step DIRE (DDIM20 + ResNet-50)"
            result["recons_image"] = pil_to_base64(recons_pil)
            result["dire_image"] = pil_to_base64(dire_pil)
        else:
            result = classify_pil_image(pil_image)
            result["pipeline_mode"] = "Pre-computed DIRE" if name == "dire" else "Direct Classifier"

        result["inference_time_ms"] = round((time.time() - start_time) * 1000, 2)
        return jsonify({
            "success": True,
            "result": result,
            "sample_name": name,
            "filename": os.path.basename(sample_map[name]),
        })
    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý mẫu: {str(e)}"}), 500


@app.route("/api/sample_image/<name>")
def api_sample_image(name):
    """Phuc vu file anh mau de hien thi preview"""
    sample_map = {
        "real": os.path.join(BASE_DIR, "data", "test2.jpg"),
        "dire": os.path.join(BASE_DIR, "outputs", "dire_test", "test.png"),
        "original": os.path.join(BASE_DIR, "data", "test.png"),
    }
    if name in sample_map and os.path.exists(sample_map[name]):
        return send_file(sample_map[name])
    return ("Not found", 404)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Ung dung web DIRE Detector san sang tai: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
