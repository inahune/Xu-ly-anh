import os
import sys
import gc
import io
import pickle
import zipfile
import base64
import time
import tempfile
import subprocess
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

# Them source/DIRE vao sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIRE_DIR = os.path.join(BASE_DIR, "source", "DIRE")
if DIRE_DIR not in sys.path:
    sys.path.insert(0, DIRE_DIR)

from utils.utils import get_network

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # Toi da 32MB

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSIFIER_PATH = os.path.join(BASE_DIR, "checkpoints", "lsun_adm.pth")
DIFFUSION_PATH = os.path.join(BASE_DIR, "checkpoints", "256x256_diffusion_uncond.pt")
DIFFUSION_BF16_PATH = os.path.join(BASE_DIR, "checkpoints", "256x256_diffusion_uncond_bf16.pt")
WORKER_PATH = os.path.join(BASE_DIR, "dire_worker.py")

# ---------------------------------------------------------
# 1. Chi nap Classifier ResNet-50 (~100MB RAM) de server sieu nhe
# ---------------------------------------------------------
print(f"[*] Khoi tao Classifier ResNet-50 (che do tiet kiem RAM) tren {DEVICE}...")
if os.path.exists(CLASSIFIER_PATH):
    try:
        with torch.device("meta"):
            classifier_model = get_network("resnet50")
        state_dict = torch.load(CLASSIFIER_PATH, map_location="cpu", weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        classifier_model.load_state_dict(state_dict, assign=True)
        del state_dict
    except Exception:
        classifier_model = get_network("resnet50")
        state_dict = torch.load(CLASSIFIER_PATH, map_location="cpu", weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        classifier_model.load_state_dict(state_dict)

    classifier_model.eval()
    if DEVICE == "cuda":
        classifier_model.cuda()
    print(f"[+] Tai Classifier thanh cong! Server san sang (~220MB RAM).")
else:
    classifier_model = get_network("resnet50")
    print(f"[!] Canh bao: Khong tim thay checkpoint Classifier tai {CLASSIFIER_PATH}")


def ensure_bf16_checkpoint():
    if os.path.exists(DIFFUSION_BF16_PATH) and os.path.getsize(DIFFUSION_BF16_PATH) > 1_000_000_000:
        return True
    if not os.path.exists(DIFFUSION_PATH) or os.path.getsize(DIFFUSION_PATH) < 2_000_000_000:
        return False

    print("[*] Dang toi uu hoa Diffusion checkpoint sang bfloat16 (chi chay 1 lan)...")
    try:
        with zipfile.ZipFile(DIFFUSION_PATH, "r") as z:
            pkl_bytes = z.read("archive/data.pkl")
            offsets = {}
            with open(DIFFUSION_PATH, "rb") as f:
                for info in z.infolist():
                    if info.filename.startswith("archive/data/"):
                        f.seek(info.header_offset)
                        lh = f.read(30)
                        fn_len = int.from_bytes(lh[26:28], "little")
                        ex_len = int.from_bytes(lh[28:30], "little")
                        data_offset = info.header_offset + 30 + fn_len + ex_len
                        key = info.filename.split("/")[-1]
                        offsets[key] = (data_offset, info.file_size)

            mmaps = {
                k: np.memmap(DIFFUSION_PATH, dtype=np.uint8, mode="r", offset=off, shape=(sz,))
                for k, (off, sz) in offsets.items()
            }

            class MemmapStorage:
                def __init__(self, key, dtype, numel):
                    self.key = key
                    self.dtype = dtype
                    self.numel = numel

            def rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                np_dtype = np.float32 if storage.dtype == torch.float32 else np.float16
                itemsize = np.dtype(np_dtype).itemsize
                byte_start = storage_offset * itemsize
                total_elems = 1
                for s in size:
                    total_elems *= s
                raw = mmaps[storage.key][byte_start : byte_start + total_elems * itemsize]
                arr = np.frombuffer(raw.tobytes(), dtype=np_dtype).reshape(size)
                return torch.from_numpy(arr.copy()).to(torch.bfloat16)

            class CustomUnpickler(pickle.Unpickler):
                def persistent_load(self, pid):
                    _, stype, key, loc, numel = pid
                    dtype = torch.float32 if "FloatStorage" in str(stype) else torch.float16
                    return MemmapStorage(key, dtype, numel)

                def find_class(self, module, name):
                    if name == "_rebuild_tensor_v2":
                        return rebuild_tensor_v2
                    if module == "collections" and name == "OrderedDict":
                        from collections import OrderedDict
                        return OrderedDict
                    return super().find_class(module, name)

            state_dict = CustomUnpickler(io.BytesIO(pkl_bytes)).load()
            tmp_path = DIFFUSION_BF16_PATH + ".tmp"
            torch.save(state_dict, tmp_path)
            del state_dict, mmaps
            gc.collect()
            if os.path.exists(DIFFUSION_BF16_PATH):
                os.remove(DIFFUSION_BF16_PATH)
            os.rename(tmp_path, DIFFUSION_BF16_PATH)
            return True
    except Exception as e:
        print(f"[!] Loi chuyen doi bfloat16: {e}")
        return False


def is_diffusion_available():
    if DEVICE == "cpu":
        return (
            (os.path.exists(DIFFUSION_BF16_PATH) and os.path.getsize(DIFFUSION_BF16_PATH) > 1_000_000_000)
            or (os.path.exists(DIFFUSION_PATH) and os.path.getsize(DIFFUSION_PATH) > 2_000_000_000)
        )
    return os.path.exists(DIFFUSION_PATH) and os.path.getsize(DIFFUSION_PATH) > 2_000_000_000


# ---------------------------------------------------------
# Ham tien xu ly & goi worker
# ---------------------------------------------------------
transform_pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def pil_to_base64(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def run_dire_worker(pil_image: Image.Image):
    """Chay Diffusion trong tien trinh rieng de khong chiem RAM server va tu giai phong RAM khi xong"""
    if DEVICE == "cpu" and not ensure_bf16_checkpoint():
        raise RuntimeError("Khong tim thay hoac khong the toi uu checkpoint Diffusion.")

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.png")
        recons_path = os.path.join(tmpdir, "recons.png")
        dire_path = os.path.join(tmpdir, "dire.png")
        pil_image.convert("RGB").save(in_path, format="PNG")

        proc = subprocess.run(
            [sys.executable, WORKER_PATH, in_path, recons_path, dire_path],
            capture_output=True,
            text=True,
            timeout=240,
        )
        if proc.returncode != 0 or not os.path.exists(dire_path):
            err_detail = (proc.stderr or "").strip().splitlines()[-1:] or ["Het RAM hoac loi tien trinh"]
            raise RuntimeError(
                f"May khong du RAM trong de chay Diffusion (~1.5 GB RAM). Hay dong bot ung dung hoac chon Che do Sieu Nhe. ({err_detail[0]})"
            )

        recons_pil = Image.open(recons_path).convert("RGB")
        recons_pil.load()
        dire_pil = Image.open(dire_path).convert("RGB")
        dire_pil.load()
        return recons_pil, dire_pil


def classify_pil_image(img: Image.Image):
    """Dua anh vao ResNet-50 de phan loai"""
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
        "arch": "ResNet-50 (DIRE Detector)",
    }


@app.route("/")
def index():
    return render_template("index.html", diffusion_ready=is_diffusion_available())


@app.route("/api/status")
def api_status():
    ready = is_diffusion_available()
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

    mode = request.form.get("mode", "dire")  # dire | direct

    try:
        start_time = time.time()
        image_bytes = file.read()
        pil_image = Image.open(BytesIO(image_bytes))

        recons_b64 = None
        dire_b64 = None

        if mode == "dire":
            if not is_diffusion_available():
                return jsonify({"error": "Chưa có mô hình Diffusion. Vui lòng chọn Chế độ Siêu Nhẹ."}), 400
            recons_pil, dire_pil = run_dire_worker(pil_image)
            recons_b64 = pil_to_base64(recons_pil)
            dire_b64 = pil_to_base64(dire_pil)
            result = classify_pil_image(dire_pil)
            result["pipeline_mode"] = "Full 2-Step DIRE (Subprocess + ResNet-50)"
        else:
            result = classify_pil_image(pil_image)
            result["pipeline_mode"] = "Chế độ Siêu Nhẹ (ResNet-50 ~220MB RAM)"

        result["inference_time_ms"] = round((time.time() - start_time) * 1000, 2)
        if recons_b64 and dire_b64:
            result["recons_image"] = recons_b64
            result["dire_image"] = dire_b64

        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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

    mode = request.args.get("mode", "direct")

    try:
        start_time = time.time()
        pil_image = Image.open(sample_map[name])

        if name != "dire" and mode == "dire" and is_diffusion_available():
            recons_pil, dire_pil = run_dire_worker(pil_image)
            result = classify_pil_image(dire_pil)
            result["pipeline_mode"] = "Full 2-Step DIRE (Subprocess + ResNet-50)"
            result["recons_image"] = pil_to_base64(recons_pil)
            result["dire_image"] = pil_to_base64(dire_pil)
        else:
            result = classify_pil_image(pil_image)
            result["pipeline_mode"] = "Pre-computed DIRE" if name == "dire" else "Chế độ Siêu Nhẹ (ResNet-50)"

        result["inference_time_ms"] = round((time.time() - start_time) * 1000, 2)
        return jsonify({
            "success": True,
            "result": result,
            "sample_name": name,
            "filename": os.path.basename(sample_map[name]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
