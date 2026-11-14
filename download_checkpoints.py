import os
import sys
import urllib.request

# Dam bao console Windows ho tro UTF-8
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

CHECKPOINTS = [
    {
        "filename": "lsun_adm.pth",
        "url": "https://huggingface.co/dkarageo/sidbench/resolve/main/dire/lsun_adm.pth",
        "min_bytes": 250_000_000,  # ~269 MB
        "desc": "Classifier ResNet-50 (lsun_adm.pth)",
    },
    {
        "filename": "256x256_diffusion_uncond.pt",
        "url": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt",
        "min_bytes": 2_000_000_000,  # ~2.06 GB
        "desc": "ADM Diffusion Model 256x256 (256x256_diffusion_uncond.pt)",
    },
]


def download_file(url: str, dest_path: str, desc: str):
    part_path = dest_path + ".part"
    print(f"[*] Dang tai {desc}...")
    print(f"    URL: {url}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r    Tien do: {pct:5.1f}% ({mb_done:.1f} / {mb_total:.1f} MB)", end="", flush=True)
        else:
            mb_done = downloaded / (1024 * 1024)
            print(f"\r    Da tai: {mb_done:.1f} MB", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, part_path, reporthook=report_progress)
        print()
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(part_path, dest_path)
        print(f"[+] Tai thanh cong: {dest_path}")
    except Exception as e:
        print(f"\n[!] Loi khi tai {desc}: {e}")
        if os.path.exists(part_path):
            os.remove(part_path)
        sys.exit(1)


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    for item in CHECKPOINTS:
        dest_path = os.path.join(CKPT_DIR, item["filename"])
        if os.path.exists(dest_path) and os.path.getsize(dest_path) >= item["min_bytes"]:
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"[OK] Da co san {item['filename']} ({size_mb:.1f} MB)")
        else:
            download_file(item["url"], dest_path, item["desc"])


if __name__ == "__main__":
    main()
