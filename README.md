# DIRE - Hệ thống Phát hiện Ảnh Nhân tạo AI (Diffusion-Generated Image Detection)

Dự án triển khai mô hình **DIRE (DIffusion Reconstruction Error - ICCV 2023)** kết hợp giao diện Web (Flask + Tailwind CSS) cho phép tải ảnh lên và phân loại **Ảnh Thật (Real)** vs **Ảnh do AI tạo (Synthetic)**.

## 1. Cấu trúc thư mục
- `app.py`: Backend Flask tích hợp pipeline 2 bước của DIRE (ADM Diffusion Reconstruction + ResNet-50 Classifier).
- `templates/index.html`: Giao diện Web màu be nhạt hỗ trợ kéo thả ảnh, xem trước, trực quan hóa ảnh tái tạo ($\hat{x}_0$) và bản đồ sai số DIRE ($|x_0 - \hat{x}_0|$).
- `run_web.bat`: Script khởi chạy nhanh web server trên Windows.
- `source/DIRE/`: Mã nguồn gốc của thuật toán DIRE và `guided-diffusion`.
- `notebooks/DIRE_Experiment.ipynb`: Notebook thực nghiệm trên Google Colab.
- `data/` & `outputs/`: Dữ liệu ảnh mẫu và kết quả trích xuất DIRE thử nghiệm.
- `checkpoints/`: Thư mục chứa trọng số mô hình (không push lên GitHub do vượt quá giới hạn 100MB).

## 2. Tải trọng số mô hình (Bắt buộc trước khi chạy)
Tải 2 tệp sau đặt vào thư mục `checkpoints/`:
1. **Classifier (`lsun_adm.pth` ~270 MB):**
   `https://huggingface.co/dkarageo/sidbench/resolve/main/dire/lsun_adm.pth`
2. **Diffusion Model (`256x256_diffusion_uncond.pt` ~2.06 GB):**
   `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt`

## 3. Hướng dẫn chạy Web
```bash
pip install flask torch torchvision pillow numpy
python app.py
```
Sau đó mở trình duyệt tại: `http://127.0.0.1:5000`
