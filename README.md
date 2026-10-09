# ITViec Company Intelligence

Ứng dụng Streamlit tìm công ty tương tự, dự đoán Recommend và xem tổng quan công ty.

## Chạy local

Sử dụng Python 3.11 (phiên bản đã kiểm tra với các mô hình hiện tại):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Streamlit Community Cloud

- Chọn repository, branch và main file `app.py`.
- Trong Advanced settings khi deploy, chọn Python 3.11.
- Giữ `requirements.txt`, `data/` và `models/` trong repository.
- Khi cập nhật dependencies, commit và push `requirements.txt` lên branch đang deploy để Cloud cài lại môi trường.

Mô hình `models/yc2_classifier.pkl` được lưu bằng scikit-learn **1.6.1**.
Không tự nâng phiên bản scikit-learn nếu chưa huấn luyện/xuất lại và kiểm tra mô hình.
Lỗi thiếu `_RemainderColsList` khi tải mô hình là dấu hiệu phiên bản scikit-learn không tương thích;
hãy kiểm tra build log để bảo đảm Cloud cài `scikit-learn==1.6.1` từ `requirements.txt`.

Nếu cần đổi phiên bản Python của app đã deploy, Community Cloud yêu cầu xóa app và deploy lại;
chỉ cập nhật thư viện thì không cần bước này.

## Kiểm tra

```bash
.venv/bin/python -m unittest discover -s tests -v
```
