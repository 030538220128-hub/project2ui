# ITViec Company Intelligence (Project 2)

Ứng dụng Streamlit gợi ý công ty IT tương tự (Content-Based Filtering), dự đoán khả năng **Recommend** từ đánh giá nhân viên và phân tích tổng quan thị trường tuyển dụng IT trên ITViec.

## Giới thiệu tính năng

1. **Company Similarity (`1_Company_Similarity.py`)**: Gợi ý Top-N công ty tương tự dựa trên 4 phương pháp (**Method A**: Content TF-IDF Cosine, **Method B**: Review Sentiment & Text Similarity, **Method C**: Hybrid Content + Review, **Method D**: LSA `TruncatedSVD + Cosine Similarity`). Hỗ trợ tìm kiếm công ty theo **từ khóa / kỹ năng tự do** (Free-text Query Search) và hiển thị thẻ chi tiết công ty gốc.
2. **Recommend Classifier (`2_Recommend_Classifier.py`)**: Dự đoán xác suất nhân viên khuyên làm việc tại công ty (`Recommend` vs `Not Recommend`) bằng mô hình **Logistic Regression** kết hợp TF-IDF văn bản đa ngôn ngữ Việt-Anh và các chỉ số đánh giá. Hỗ trợ biểu đồ 6 tiêu chí đánh giá chi tiết và dự đoán hàng loạt qua file CSV.
3. **Company Overview (`3_Company_Overview.py`)**: Thống kê KPIs, phân bố điểm đánh giá và tra cứu hồ sơ 478 công ty IT trên ITViec.

## Chạy local

Sử dụng Python 3.11 hoặc Python 3.12 (phiên bản đã kiểm tra với các mô hình hiện tại):

```bash
python -m venv .venv
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
