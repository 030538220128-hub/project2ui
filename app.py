from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="ITViec Company Intelligence",
    page_icon="💼",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


@st.cache_data
def load_data():
    companies = pd.read_csv(
        DATA_DIR / "Overview_Companies_clean.csv"
    )

    reviews = pd.read_csv(
        DATA_DIR / "Reviews_clean.csv"
    )

    return companies, reviews


st.title("💼 ITViec Company Intelligence")

st.write(
    """
    Hệ thống hỗ trợ khám phá và đánh giá công ty
    dựa trên dữ liệu ITViec.
    """
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔎 Công ty tương tự")
    st.write(
        """
        Chọn một công ty và tìm các công ty
        có nội dung tương đồng nhất.
        """
    )

with col2:
    st.subheader("🤖 Recommend / Not")
    st.write(
        """
        Nhập nội dung review và các mức rating
        để dự đoán người dùng có recommend công ty hay không.
        """
    )

with col3:
    st.subheader("📊 Company Overview")
    st.write(
        """
        Xem thông tin và thống kê review
        của từng công ty.
        """
    )

st.divider()

try:
    companies, reviews = load_data()

    c1, c2 = st.columns(2)

    c1.metric(
        "Company profiles",
        f"{len(companies):,}"
    )

    c2.metric(
        "Reviews analyzed",
        f"{len(reviews):,}"
    )

except Exception as e:
    st.error(
        f"Không thể đọc dữ liệu: {e}"
    )

st.info(
    "Chọn chức năng ở menu bên trái."
)