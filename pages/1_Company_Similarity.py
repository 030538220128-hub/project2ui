from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Company Similarity",
    page_icon="🔎",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "yc1_similarity.pkl"


# =========================================================
# LOAD ARTIFACT
# =========================================================

@st.cache_resource
def load_artifact(model_path: Path):
    # Include the model path in the cache key to isolate page artifacts.
    return joblib.load(model_path)


try:
    artifact = load_artifact(MODEL_PATH)
except Exception as e:
    st.error(f"Không thể load yc1_similarity.pkl: {e}")
    st.stop()


companies = artifact["companies"]


# =========================================================
# METHOD CONFIG
# =========================================================

METHODS = {}
for key in ["A", "B", "C", "D"]:
    sim_key = f"similarity_{key}"
    zero_key = f"zero_{key}"
    if sim_key in artifact and key in artifact.get("method_labels", {}):
        METHODS[key] = {
            "name": artifact["method_labels"][key],
            "matrix": artifact[sim_key],
            "zero_mask": artifact[zero_key],
        }


def normalize_text(value: str) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# =========================================================
# RECOMMEND FUNCTIONS
# =========================================================

def get_recommendations(
    company_name: str,
    method_key: str,
    top_n: int,
):
    method = METHODS[method_key]
    similarity_matrix = method["matrix"]
    zero_mask = method["zero_mask"]

    matches = companies.index[
        companies["Company Name"].astype(str).eq(company_name)
    ].tolist()

    if not matches:
        raise ValueError("Không tìm thấy công ty.")

    company_idx = matches[0]

    if bool(zero_mask[company_idx]):
        return pd.DataFrame()

    scores = np.asarray(similarity_matrix[company_idx]).copy()
    scores[company_idx] = -1

    top_indices = np.argsort(scores)[::-1][:top_n]
    result = companies.iloc[top_indices].copy()
    result["Similarity"] = scores[top_indices]

    return result.reset_index(drop=True)


def search_companies_by_query(query_text: str, top_n: int) -> pd.DataFrame:
    cleaned = normalize_text(query_text)
    if not cleaned:
        return pd.DataFrame()

    if "query_vectorizer" in artifact and "query_tfidf_matrix" in artifact:
        vec = artifact["query_vectorizer"].transform([cleaned])
        scores = cosine_similarity(vec, artifact["query_tfidf_matrix"]).ravel()
    else:
        return pd.DataFrame()

    positive_idx = np.where(scores > 0)[0]
    if len(positive_idx) == 0:
        return pd.DataFrame()

    ranked_idx = positive_idx[np.argsort(scores[positive_idx])[::-1]][:top_n]
    result = companies.iloc[ranked_idx].copy()
    result["Similarity"] = scores[ranked_idx]
    return result.reset_index(drop=True)


def render_company_cards(result: pd.DataFrame):
    for index, row in result.iterrows():
        similarity = float(row["Similarity"])
        similarity_pct = similarity * 100

        with st.container(border=True):
            info_col, score_col = st.columns([4, 1])

            with info_col:
                st.subheader(f"#{index + 1} {row['Company Name']}")

                if "Company industry" in result.columns and pd.notna(row["Company industry"]):
                    st.write("**Industry:** " + str(row["Company industry"]))

                if "Company Type" in result.columns and pd.notna(row["Company Type"]):
                    st.write("**Company type:** " + str(row["Company Type"]))

                if "Country" in result.columns and pd.notna(row["Country"]):
                    st.write("**Country:** " + str(row["Country"]))

                if "Company size" in result.columns and pd.notna(row["Company size"]):
                    st.write("**Company size:** " + str(row["Company size"]))

                if "Company overview" in result.columns and pd.notna(row["Company overview"]):
                    with st.expander("Company overview"):
                        words = str(row["Company overview"]).split()
                        preview = " ".join(words[:120]) + ("..." if len(words) > 120 else "")
                        st.write(preview)

                if "Our key skills" in result.columns and pd.notna(row["Our key skills"]):
                    with st.expander("Key skills"):
                        st.write(row["Our key skills"])

                if "Href" in result.columns and pd.notna(row["Href"]):
                    st.link_button("Xem trên ITViec", str(row["Href"]))

            with score_col:
                st.metric("Similarity", f"{similarity_pct:.2f}%")


# =========================================================
# UI
# =========================================================

st.title("🔎 Company Similarity Recommendation")

st.write(
    """
    Tìm các công ty có nội dung mô tả & kỹ năng tương đồng nhất
    từ dữ liệu **478 công ty trên ITViec**.
    """
)

st.divider()

search_mode = st.radio(
    "Chế độ tìm kiếm",
    [
        "🏢 Chọn công ty có sẵn",
        "🔎 Tìm kiếm theo từ khóa / mô tả tự do",
    ],
    horizontal=True,
)


if search_mode == "🏢 Chọn công ty có sẵn":
    company_names = sorted(
        companies["Company Name"].dropna().astype(str).unique()
    )

    left, right = st.columns([2, 1])
    with left:
        selected_company = st.selectbox("Chọn công ty", company_names)
    with right:
        top_n = st.selectbox("Số kết quả", [3, 5, 10], index=1)

    method_key = st.radio(
        "Phương pháp tính similarity",
        options=list(METHODS.keys()),
        format_func=lambda key: f"{key} — {METHODS[key]['name']}",
    )

    st.caption(
        """
        A và B sử dụng Company Overview (scikit-learn TF-IDF vs Gensim).
        C mở rộng thêm Key Skills. D sử dụng giảm chiều ngữ nghĩa LSA (TruncatedSVD + Cosine).
        """
    )

    # Hiển thị thông tin công ty nguồn đang chọn
    src_rows = companies[companies["Company Name"].astype(str).eq(selected_company)]
    if not src_rows.empty:
        src_row = src_rows.iloc[0]
        with st.expander(f"📌 Thông tin công ty đang chọn: {selected_company}", expanded=False):
            st.write(
                f"**Industry:** {src_row.get('Company industry', 'N/A')} | "
                f"**Type:** {src_row.get('Company Type', 'N/A')} | "
                f"**Country:** {src_row.get('Country', 'N/A')}"
            )
            if pd.notna(src_row.get("Our key skills")):
                st.write(f"**Key skills:** {src_row.get('Our key skills')}")
            if pd.notna(src_row.get("Company overview")):
                words = str(src_row.get("Company overview")).split()
                st.write(" ".join(words[:100]) + ("..." if len(words) > 100 else ""))

    if st.button("🔎 Tìm công ty tương tự", type="primary", width="stretch"):
        try:
            result = get_recommendations(
                company_name=selected_company,
                method_key=method_key,
                top_n=top_n,
            )

            if result.empty:
                st.warning(
                    """
                    Công ty này không có đủ dữ liệu
                    để tính similarity bằng phương pháp đã chọn.
                    """
                )
            else:
                st.success(f"Top {len(result)} công ty tương tự với {selected_company}")
                st.caption(f"Phương pháp: {METHODS[method_key]['name']}")
                st.divider()
                render_company_cards(result)
        except Exception as e:
            st.error(f"Có lỗi xảy ra: {e}")

else:
    st.subheader("🔎 Tìm kiếm công ty bằng từ khóa / kỹ năng / mô tả tự do")
    q_left, q_right = st.columns([3, 1])
    with q_left:
        query_input = st.text_input(
            "Nhập kỹ năng, công nghệ, lĩnh vực hoặc mô tả công ty mong muốn",
            value="fintech payment python java backend cloud",
            placeholder="Ví dụ: fintech payment python java backend...",
        )
    with q_right:
        top_n_query = st.selectbox("Số kết quả gợi ý", [3, 5, 10], index=1)

    if st.button("🔎 Tìm công ty phù hợp", type="primary", width="stretch"):
        result = search_companies_by_query(query_input, top_n=top_n_query)
        if result.empty:
            st.warning("Không tìm thấy công ty khớp với từ khóa đã nhập.")
        else:
            st.success(f"Tìm thấy Top {len(result)} công ty khớp nhất với truy vấn: '{query_input}'")
            st.divider()
            render_company_cards(result)
