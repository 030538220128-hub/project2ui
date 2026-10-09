from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Company Similarity",
    page_icon="🔎",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "yc1_similarity.pkl"
)


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
    st.error(
        f"Không thể load yc1_similarity.pkl: {e}"
    )
    st.stop()


companies = artifact["companies"]


# =========================================================
# METHOD CONFIG
# =========================================================

METHODS = {
    "A": {
        "name": artifact["method_labels"]["A"],
        "matrix": artifact["similarity_A"],
        "zero_mask": artifact["zero_A"],
    },

    "B": {
        "name": artifact["method_labels"]["B"],
        "matrix": artifact["similarity_B"],
        "zero_mask": artifact["zero_B"],
    },

    "C": {
        "name": artifact["method_labels"]["C"],
        "matrix": artifact["similarity_C"],
        "zero_mask": artifact["zero_C"],
    },
}


# =========================================================
# RECOMMEND FUNCTION
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
        companies["Company Name"]
        .astype(str)
        .eq(company_name)
    ].tolist()

    if not matches:
        raise ValueError(
            "Không tìm thấy công ty."
        )

    company_idx = matches[0]

    # Vector rỗng -> không đủ text để tính similarity
    if bool(zero_mask[company_idx]):
        return pd.DataFrame()

    scores = np.asarray(
        similarity_matrix[company_idx]
    ).copy()

    # Không recommend lại chính công ty đang chọn
    scores[company_idx] = -1

    top_indices = np.argsort(
        scores
    )[::-1][:top_n]

    result = companies.iloc[
        top_indices
    ].copy()

    result["Similarity"] = (
        scores[top_indices]
    )

    return result.reset_index(drop=True)


# =========================================================
# UI
# =========================================================

st.title(
    "🔎 Company Similarity Recommendation"
)

st.write(
    """
    Tìm các công ty có nội dung tương tự
    với một công ty được chọn.
    """
)

st.divider()


company_names = sorted(
    companies["Company Name"]
    .dropna()
    .astype(str)
    .unique()
)


left, right = st.columns(
    [2, 1]
)


with left:

    selected_company = st.selectbox(
        "Chọn công ty",
        company_names,
    )


with right:

    top_n = st.selectbox(
        "Số kết quả",
        [3, 5, 10],
        index=1,
    )


method_key = st.radio(
    "Phương pháp tính similarity",
    options=["A", "B", "C"],
    format_func=lambda key: (
        f"{key} — {METHODS[key]['name']}"
    ),
)


st.caption(
    """
    A và B sử dụng Company Overview.
    C mở rộng thêm Key Skills.
    """
)


if st.button(
    "🔎 Tìm công ty tương tự",
    type="primary",
    use_container_width=True,
):

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

            st.success(
                f"Top {len(result)} công ty tương tự "
                f"với {selected_company}"
            )

            st.caption(
                f"Phương pháp: "
                f"{METHODS[method_key]['name']}"
            )

            st.divider()

            for index, row in result.iterrows():

                similarity = float(
                    row["Similarity"]
                )

                similarity_pct = (
                    similarity * 100
                )

                with st.container(
                    border=True
                ):

                    info_col, score_col = (
                        st.columns([4, 1])
                    )

                    with info_col:

                        st.subheader(
                            f"#{index + 1} "
                            f"{row['Company Name']}"
                        )

                        if (
                            "Company industry"
                            in result.columns
                            and pd.notna(
                                row["Company industry"]
                            )
                        ):
                            st.write(
                                "**Industry:** "
                                + str(
                                    row[
                                        "Company industry"
                                    ]
                                )
                            )

                        if (
                            "Company Type"
                            in result.columns
                            and pd.notna(
                                row["Company Type"]
                            )
                        ):
                            st.write(
                                "**Company type:** "
                                + str(
                                    row[
                                        "Company Type"
                                    ]
                                )
                            )

                        if (
                            "Country"
                            in result.columns
                            and pd.notna(
                                row["Country"]
                            )
                        ):
                            st.write(
                                "**Country:** "
                                + str(
                                    row["Country"]
                                )
                            )

                        if (
                            "Company size"
                            in result.columns
                            and pd.notna(
                                row["Company size"]
                            )
                        ):
                            st.write(
                                "**Company size:** "
                                + str(
                                    row[
                                        "Company size"
                                    ]
                                )
                            )

                        if (
                            "Our key skills"
                            in result.columns
                            and pd.notna(
                                row["Our key skills"]
                            )
                        ):

                            with st.expander(
                                "Key skills"
                            ):
                                st.write(
                                    row[
                                        "Our key skills"
                                    ]
                                )

                        if (
                            "Href"
                            in result.columns
                            and pd.notna(
                                row["Href"]
                            )
                        ):

                            st.link_button(
                                "Xem trên ITViec",
                                str(row["Href"])
                            )


                    with score_col:

                        st.metric(
                            "Similarity",
                            f"{similarity_pct:.2f}%",
                        )

    except Exception as e:

        st.error(
            f"Có lỗi xảy ra: {e}"
        )
