from pathlib import Path
import re

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Recommend Classifier",
    page_icon="🤖",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "yc2_classifier.pkl"
REVIEWS_PATH = BASE_DIR / "data" / "Reviews_clean.csv"


# =========================================================
# LOAD MODEL + DATA
# =========================================================

@st.cache_resource
def load_artifact(model_path: Path):
    # Include the model path in the cache key to isolate page artifacts.
    return joblib.load(model_path)


@st.cache_data
def load_reviews(reviews_path: Path):
    return pd.read_csv(reviews_path)


try:
    artifact = load_artifact(MODEL_PATH)
    reviews = load_reviews(REVIEWS_PATH)
except Exception as e:
    st.error(f"Không thể load model hoặc dữ liệu: {e}")
    st.stop()


model = artifact["model"]
model_name = artifact["model_name"]
label_map = artifact["label_map"]

TEXT_COLUMNS = artifact["text_columns"]
NUMERIC_COLUMNS = artifact["numeric_columns"]


# =========================================================
# TEXT PREPROCESSING
# =========================================================

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_review_text(title, liked, improvement):
    text = " ".join(
        [
            str(title or ""),
            str(liked or ""),
            str(improvement or ""),
        ]
    )
    return normalize_text(text)


def prepare_input(
    title,
    liked,
    improvement,
    rating,
    salary,
    training,
    management,
    culture,
    office,
):
    review_text = build_review_text(title, liked, improvement)
    return pd.DataFrame(
        [
            {
                "review_text": review_text,
                "Rating": rating,
                "Salary & benefits": salary,
                "Training & learning": training,
                "Management cares about me": management,
                "Culture & fun": culture,
                "Office & workspace": office,
            }
        ]
    )


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_review(input_df):
    prediction = model.predict(input_df)[0]

    try:
        prediction_int = int(prediction)
        label = label_map[prediction_int]
    except Exception:
        label = str(prediction)

    score = None
    score_type = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        score = float(probabilities[-1])
        score_type = "Probability Recommend"
    elif hasattr(model, "decision_function"):
        decision_score = model.decision_function(input_df)
        if hasattr(decision_score, "__len__"):
            decision_score = decision_score[0]
        score = float(decision_score)
        score_type = "Decision score"

    return label, score, score_type


# =========================================================
# RESULT COMPONENT
# =========================================================

def show_prediction(input_df, actual_label=None):
    try:
        label, score, score_type = predict_review(input_df)
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")
        return

    st.divider()
    st.subheader("Kết quả dự đoán")

    if str(label).lower() == "yes":
        st.success("✅ RECOMMEND")
        st.write(
            """
            Model dự đoán người viết review
            **có xu hướng giới thiệu công ty này**.
            """
        )
    else:
        st.error("❌ NOT RECOMMEND")
        st.write(
            """
            Model dự đoán người viết review
            **không có xu hướng giới thiệu công ty này**.
            """
        )

    if actual_label is None:
        c1, c2 = st.columns(2)
    else:
        c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Model Prediction", label)

    with c2:
        if score is not None:
            if score_type == "Probability Recommend":
                st.metric(score_type, f"{score * 100:.2f}%")
            else:
                st.metric(score_type, f"{score:.4f}")
        else:
            st.metric("Model Score", "N/A")

    if actual_label is not None:
        actual_label = str(actual_label).strip()
        with c3:
            st.metric("Actual Label", actual_label)

        is_correct = str(label).lower() == actual_label.lower()
        if is_correct:
            st.success("🎯 Prediction trùng với nhãn thực tế.")
        else:
            st.warning("⚠️ Prediction không trùng với nhãn thực tế.")

    if score_type == "Decision score":
        st.caption(
            """
            **Decision score** không phải xác suất.
            - Score > 0: model nghiêng về Recommend
            - Score < 0: model nghiêng về Not Recommend
            - Score càng xa 0: model càng nghiêng mạnh về một phía
            """
        )

    # Biểu đồ cột thể hiện điểm số từng tiêu chí (theo gợi ý giảng viên)
    rating_cols = [
        "Rating",
        "Salary & benefits",
        "Training & learning",
        "Management cares about me",
        "Culture & fun",
        "Office & workspace",
    ]
    if all(c in input_df.columns for c in rating_cols):
        chart_scores = pd.DataFrame(
            {
                "Tiêu chí": rating_cols,
                "Điểm": [float(input_df.iloc[0][c]) for c in rating_cols],
            }
        ).set_index("Tiêu chí")
        st.write("**Biểu đồ điểm đánh giá các tiêu chí:**")
        st.bar_chart(chart_scores, horizontal=True)

    with st.expander("Xem dữ liệu thực sự đưa vào model"):
        st.dataframe(input_df, width="stretch", hide_index=True)


# =========================================================
# PAGE HEADER
# =========================================================

st.title("🤖 Recommend / Not Recommend")

st.write(
    """
    Dự đoán một review có xu hướng **Recommend** công ty hay **Not Recommend**
    dựa trên nội dung bình luận song ngữ và 6 tiêu chí điểm đánh giá.
    """
)

st.caption(f"Model đang sử dụng: {model_name} (class_weight='balanced')")
st.divider()


# =========================================================
# MODE
# =========================================================

mode = st.radio(
    "Nguồn review",
    [
        "📚 Chọn review có sẵn",
        "✍️ Nhập review mới",
        "📤 Upload CSV dự đoán hàng loạt",
    ],
    horizontal=True,
)


# =========================================================
# MODE 1 — EXISTING REVIEW
# =========================================================

if mode == "📚 Chọn review có sẵn":
    st.subheader("Chọn một review từ dataset")

    company_names = sorted(
        reviews["Company Name"].dropna().astype(str).unique()
    )
    selected_company = st.selectbox("Công ty", company_names)

    company_reviews = reviews[
        reviews["Company Name"].astype(str).eq(selected_company)
    ].copy()

    st.caption(
        f"Có {len(company_reviews):,} review của {selected_company} trong dataset."
    )

    if company_reviews.empty:
        st.warning("Không có review cho công ty này.")
        st.stop()

    review_indices = company_reviews.index.tolist()

    def review_label(idx):
        row = reviews.loc[idx]
        date = str(row.get("Cmt_day", ""))
        title = str(row.get("Title", "Không có tiêu đề"))
        actual = str(row.get("Recommend?", ""))
        if len(title) > 80:
            title = title[:77] + "..."
        return f"{date} — {title} [Actual: {actual}]"

    selected_review_index = st.selectbox(
        "Review",
        review_indices,
        format_func=review_label,
    )
    row = reviews.loc[selected_review_index]

    st.divider()
    st.subheader("Nội dung review")

    title = "" if pd.isna(row["Title"]) else str(row["Title"])
    liked = "" if pd.isna(row["What I liked"]) else str(row["What I liked"])
    improvement = (
        ""
        if pd.isna(row["Suggestions for improvement"])
        else str(row["Suggestions for improvement"])
    )

    st.text_input("Review title", value=title, disabled=True)
    st.text_area("What I liked", value=liked, height=140, disabled=True)
    st.text_area("Suggestions for improvement", value=improvement, height=140, disabled=True)

    st.subheader("Điểm đánh giá")
    rating_values = {
        "Rating": row["Rating"],
        "Salary & benefits": row["Salary & benefits"],
        "Training & learning": row["Training & learning"],
        "Management cares about me": row["Management cares about me"],
        "Culture & fun": row["Culture & fun"],
        "Office & workspace": row["Office & workspace"],
    }

    cols = st.columns(6)
    for i, (column, value) in enumerate(rating_values.items()):
        with cols[i]:
            try:
                value_display = f"{float(value):.0f} / 5"
            except Exception:
                value_display = "N/A"
            st.metric(column, value_display)

    if st.button("🤖 Dự đoán review này", type="primary", width="stretch"):
        input_df = prepare_input(
            title=title,
            liked=liked,
            improvement=improvement,
            rating=row["Rating"],
            salary=row["Salary & benefits"],
            training=row["Training & learning"],
            management=row["Management cares about me"],
            culture=row["Culture & fun"],
            office=row["Office & workspace"],
        )
        show_prediction(input_df, actual_label=row["Recommend?"])


# =========================================================
# MODE 2 — NEW REVIEW
# =========================================================

elif mode == "✍️ Nhập review mới":
    st.subheader("Nhập review mới")
    st.info(
        """
        Chế độ này mô phỏng trường hợp một người dùng nhập review mới
        chưa tồn tại trong dataset.
        """
    )

    title = st.text_input(
        "Review title",
        placeholder="Ví dụ: Môi trường làm việc tốt",
    )
    liked = st.text_area(
        "What I liked",
        placeholder="Ví dụ: Đồng nghiệp thân thiện, môi trường học hỏi tốt...",
        height=130,
    )
    improvement = st.text_area(
        "Suggestions for improvement",
        placeholder="Ví dụ: Công ty nên cải thiện chính sách lương...",
        height=130,
    )

    st.subheader("Điểm đánh giá")
    col1, col2, col3 = st.columns(3)

    with col1:
        rating = st.slider("Overall Rating", 1, 5, 4)
        salary = st.slider("Salary & benefits", 1, 5, 4)
    with col2:
        training = st.slider("Training & learning", 1, 5, 4)
        management = st.slider("Management cares about me", 1, 5, 4)
    with col3:
        culture = st.slider("Culture & fun", 1, 5, 4)
        office = st.slider("Office & workspace", 1, 5, 4)

    if st.button("🤖 Dự đoán", type="primary", width="stretch"):
        if not any([title.strip(), liked.strip(), improvement.strip()]):
            st.warning("Hãy nhập ít nhất một nội dung review.")
            st.stop()

        input_df = prepare_input(
            title=title,
            liked=liked,
            improvement=improvement,
            rating=rating,
            salary=salary,
            training=training,
            management=management,
            culture=culture,
            office=office,
        )
        show_prediction(input_df)


# =========================================================
# MODE 3 — BATCH CSV PREDICTION
# =========================================================

else:
    st.subheader("📤 Upload CSV dự đoán Recommend hàng loạt")
    st.write(
        """
        Upload file CSV chứa danh sách các đánh giá (`Title`, `What I liked`,
        `Suggestions for improvement` và 6 cột điểm số) để dự đoán hàng loạt.
        """
    )

    csv_file = st.file_uploader(
        "Chọn file CSV chứa danh sách review",
        type=["csv"],
        key="batch_review_csv",
    )

    if csv_file is not None:
        try:
            df_up = pd.read_csv(csv_file)
            st.write(f"Đã tải lên **{len(df_up):,}** dòng | **{len(df_up.columns)}** cột.")
            st.dataframe(df_up.head(10), width="stretch")

            if st.button("🤖 Chạy dự đoán cho toàn bộ file CSV", type="primary", width="stretch"):
                work_df = df_up.copy()
                if "review_text" not in work_df.columns:
                    t_col = work_df["Title"] if "Title" in work_df.columns else ""
                    l_col = work_df["What I liked"] if "What I liked" in work_df.columns else ""
                    i_col = (
                        work_df["Suggestions for improvement"]
                        if "Suggestions for improvement" in work_df.columns
                        else ""
                    )
                    work_df["review_text"] = (
                        pd.Series(t_col).fillna("").map(normalize_text)
                        + " "
                        + pd.Series(l_col).fillna("").map(normalize_text)
                        + " "
                        + pd.Series(i_col).fillna("").map(normalize_text)
                    ).str.strip()

                for num_c in NUMERIC_COLUMNS:
                    if num_c not in work_df.columns:
                        work_df[num_c] = 4.0

                X_batch = work_df[["review_text"] + NUMERIC_COLUMNS].copy()
                raw_preds = model.predict(X_batch)
                labels = [label_map.get(int(p), str(p)) for p in raw_preds]
                work_df["predicted_recommend"] = labels

                if hasattr(model, "decision_function"):
                    work_df["decision_score"] = model.decision_function(X_batch).round(4)

                yes_count = sum(1 for x in labels if str(x).lower() == "yes")
                no_count = len(labels) - yes_count

                k1, k2, k3 = st.columns(3)
                k1.metric("Tổng số review", f"{len(labels):,}")
                k2.metric("Dự đoán Recommend (Yes)", f"{yes_count:,} ({yes_count / max(len(labels), 1) * 100:.1f}%)")
                k3.metric("Dự đoán Not Recommend (No)", f"{no_count:,} ({no_count / max(len(labels), 1) * 100:.1f}%)")

                st.dataframe(work_df.head(50), width="stretch")

                st.download_button(
                    "📥 Tải kết quả phân loại review (.csv)",
                    data=work_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="ket_qua_phan_loai_review.csv",
                    mime="text/csv",
                    width="stretch",
                )
        except Exception as e:
            st.error(f"Không thể xử lý file CSV: {e}")