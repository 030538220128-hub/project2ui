from pathlib import Path
import re

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Recommend Classifier",
    page_icon="🤖",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "yc2_classifier.pkl"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_artifact(model_path: Path):
    # Include the model path in the cache key to isolate page artifacts.
    return joblib.load(model_path)


try:
    artifact = load_artifact(MODEL_PATH)

except Exception as e:
    st.error(
        f"Không thể load yc2_classifier.pkl: {e}"
    )
    st.stop()


model = artifact["model"]
model_name = artifact["model_name"]
label_map = artifact["label_map"]

TEXT_COLUMNS = artifact["text_columns"]
NUMERIC_COLUMNS = artifact["numeric_columns"]


# =========================================================
# TEXT PREPROCESSING
# Giống logic trong notebook YC2
# =========================================================

def normalize_text(text):
    text = str(text).lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


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
    # Notebook ghép 3 text field thành review_text
    review_text = " ".join(
        [
            str(title),
            str(liked),
            str(improvement),
        ]
    )

    review_text = normalize_text(
        review_text
    )

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
# UI
# =========================================================

st.title("🤖 Recommend / Not Recommend")

st.write(
    """
    Nhập nội dung review và các mức đánh giá.
    Mô hình sẽ dự đoán người viết review có xu hướng
    **Recommend** công ty hay không.
    """
)

st.caption(
    f"Model đang sử dụng: {model_name}"
)

st.divider()


# =========================================================
# TEXT INPUT
# =========================================================

st.subheader("1. Nội dung review")

title = st.text_input(
    "Review title",
    placeholder="Ví dụ: Môi trường làm việc tốt",
)

liked = st.text_area(
    "What I liked",
    placeholder=(
        "Ví dụ: Đồng nghiệp thân thiện, "
        "môi trường học hỏi tốt..."
    ),
    height=130,
)

improvement = st.text_area(
    "Suggestions for improvement",
    placeholder=(
        "Ví dụ: Công ty nên cải thiện "
        "chính sách lương và phúc lợi..."
    ),
    height=130,
)


# =========================================================
# RATING INPUT
# =========================================================

st.subheader("2. Điểm đánh giá")

col1, col2, col3 = st.columns(3)

with col1:

    rating = st.slider(
        "Overall Rating",
        min_value=1,
        max_value=5,
        value=4,
    )

    salary = st.slider(
        "Salary & benefits",
        min_value=1,
        max_value=5,
        value=4,
    )


with col2:

    training = st.slider(
        "Training & learning",
        min_value=1,
        max_value=5,
        value=4,
    )

    management = st.slider(
        "Management cares about me",
        min_value=1,
        max_value=5,
        value=4,
    )


with col3:

    culture = st.slider(
        "Culture & fun",
        min_value=1,
        max_value=5,
        value=4,
    )

    office = st.slider(
        "Office & workspace",
        min_value=1,
        max_value=5,
        value=4,
    )


st.divider()


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "🤖 Dự đoán",
    type="primary",
    use_container_width=True,
):

    # Không bắt buộc cả 3 text đều có,
    # nhưng ít nhất nên có một nội dung.
    if not any(
        [
            title.strip(),
            liked.strip(),
            improvement.strip(),
        ]
    ):

        st.warning(
            "Hãy nhập ít nhất một nội dung review."
        )

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


    try:

        prediction = model.predict(
            input_df
        )[0]

        prediction = int(
            prediction
        )

        label = label_map[
            prediction
        ]


        # ==============================================
        # SCORE
        # ==============================================

        score = None
        score_type = None

        if hasattr(
            model,
            "predict_proba",
        ):

            probability = (
                model
                .predict_proba(
                    input_df
                )[0][1]
            )

            score = float(
                probability
            )

            score_type = (
                "Probability Recommend"
            )


        elif hasattr(
            model,
            "decision_function",
        ):

            decision_score = (
                model
                .decision_function(
                    input_df
                )
            )

            if hasattr(
                decision_score,
                "__len__",
            ):
                decision_score = (
                    decision_score[0]
                )

            score = float(
                decision_score
            )

            score_type = (
                "Decision score"
            )


        # ==============================================
        # OUTPUT
        # ==============================================

        if label == "Yes":

            st.success(
                "✅ RECOMMEND"
            )

            st.write(
                """
                Mô hình dự đoán review này có xu hướng
                **giới thiệu công ty cho người khác**.
                """
            )

        else:

            st.error(
                "❌ NOT RECOMMEND"
            )

            st.write(
                """
                Mô hình dự đoán review này có xu hướng
                **không giới thiệu công ty**.
                """
            )


        result_col1, result_col2 = (
            st.columns(2)
        )


        with result_col1:

            st.metric(
                "Prediction",
                label,
            )


        with result_col2:

            if score is not None:

                if score_type == (
                    "Probability Recommend"
                ):

                    st.metric(
                        score_type,
                        f"{score * 100:.2f}%",
                    )

                else:

                    st.metric(
                        score_type,
                        f"{score:.4f}",
                    )


        if score_type == "Decision score":

            st.caption(
                """
                Decision score > 0:
                model nghiêng về Recommend.

                Decision score < 0:
                model nghiêng về Not Recommend.

                Đây không phải xác suất.
                """
            )


        # ==============================================
        # SHOW INPUT
        # ==============================================

        with st.expander(
            "Xem dữ liệu đưa vào model"
        ):

            st.dataframe(
                input_df,
                use_container_width=True,
                hide_index=True,
            )


    except Exception as e:

        st.error(
            f"Lỗi khi dự đoán: {e}"
        )
