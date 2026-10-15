from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Company Overview",
    page_icon="📊",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    companies = pd.read_csv(
        DATA_DIR / "Overview_Companies_clean.csv"
    )

    reviews = pd.read_csv(
        DATA_DIR / "Overview_Reviews_clean.csv"
    )

    df = companies.merge(
        reviews.drop(columns=["id"], errors="ignore"),
        on="Company Name",
        how="left",
    )

    return df


try:
    df = load_data()

except Exception as e:

    st.error(
        f"Không thể đọc dữ liệu: {e}"
    )

    st.stop()


# =========================================================
# HELPER
# =========================================================

def safe_value(value, default="Không có dữ liệu"):

    if pd.isna(value):
        return default

    value = str(value).strip()

    if not value:
        return default

    return value


def safe_number(value, default=0.0):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


# =========================================================
# UI HEADER
# =========================================================

st.title("📊 Company Overview")

st.write(
    """
    Khám phá thông tin công ty và tổng hợp đánh giá
    từ dữ liệu ITViec.
    """
)

st.divider()


# =========================================================
# COMPANY SELECT
# =========================================================

# =========================================================
# COMPANY FILTER
# =========================================================

show_all = st.checkbox(
    "Hiển thị cả công ty chưa có review",
    value=False,
)


if show_all:

    company_names = sorted(
        df["Company Name"]
        .dropna()
        .astype(str)
        .unique()
    )

else:

    company_names = sorted(
        df.loc[
            pd.to_numeric(
                df["Number of reviews"],
                errors="coerce"
            ).fillna(0) > 0,
            "Company Name",
        ]
        .dropna()
        .astype(str)
        .unique()
    )


# Ưu tiên Accenture làm công ty demo mặc định
default_index = 0

if "Accenture" in company_names:
    default_index = company_names.index(
        "Accenture"
    )


selected_company = st.selectbox(
    "Chọn công ty",
    company_names,
    index=default_index,
)

company_df = df[
    df["Company Name"] == selected_company
]

if company_df.empty:

    st.warning(
        "Không tìm thấy dữ liệu công ty."
    )

    st.stop()


company = company_df.iloc[0]


# =========================================================
# COMPANY TITLE
# =========================================================

st.header(
    selected_company
)


company_type = safe_value(
    company.get("Company Type")
)

industry = safe_value(
    company.get("Company industry")
)

country = safe_value(
    company.get("Country")
)

size = safe_value(
    company.get("Company size")
)


st.caption(
    f"{company_type} • {industry} • {country}"
)


# =========================================================
# MAIN METRICS
# =========================================================

st.subheader("Tổng quan đánh giá")


number_reviews = safe_number(
    company.get("Number of reviews")
)

overall_rating = safe_number(
    company.get("Overall rating")
)

recommend_rate = safe_number(
    company.get(
        "Recommend working here to a friend"
    )
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Số review",
        f"{int(number_reviews):,}"
    )


with col2:

    if number_reviews > 0:

        st.metric(
            "Overall Rating",
            f"{overall_rating:.1f} / 5"
        )

    else:

        st.metric(
            "Overall Rating",
            "N/A"
        )


with col3:

    if number_reviews > 0:

        st.metric(
            "Recommend",
            f"{recommend_rate:.0f}%"
        )

    else:

        st.metric(
            "Recommend",
            "N/A"
        )


if number_reviews == 0:

    st.info(
        """
        Công ty này chưa có review tổng hợp trong dataset,
        nên các chỉ số rating và Recommend chưa có ý nghĩa.
        """
    )


st.divider()


# =========================================================
# RATING DETAILS
# =========================================================

st.subheader("Chi tiết điểm đánh giá")


rating_columns = {
    "Salary & benefits": "Lương & phúc lợi",
    "Training & learning": "Đào tạo & học hỏi",
    "Management cares about me": "Quản lý",
    "Culture & fun": "Văn hóa",
    "Office & workspace": "Văn phòng",
}


rating_data = []


for column, label in rating_columns.items():

    value = safe_number(
        company.get(column)
    )

    rating_data.append(
        {
            "Tiêu chí": label,
            "Điểm": value,
        }
    )


rating_df = pd.DataFrame(
    rating_data
)


if number_reviews > 0:

    cols = st.columns(5)

    for index, row in rating_df.iterrows():

        with cols[index]:

            st.metric(
                row["Tiêu chí"],
                f"{row['Điểm']:.1f} / 5"
            )


    st.write("")

    chart_df = (
        rating_df
        .set_index("Tiêu chí")
    )

    st.bar_chart(
        chart_df,
        y="Điểm",
        horizontal=True,
    )


else:

    st.warning(
        "Chưa có dữ liệu rating để hiển thị."
    )


st.divider()


# =========================================================
# COMPANY INFORMATION
# =========================================================

st.subheader("Thông tin công ty")


info_col1, info_col2 = st.columns(2)


with info_col1:

    st.write(
        f"**Company type:** {company_type}"
    )

    st.write(
        f"**Industry:** {industry}"
    )

    st.write(
        f"**Company size:** {size}"
    )


with info_col2:

    st.write(
        f"**Country:** {country}"
    )

    st.write(
        "**Working days:** "
        + safe_value(
            company.get("Working days")
        )
    )

    st.write(
        "**Overtime policy:** "
        + safe_value(
            company.get("Overtime Policy")
        )
    )


st.divider()


# =========================================================
# COMPANY OVERVIEW
# =========================================================

overview = safe_value(
    company.get("Company overview")
)


st.subheader("Giới thiệu công ty")

st.write(
    overview
)


# =========================================================
# KEY SKILLS
# =========================================================

skills = safe_value(
    company.get("Our key skills")
)


with st.expander(
    "🛠 Key Skills"
):

    st.write(
        skills
    )


# =========================================================
# WHY LOVE WORKING HERE
# =========================================================

why_love = safe_value(
    company.get(
        "Why you'll love working here"
    )
)


with st.expander(
    "❤️ Why you'll love working here"
):

    st.write(
        why_love
    )


# =========================================================
# LOCATION
# =========================================================

location = safe_value(
    company.get("Location")
)


with st.expander(
    "📍 Location"
):

    st.write(
        location
    )


# =========================================================
# ITVIEC LINK
# =========================================================

href = company.get("Href")


if (
    pd.notna(href)
    and str(href).strip()
):

    st.link_button(
        "🔗 Xem công ty trên ITViec",
        str(href),
        width="stretch",
    )