import streamlit as st
import pandas as pd
from model import run_prediction

st.set_page_config(layout="wide")
st.title("💡 Предиктивное ТО системы ECS Airbus A320")

st.markdown("📤 Загрузите файл данных `ecs_data.xlsx` (с параметрами системы ECS).")

uploaded_file = st.file_uploader("Выберите Excel-файл", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ Файл успешно загружен")

    if st.button("🚀 Запустить предсказание"):
        result_df, fig_roc, fig_temp, fig_dtemp, status = run_prediction(df)

        st.subheader("📋 Диагностическое заключение:")
        st.info(status)

        st.subheader("📊 ROC-кривая")
        st.pyplot(fig_roc)

        st.subheader("🌡 Температура PACK с отказами")
        st.pyplot(fig_temp)

        st.subheader("📈 Производная температуры PACK (dTemp)")
        st.pyplot(fig_dtemp)

        st.download_button(
            "📥 Скачать результаты анализа",
            result_df.to_csv(index=False).encode('utf-8'),
            file_name="ecs_prediction_result.csv"
        )
