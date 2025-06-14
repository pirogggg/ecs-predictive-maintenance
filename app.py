import streamlit as st
import pandas as pd
from model import run_prediction  # убедись, что model.py лежит в той же папке

st.set_page_config(page_title="ECS Maintenance AI", layout="wide")

st.title("🛫 Предиктивное техническое обслуживание ECS")
st.markdown("""
Прототип интерфейса для оценки состояния системы кондиционирования воздуха (ECS) на основе реальных данных и предсказаний модели Random Forest.
""")

uploaded_file = st.file_uploader("📂 Загрузите файл .xlsx с эксплуатационными данными", type="xlsx")

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ Файл успешно загружен.")
        st.write("🔍 Первые строки таблицы:")
        st.dataframe(df.head())

        with st.spinner("🔄 Обработка и предсказание..."):
            result_df, fig_roc, fig_temp, fig_dtemp, fig_shap, status = run_prediction(df)

        st.subheader("📈 ROC-кривая")
        st.pyplot(fig_roc)

        st.subheader("🌡️ График температуры PACK")
        st.pyplot(fig_temp)

        st.subheader("📉 Производная температуры PACK (dTemp)")
        st.pyplot(fig_dtemp)

        st.subheader("📊 SHAP-график важности признаков")
        st.pyplot(fig_shap)

        st.subheader("📋 Сводка предсказаний:")
        st.dataframe(result_df[["Pack Outlet Temp (Â°C)", "dTemp", "Predicted Failure"]].head(15))

        st.success(f"🧠 Итоговая оценка системы: {status}")

    except Exception as e:
        st.error(f"❌ Ошибка при обработке данных: {e}")
else:
    st.info("⬆️ Загрузите .xlsx файл для начала анализа.")
