import streamlit as st
import pandas as pd
from model import run_prediction
import matplotlib.pyplot as plt # Важно: импортируем для очистки графиков

# Устанавливаем широкую компоновку страницы Streamlit
st.set_page_config(layout="wide")
st.title("💡 Предиктивное ТО системы ECS Airbus A320")

st.markdown("📤 Загрузите файл данных `ecs_data.xlsx` (с параметрами системы ECS).")

# Виджет для загрузки файла
uploaded_file = st.file_uploader("Выберите Excel-файл", type=["xlsx"])

# Если файл успешно загружен
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ Файл успешно загружен")

        # Кнопка для запуска предсказания
        if st.button("🚀 Запустить предсказание"):
            # Очищаем все предыдущие графики перед новым запуском, чтобы избежать их наложения
            plt.close('all') 
            
            # Вызов функции run_prediction из model.py.
            # ***КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: УБЕДИТЕСЬ, ЧТО ЭТОТ ПОРЯДОК РАСПАКОВКИ ТОЧНО СООТВЕТСТВУЕТ
            # ПОРЯДКУ ВОЗВРАТА ЗНАЧЕНИЙ В ФУНКЦИИ run_prediction В model.py.***
            # model.py возвращает: df, fig_roc, fig_temp, fig_dtemp, status, fig_shap (6 значений)
            result_df, fig_roc, fig_temp, fig_dtemp, status, fig_shap = run_prediction(df)

            st.subheader("📋 Диагностическое заключение:")
            st.info(status) # Здесь отображается строка статуса

            # Проверяем, что DataFrame с результатами не пустой.
            # Если он пустой, значит, в model.py возникла ошибка (например, отсутствуют колонки),
            # и `status` уже содержит сообщение об ошибке. В этом случае графики не отображаем.
            if not result_df.empty: 
                st.subheader("📊 ROC-кривая")
                st.pyplot(fig_roc) # Отображает график fig_roc

                st.subheader("💡 Важность признаков (SHAP Values)") 
                st.pyplot(fig_shap) # Отображает график fig_shap

                st.subheader("🌡 Температура PACK с предсказанными отказами")
                st.pyplot(fig_temp) # Отображает график fig_temp

                st.subheader("📈 Производная температуры PACK (dTemp) с предсказанными отказами")
                st.pyplot(fig_dtemp) # Отображает график fig_dtemp

                # Кнопка для скачивания результатов анализа
                st.download_button(
                    "📥 Скачать результаты анализа",
                    result_df.to_csv(index=False).encode('utf-8'),
                    file_name="ecs_prediction_result.csv"
                )
            # else: Если result_df пустой (из-за ошибок в model.py),
            #       то статус уже отображается через st.info(status),
            #       дополнительных действий здесь не требуется.

    except Exception as e:
        # Обработка общих ошибок при чтении файла или в процессе работы
        st.error(f"❌ Произошла ошибка при обработке файла: {e}")
        st.warning("Пожалуйста, убедитесь, что файл `ecs_data.xlsx` имеет корректный формат и содержит все необходимые колонки.")
        # Обработка общих ошибок при чтении файла или в процессе работы
        st.error(f"❌ Произошла ошибка при обработке файла: {e}")
        st.warning("Пожалуйста, убедитесь, что файл `ecs_data.xlsx` имеет корректный формат и содержит все необходимые колонки.")

