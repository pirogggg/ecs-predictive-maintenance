import streamlit as st
import pandas as pd
from model import run_prediction
import matplotlib.pyplot as plt
import io
import base64

# Устанавливаем широкую компоновку страницы Streamlit
st.set_page_config(layout="wide")

# Изменение заголовка: иконка самолета и сокращенный текст
st.title("✈️ Предиктивное ТО системы ECS")

st.markdown("📤 Загрузите файл данных `ecs_data.xlsx` (с параметрами системы ECS).")

# Виджет для загрузки файла
uploaded_file = st.file_uploader("Выберите Excel-файл", type=["xlsx"])

# Функция для конвертации Matplotlib фигуры в base64 строку для встраивания в HTML
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) # Закрываем фигуру, чтобы освободить память
    return f'<img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto; display: block; margin: 10px auto;">'

# Если файл успешно загружен
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ Файл успешно загружен")

        # Кнопка для запуска предсказания
        if st.button("🚀 Запустить предсказание"):
            # Очищаем все предыдущие графики перед новым запуском
            plt.close('all') 
            
            # Вызов функции run_prediction из model.py.
            # Теперь ожидаем 8 возвращаемых значений:
            result_df, fig_roc, fig_temp, fig_dtemp, status, fig_shap, classification_report_str, roc_auc_str = run_prediction(df)

            st.subheader("📋 Диагностическое заключение:")
            st.info(status) 

            # Проверяем, что DataFrame с результатами не пустой, чтобы отображать графики и отчеты
            if not result_df.empty: 
                st.subheader("📊 ROC-кривая")
                st.pyplot(fig_roc) 

                st.subheader("💡 Важность признаков (SHAP Values)") 
                st.pyplot(fig_shap) 

                st.subheader("🌡 Температура PACK с предсказанными отказами")
                st.pyplot(fig_temp) 

                st.subheader("📈 Производная температуры PACK (dTemp) с предсказанными отказами")
                st.pyplot(fig_dtemp) 

                # --- Создание HTML-отчета для скачивания ---
                html_report_content = f"""
                <!DOCTYPE html>
                <html lang="ru">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Отчет по предиктивному ТО системы ECS</title>
                    <style>
                        body {{ font-family: 'Inter', sans-serif; line-height: 1.6; color: #333; margin: 20px; background-color: #f4f4f4; }}
                        .container {{ max-width: 900px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                        h1, h2, h3 {{ color: #0056b3; }}
                        .status-box {{ padding: 15px; margin: 20px 0; border-radius: 8px; font-weight: bold; }}
                        .status-red {{ background-color: #ffe0e0; border: 1px solid #ff4d4d; color: #ff4d4d; }}
                        .status-yellow {{ background-color: #fffbe0; border: 1px solid #ffd700; color: #ffd700; }}
                        .status-green {{ background-color: #e0ffe0; border: 1px solid #4CAF50; color: #4CAF50; }}
                        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        pre {{ background-color: #eee; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                        .section {{ margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
                        .section:last-child {{ border-bottom: none; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>✈️ Отчет по предиктивному ТО системы ECS</h1>
                        <p>Дата генерации отчета: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

                        <div class="section">
                            <h2>Диагностическое заключение</h2>
                            <div class="status-box {'status-red' if '🔴' in status else 'status-yellow' if '🟡' in status else 'status-green'}">
                                {status}
                            </div>
                        </div>

                        <div class="section">
                            <h2>Метрики производительности модели</h2>
                            <h3>Отчет классификации</h3>
                            {classification_report_str}
                            <p>{roc_auc_str}</p>
                        </div>
                        
                        <div class="section">
                            <h2>Графики анализа</h2>
                            <h3>ROC-кривая</h3>
                            {fig_to_base64(fig_roc)}
                            
                            <h3>Важность признаков (SHAP Values)</h3>
                            {fig_to_base64(fig_shap)}

                            <h3>Температура PACK с предсказанными отказами</h3>
                            {fig_to_base64(fig_temp)}

                            <h3>Производная температуры PACK (dTemp) с предсказанными отказами</h3>
                            {fig_to_base64(fig_dtemp)}
                        </div>

                        <div class="section">
                            <h2>Детали предсказаний</h2>
                            <h3>Таблица предсказаний (первые 20 строк)</h3>
                            {result_df.head(20).to_html(index=False)}
                            <p><i>Отображены первые 20 строк для примера. Полный DataFrame доступен при скачивании CSV.</i></p>
                        </div>

                    </div>
                </body>
                </html>
                """
                
                # Кнопка для скачивания полного HTML-отчета
                st.download_button(
                    "📥 Скачать полный отчет (HTML)",
                    html_report_content.encode('utf-8'),
                    file_name="ecs_predictive_maintenance_report.html",
                    mime="text/html"
                )

                # Оставляем кнопку для скачивания CSV с полными данными на всякий случай
                st.download_button(
                    "📥 Скачать результаты анализа (CSV)",
                    result_df.to_csv(index=False).encode('utf-8'),
                    file_name="ecs_prediction_result.csv",
                    mime="text/csv"
                )

    except Exception as e:
        # Обработка общих ошибок при чтении файла или в процессе работы
        st.error(f"❌ Произошла ошибка при обработке файла: {e}")
        st.warning("Пожалуйста, убедитесь, что файл `ecs_data.xlsx` имеет корректный формат и содержит все необходимые колонки.")
        st.error(f"❌ Произошла ошибка при обработке файла: {e}")
        st.warning("Пожалуйста, убедитесь, что файл `ecs_data.xlsx` имеет корректный формат и содержит все необходимые колонки.")

