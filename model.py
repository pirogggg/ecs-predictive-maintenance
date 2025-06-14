import pandas as pd
import numpy as np # Добавлено, так как используется в Colab коде
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

def run_prediction(df):
    # Используем .copy() для df, чтобы избежать SettingWithCopyWarning
    df_processed = df.copy() 
    df_processed.rename(columns={
        "Pack Outlet Temp (Â°C)": "Pack Outlet Temp (°C)", # Исправление кодировки
        "Fan Speed": "Fan Speed (rpm)",
        "Temperature Deviation": "Temp Deviation (°C)", # Исправление кодировки
        "Valve Position": "Valve Position (%)",
        "Bleed Pressure": "Bleed Air Pressure (psi)",
        "Failure": "Failure in 10h" # На случай, если целевая колонка называется просто "Failure"
    }, inplace=True)

    # Автоматический расчет Temp Deviation, если она отсутствует, но есть необходимые колонки
    if "Temp Deviation (°C)" not in df_processed.columns and \
       "Cabin Temp Setpoint (°C)" in df_processed.columns and \
       "Cabin Actual Temp (°C)" in df_processed.columns:
        # Убедитесь, что эти колонки также используют правильный символ градуса, если они есть в Excel.
        df_processed["Temp Deviation (°C)"] = df_processed["Cabin Temp Setpoint (°C)"] - df_processed["Cabin Actual Temp (°C)"]

    df = df_processed # Теперь работаем с обработанным DataFrame

    # Определение всех необходимых колонок, включая те, что могут быть созданы
    required_core_columns = [
        "Pack Outlet Temp (°C)", "Fan Speed (rpm)", "Temp Deviation (°C)",
        "Valve Position (%)", "Bleed Air Pressure (psi)", "Failure in 10h"
    ]
    
    # Проверка на наличие всех необходимых колонок в загруженном DataFrame
    missing_columns_initial = [col for col in required_core_columns if col not in df.columns]

    if missing_columns_initial:
        status_messages = []
        if "Temp Deviation (°C)" in missing_columns_initial:
            status_messages.append(
                "❌ Колонка 'Temp Deviation (°C)' отсутствует. "
                "Обеспечьте наличие 'Temperature Deviation' (оригинальное название) "
                "ИЛИ обеих колонок 'Cabin Temp Setpoint (°C)' и 'Cabin Actual Temp (°C)' в вашем Excel-файле для автоматического расчета."
            )
            # Удаляем Temp Deviation из списка, чтобы не повторять сообщение
            missing_columns_initial.remove("Temp Deviation (°C)") 
        
        if missing_columns_initial: 
            status_messages.append(
                f"❌ Отсутствуют следующие обязательные колонки: {', '.join(missing_columns_initial)}. "
                f"Пожалуйста, убедитесь, что файл `ecs_data.xlsx` содержит все нужные данные."
            )
        
        # Возвращаем пустые фигуры и DataFrame при ошибке для корректной работы app.py
        # Добавлено 6 значений, чтобы app.py мог их корректно распаковать
        return pd.DataFrame(), plt.figure(), plt.figure(), plt.figure(), "\n".join(status_messages), plt.figure()

    # Создание дополнительных признаков
    df["dTemp"] = df["Pack Outlet Temp (°C)"].diff().fillna(0)
    df["Temp_MA10"] = df["Pack Outlet Temp (°C)"].rolling(window=10).mean().bfill()
    df["Valve_range_10"] = df["Valve Position (%)"].rolling(window=10).apply(lambda x: max(x) - min(x), raw=False).fillna(0)

    features = [
        "Pack Outlet Temp (°C)",
        "Fan Speed (rpm)",
        "Temp Deviation (°C)",
        "Valve Position (%)",
        "Bleed Air Pressure (psi)",
        "dTemp",
        "Temp_MA10",
        "Valve_range_10"
    ]
    
    # Явное преобразование всех колонок признаков в числовой формат и заполнение NaN
    for col in features:
        if col in df.columns: 
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    X = df[features].fillna(0) 
    y = df["Failure in 10h"]

    # Проверка на достаточное количество классов для обучения
    if y.nunique() < 2:
        status = "🔴 Ошибка: Целевая колонка 'Failure in 10h' содержит менее двух уникальных значений. " \
                 "Невозможно выполнить обучение модели классификации. Убедитесь, что в данных есть как отказы, так и их отсутствие."
        return pd.DataFrame(), plt.figure(), plt.figure(), plt.figure(), status, plt.figure()

    # Проверка, что достаточно данных для train_test_split после стратификации
    unique_classes, counts = y.value_counts().index, y.value_counts().values
    if any(count < 2 for count in counts):
         status = "🔴 Ошибка: Недостаточно сэмплов для одного из классов в колонке 'Failure in 10h' " \
                  "для стратифицированного разделения данных. Убедитесь, что каждый класс содержит не менее 2 записей."
         return pd.DataFrame(), plt.figure(), plt.figure(), plt.figure(), status, plt.figure()

    # Проверка соответствия размеров X и y
    if len(X) != len(y):
        status = "🔴 Ошибка: Количество строк в признаках (X) и целевой переменной (y) не совпадает. " \
                 "Проверьте исходный файл данных."
        return pd.DataFrame(), plt.figure(), plt.figure(), plt.figure(), status, plt.figure()

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Обучение модели RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight='balanced_subsample',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Предсказания и расчет метрик
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba) 
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    # Построение ROC-кривой
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    ax_roc.set_title("ROC-кривая модели", fontsize=14) # Обновлен заголовок для ясности
    ax_roc.set_xlabel("False Positive Rate", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate", fontsize=12)
    ax_roc.legend(fontsize=10)
    ax_roc.grid(True, linestyle='--', alpha=0.6)
    fig_roc.tight_layout() 

    # Построение SHAP-графика для объяснения важности признаков
    # Используем data=X_train для TreeExplainer, как в Colab
    explainer = shap.TreeExplainer(model, data=X_train) 
    shap_values = explainer.shap_values(X_test)

    # Логика обработки shap_values из Colab для выбора positive_class
    shap_values_positive_class = None
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            shap_values_positive_class = shap_values[1] # Для бинарной классификации, класс 1
        else:
            shap_values_positive_class = shap_values[0] # Если список, но только один элемент
    else:
        if shap_values.ndim == 3 and shap_values.shape[2] == 2:
            shap_values_positive_class = shap_values[:, :, 1]
        elif shap_values.ndim == 2:
            shap_values_positive_class = shap_values
        else:
            print(f"Unexpected shap_values shape: {shap_values.shape}") # Отладочный вывод

    fig_shap = plt.figure(figsize=(10, 7)) # Создаем фигуру перед вызовом shap.summary_plot
    if shap_values_positive_class is not None:
        # Гарантируем, что X_test для SHAP имеет правильные колонки и порядок
        X_test_for_shap_plot = pd.DataFrame(X_test, columns=features)
        
        # Добавлена проверка на совпадение форм перед вызовом
        if shap_values_positive_class.shape[1] == X_test_for_shap_plot.shape[1]:
            shap.summary_plot(shap_values_positive_class, X_test_for_shap_plot, plot_type="bar", show=False, feature_names=features)
        else:
            print(f"Shape mismatch before SHAP plotting: shap_values_positive_class.shape={shap_values_positive_class.shape}, X_test_for_shap_plot.shape={X_test_for_shap_plot.shape}")
            # Fallback: попытаться сбросить названия колонок, если проблема в них
            shap.summary_plot(shap_values_positive_class, X_test_for_shap_plot.values, plot_type="bar", show=False, feature_names=features)
            
    fig_shap.tight_layout() 
    # Внимание: plt.close(fig_shap) здесь не нужен, так как Streamlit сам управляет отображением.
    # Он будет закрыт после того, как Streamlit его использует.

    # Добавление предсказанных отказов в исходный DataFrame
    df["Predicted Failure"] = model.predict(X)

    # Построение графика температуры PACK с предсказанными отказами
    fig_temp, ax_temp = plt.subplots(figsize=(12, 6)) # Переименовал ax2 в ax_temp для единообразия
    ax_temp.plot(df["Pack Outlet Temp (°C)"].reset_index(drop=True), label="Температура PACK", color="blue", linewidth=1.5)
    ax_temp.scatter(df[df["Predicted Failure"] == 1].index,
                df.loc[df["Predicted Failure"] == 1, "Pack Outlet Temp (°C)"],
                color="red", label="Предсказан отказ", s=50, zorder=5, alpha=0.7) 
    ax_temp.set_title("Температура PACK с предсказанными отказами", fontsize=14) # Обновлен заголовок
    ax_temp.set_xlabel("Точки данных", fontsize=12) # Обновлен заголовок оси X
    ax_temp.set_ylabel("Температура PACK (°C)", fontsize=12) # Обновлен заголовок оси Y
    ax_temp.grid(True, linestyle='--', alpha=0.6)
    ax_temp.legend(fontsize=10)
    fig_temp.tight_layout()

    # Построение графика производной температуры PACK с предсказанными отказами
    fig_dtemp, ax_dtemp = plt.subplots(figsize=(12, 6)) # Переименовал ax3 в ax_dtemp для единообразия
    ax_dtemp.plot(df["dTemp"].reset_index(drop=True), label="Производная температуры (dTemp)", color="orange", linewidth=1.5)
    ax_dtemp.scatter(df[df["Predicted Failure"] == 1].index,
                df.loc[df["Predicted Failure"] == 1, "dTemp"],
                color="red", label="Предсказан отказ", s=50, zorder=5, alpha=0.7)
    ax_dtemp.set_title("Производная температуры PACK с предсказанными отказами", fontsize=14) # Обновлен заголовок
    ax_dtemp.set_xlabel("Точки данных", fontsize=12) # Обновлен заголовок оси X
    ax_dtemp.set_ylabel("dTemp (°C/точка)", fontsize=12) # Обновлен заголовок оси Y
    ax_dtemp.grid(True, linestyle='--', alpha=0.6)
    ax_dtemp.legend(fontsize=10)
    fig_dtemp.tight_layout()

    # Формирование диагностического заключения
    total_failures = df["Predicted Failure"].sum()
    max_dtemp = df["dTemp"].abs().max()
    dtemp_threshold = 3

    if total_failures > 3:
        status = "🔴 Необходим ремонт или технический осмотр."
    elif total_failures > 0 or max_dtemp > dtemp_threshold:
        status = "🟡 Рекомендуется наблюдение за системой."
    else:
        status = "🟢 Система функционирует стабильно."

    # Отладочная информация (будет выведена в консоль Streamlit)
    try:
        print(classification_report(y_test, y_pred, zero_division=0))
        print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    except Exception as e:
        print(f"Ошибка при выводе отчета классификации или ROC-AUC: {e}")

    # Возвращаем обновленный DataFrame, все объекты фигур Matplotlib и статус
    # Порядок: df, fig_roc, fig_temp, fig_dtemp, status, fig_shap
    return df, fig_roc, fig_temp, fig_dtemp, status, fig_shap

