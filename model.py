import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

def run_prediction(df):
    # Добавление новых признаков
    # Исправление: Изменяем 'Â°C' на '°C' для корректного чтения названия колонки.
    # Убедитесь, что название колонки в вашем файле Excel точно соответствует 'Pack Outlet Temp (°C)'.

    # *** НОВОЕ ИЗМЕНЕНИЕ: Автоматическая обработка разных названий колонок в начале ***
    # Используем .copy() для df, чтобы избежать SettingWithCopyWarning при inplace=True
    df_processed = df.copy() 
    df_processed.rename(columns={
        "Pack Outlet Temp (Â°C)": "Pack Outlet Temp (°C)",
        "Fan Speed": "Fan Speed (rpm)",
        "Temperature Deviation": "Temp Deviation (°C)",
        "Valve Position": "Valve Position (%)",
        "Bleed Pressure": "Bleed Air Pressure (psi)",
        # Добавляем целевую переменную на случай, если ее название отличается
        "Failure": "Failure in 10h" 
    }, inplace=True)

    # Автоматический расчет Temp Deviation, если не хватает
    # Убедитесь, что 'Cabin Temp Setpoint (°C)' и 'Cabin Actual Temp (°C)'
    # также используют правильный символ градуса, если они есть в Excel.
    if "Temp Deviation (°C)" not in df_processed.columns and \
       "Cabin Temp Setpoint (°C)" in df_processed.columns and \
       "Cabin Actual Temp (°C)" in df_processed.columns:
        df_processed["Temp Deviation (°C)"] = df_processed["Cabin Temp Setpoint (°C)"] - df_processed["Cabin Actual Temp (°C)"]

    # Теперь используем df_processed для всех дальнейших операций
    df = df_processed # Присваиваем обратно df, чтобы остальной код работал без изменений

    # Define all required columns including those that might be created
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
        
        if missing_columns_initial: # Если после проверки Temp Deviation остались другие отсутствующие колонки
            status_messages.append(
                f"❌ Отсутствуют следующие обязательные колонки: {', '.join(missing_columns_initial)}. "
                f"Пожалуйста, убедитесь, что файл `ecs_data.xlsx` содержит все нужные данные."
            )
        
        # Возвращаем ошибку с детальным сообщением
        return pd.DataFrame(), plt.figure(), plt.figure(), plt.figure(), "\n".join(status_messages), plt.figure()

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
    
    # *** КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ ОШИБКИ SHAP: Обработка NaN в X перед обучением и SHAP ***
    # Модели машинного обучения и SHAP не работают с пропущенными значениями (NaN).
    # Заполняем все NaN нулями. В реальном проекте может потребоваться более сложная импутация.
    # Это важно сделать ПОСЛЕ создания новых признаков, так как они тоже могут содержать NaN.
    X = df[features].fillna(0) 
    y = df["Failure in 10h"]

    # Проверка, что в целевой переменной `y` есть хотя бы два класса (0 и 1)
    # Это необходимо для стратифицированного разделения данных и обучения модели классификации.
    if len(y.unique()) < 2:
        status = "🔴 Ошибка: Колонка 'Failure in 10h' должна содержать как минимум два разных значения (например, 0 и 1) " \
                 "для обучения модели. Текущие уникальные значения: " + str(y.unique())
        return pd.DataFrame(), plt.figure(), plt.figure(), plt.figure(), status, plt.figure()

    # Проверка, что достаточно данных для train_test_split после стратификации.
    # Если какой-либо класс имеет менее 2 сэмплов, `stratify=y` вызовет ошибку.
    # Проверяем количество уникальных значений в y.
    # Если y состоит только из одного значения (например, все 0 или все 1),
    # stratify=y вызовет ошибку ValueError.
    if y.nunique() < 2:
        status = "🔴 Ошибка: Целевая колонка 'Failure in 10h' содержит менее двух уникальных значений. " \
                 "Невозможно выполнить обучение модели классификации. Убедитесь, что в данных есть как отказы, так и их отсутствие."
        return pd.DataFrame(), plt.figure(), plt.figure(), plt.figure(), status, plt.figure()

    unique_classes, counts = y.value_counts().index, y.value_counts().values
    if any(count < 2 for count in counts):
         status = "🔴 Ошибка: Недостаточно сэмплов для одного из классов в колонке 'Failure in 10h' " \
                  "для стратифицированного разделения данных. Убедитесь, что каждый класс содержит не менее 2 записей."
         return pd.DataFrame(), plt.figure(), plt.figure(), plt.figure(), status, plt.figure()

    # Разделение данных на обучающую и тестовую выборки
    # Убедитесь, что X и y имеют одинаковое количество строк
    if len(X) != len(y):
        status = "🔴 Ошибка: Количество строк в признаках (X) и целевой переменной (y) не совпадает. " \
                 "Проверьте исходный файл данных."
        return pd.DataFrame(), plt.figure(), plt.figure(), plt.figure(), status, plt.figure()


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
    roc_auc = roc_auc_score(y_test, y_proba) # Используем roc_auc_score
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    # Построение ROC-кривой
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    ax_roc.set_title("ROC-кривая", fontsize=14)
    ax_roc.set_xlabel("False Positive Rate", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate", fontsize=12)
    ax_roc.legend(fontsize=10)
    ax_roc.grid(True, linestyle='--', alpha=0.6)
    fig_roc.tight_layout() # Улучшение компоновки

    # Построение SHAP-графика для объяснения важности признаков
    explainer = shap.TreeExplainer(model)
    # Расчет SHAP-значений для тестовой выборки
    # shap_values[1] содержит значения для класса "1" (отказ)
    shap_values = explainer.shap_values(X_test) 
    
    # Создаем новую фигуру для SHAP-графика
    fig_shap, ax_shap = plt.subplots(figsize=(10, 7))
    # Передаем axis в shap.summary_plot, чтобы он рисовал на нашей фигуре.
    # X_test передается как DataFrame для корректного отображения названий признаков.
    # Используем shap_values[1] для бинарной классификации (для предсказания класса 1)
    shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False, ax=ax_shap)
    ax_shap.set_title("Важность признаков (SHAP Values)", fontsize=14)
    fig_shap.tight_layout() # Улучшение компоновки

    # Добавление предсказанных отказов в исходный DataFrame
    # Важно: Используем DataFrame `X` (который уже обработан fillna(0)) для предсказания
    # Применяем предсказания на исходный `df` (который теперь является `df_processed`)
    df["Predicted Failure"] = model.predict(X)

    # Построение графика температуры PACK с предсказанными отказами
    fig_temp, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(df["Pack Outlet Temp (°C)"].reset_index(drop=True), label="Температура PACK", color="blue", linewidth=1.5)
    ax2.scatter(df[df["Predicted Failure"] == 1].index,
                df.loc[df["Predicted Failure"] == 1, "Pack Outlet Temp (°C)"],
                color="red", label="Предсказан отказ", s=50, zorder=5, alpha=0.7) 
    ax2.set_title("Температура PACK с предсказанными отказами", fontsize=14)
    ax2.set_xlabel("Точки данных", fontsize=12)
    ax2.set_ylabel("Температура PACK (°C)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)
    fig_temp.tight_layout()

    # Построение графика производной температуры PACK с предсказанными отказами
    fig_dtemp, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(df["dTemp"].reset_index(drop=True), label="Производная температуры (dTemp)", color="orange", linewidth=1.5)
    ax3.scatter(df[df["Predicted Failure"] == 1].index,
                df.loc[df["Predicted Failure"] == 1, "dTemp"],
                color="red", label="Предсказан отказ", s=50, zorder=5, alpha=0.7)
    ax3.set_title("Производная температуры PACK с предсказанными отказами", fontsize=14)
    ax3.set_xlabel("Точки данных", fontsize=12)
    ax3.set_ylabel("dTemp (°C/точка)", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(fontsize=10)
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
    return df, fig_roc, fig_temp, fig_dtemp, status, fig_shap


