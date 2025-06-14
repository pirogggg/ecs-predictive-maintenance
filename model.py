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
    df["dTemp"] = df["Pack Outlet Temp (°C)"].diff().fillna(0)
    df["Temp_MA10"] = df["Pack Outlet Temp (°C)"].rolling(window=10).mean().fillna(method="bfill")
    # fillna(0) используется для обработки NaN, которые могут появиться в начале после .rolling() и .apply()
    df["Valve_range_10"] = df["Valve Position (%)"].rolling(window=10).apply(lambda x: max(x) - min(x), raw=False).fillna(0)

    features = [
        "Pack Outlet Temp (°C)", # Исправлено название колонки
        "Fan Speed (rpm)",
        "Temp Deviation (°C)",   # Исправлено название колонки
        "Valve Position (%)",
        "Bleed Air Pressure (psi)",
        "dTemp",
        "Temp_MA10",
        "Valve_range_10"
    ]
    
    # Проверка на наличие всех необходимых колонок в загруженном DataFrame
    missing_features = [col for col in features if col not in df.columns]
    # Также проверяем наличие целевой переменной
    if "Failure in 10h" not in df.columns:
        missing_features.append("Failure in 10h")

    if missing_features:
        # Если каких-то колонок не хватает, возвращаем ошибку, чтобы Streamlit мог ее отобразить
        status = f"🔴 Ошибка: Отсутствуют необходимые колонки в файле: {', '.join(missing_features)}. " \
                 f"Пожалуйста, убедитесь, что файл `ecs_data.xlsx` содержит все нужные данные."
        # Возвращаем пустые фигуры и DataFrame, чтобы избежать ошибок в app.py
        # Добавлено plt.figure() для fig_shap, чтобы возвращать 6 элементов, как ожидается в app.py
        return pd.DataFrame(), plt.figure(), plt.figure(), plt.figure(), status, plt.figure()

    X = df[features]
    y = df["Failure in 10h"]

    # *** КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ ОШИБКИ SHAP: Обработка NaN в X перед обучением и SHAP ***
    # Модели машинного обучения и SHAP не работают с пропущенными значениями (NaN).
    # Заполняем все NaN нулями. В реальном проекте может потребоваться более сложная импутация.
    X = X.fillna(0) 

    # Проверка, что в целевой переменной `y` есть хотя бы два класса (0 и 1)
    # Это необходимо для стратифицированного разделения данных и обучения модели классификации.
    if len(y.unique()) < 2:
        status = "🔴 Ошибка: Колонка 'Failure in 10h' должна содержать как минимум два разных значения (например, 0 и 1) " \
                 "для обучения модели. Текущие уникальные значения: " + str(y.unique())
        return pd.DataFrame(), plt.figure(), plt.figure(), plt.figure(), status, plt.figure()

    # Проверка, что достаточно данных для train_test_split после стратификации.
    # Если какой-либо класс имеет менее 2 сэмплов, `stratify=y` вызовет ошибку.
    unique_classes, counts = y.value_counts().index, y.value_counts().values
    if any(count < 2 for count in counts):
         status = "🔴 Ошибка: Недостаточно сэмплов для одного из классов в колонке 'Failure in 10h' " \
                  "для стратифицированного разделения данных. Убедитесь, что каждый класс содержит не менее 2 записей."
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
    fig_roc, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    ax.set_title("ROC-кривая", fontsize=14)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
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
    shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False, ax=ax_shap)
    ax_shap.set_title("Важность признаков (SHAP Values)", fontsize=14)
    fig_shap.tight_layout() # Улучшение компоновки

    # Добавление предсказанных отказов в исходный DataFrame
    # Важно: Используем DataFrame `X` (который уже обработан fillna(0)) для предсказания
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

    # Возвращаем обновленный DataFrame, все объекты фигур Matplotlib и статус
    return df, fig_roc, fig_temp, fig_dtemp, status, fig_shap
