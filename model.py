import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import shap

def run_prediction(df):
 # Унификация названий столбцов
df.rename(columns={
    "Pack Outlet Temp (Â°C)": "Pack Outlet Temp (°C)"
}, inplace=True)

# Проверка
required_columns = [
    "Pack Outlet Temp (°C)", "Fan Speed (rpm)", "Temp Deviation (°C)",
    "Valve Position (%)", "Bleed Air Pressure (psi)", "Failure in 10h"
]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise KeyError(f"❌ Отсутствуют колонки: {missing}")


    # ✅ Создание производных признаков
    df["dTemp"] = df["Pack Outlet Temp (Â°C)"].diff().fillna(0)
    df["Temp_MA10"] = df["Pack Outlet Temp (Â°C)"].rolling(window=10).mean().fillna(method="bfill")
    df["Valve_range_10"] = df["Valve Position (%)"].rolling(window=10).apply(lambda x: max(x) - min(x)).fillna(0)

    # ✅ Обновлённый список признаков
    features = [
        "Pack Outlet Temp (Â°C)", "Fan Speed (rpm)", "Temp Deviation (Â°C)",
        "Valve Position (%)", "Bleed Air Pressure (psi)",
        "dTemp", "Temp_MA10", "Valve_range_10"
    ]
    X = df[features]
    y = df["Failure in 10h"]

    # ✅ Обучение модели
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced_subsample", random_state=42)
    model.fit(X_train, y_train)

    # ✅ Метрики
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)

    # ✅ ROC-кривая
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc = plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая модели")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ✅ SHAP-анализ
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig_shap = plt.figure()
    shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
    plt.tight_layout()

    # ✅ Предсказания по всей выборке
    df["Predicted Failure"] = model.predict(X)

    # ✅ График температуры
    fig_temp = plt.figure(figsize=(12, 5))
    plt.plot(df["Pack Outlet Temp (Â°C)"].reset_index(drop=True), label="PACK Temp", color="blue")
    plt.scatter(
        df[df["Predicted Failure"] == 1].index,
        df.loc[df["Predicted Failure"] == 1, "Pack Outlet Temp (Â°C)"],
        color="red", label="Предсказан отказ", s=30
    )
    plt.title("Температура PACK с предсказанными отказами")
    plt.xlabel("Индекс")
    plt.ylabel("Температура (°C)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ✅ График dTemp
    fig_dtemp = plt.figure(figsize=(12, 5))
    plt.plot(df["dTemp"].reset_index(drop=True), label="d(PACK Temp)/dt", color="orange")
    plt.scatter(
        df[df["Predicted Failure"] == 1].index,
        df.loc[df["Predicted Failure"] == 1, "dTemp"],
        color="red", label="Предсказан отказ", s=30
    )
    plt.title("Производная температуры PACK и предсказания отказов")
    plt.xlabel("Индекс")
    plt.ylabel("dTemp (°C/min)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ✅ Статус оценки
    total_failures = df["Predicted Failure"].sum()
    max_dtemp = df["dTemp"].abs().max()
    dtemp_threshold = 3
    if total_failures > 3:
        status = "🔴 Многократные потенциальные отказы. Ремонт обязателен."
    elif total_failures > 0 or max_dtemp > dtemp_threshold:
        status = "🟡 Аномалии. Наблюдение."
    else:
        status = "🟢 Система стабильна."

    return df, fig_roc, fig_temp, fig_dtemp, fig_shap, status
