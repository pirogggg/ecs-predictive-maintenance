import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

def run_prediction(df):
    df.rename(columns={
        "Pack Outlet Temp (Â°C)": "Pack Outlet Temp (°C)",
        "Fan Speed": "Fan Speed (rpm)",
        "Temperature Deviation": "Temp Deviation (°C)",
        "Valve Position": "Valve Position (%)",
        "Bleed Pressure": "Bleed Air Pressure (psi)"
    }, inplace=True)

    # Автоматический расчет Temp Deviation, если не хватает
    if "Temp Deviation (°C)" not in df.columns and        "Cabin Temp Setpoint (°C)" in df.columns and        "Cabin Actual Temp (°C)" in df.columns:
        df["Temp Deviation (°C)"] = df["Cabin Temp Setpoint (°C)"] - df["Cabin Actual Temp (°C)"]

    required_columns = [
        "Pack Outlet Temp (°C)", "Fan Speed (rpm)", "Temp Deviation (°C)",
        "Valve Position (%)", "Bleed Air Pressure (psi)", "Failure in 10h"
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"❌ Отсутствуют колонки: {missing}")

    df["dTemp"] = df["Pack Outlet Temp (°C)"].diff().fillna(0)
    df["Temp_MA10"] = df["Pack Outlet Temp (°C)"].rolling(window=10).mean().bfill()
    df["Valve_range_10"] = df["Valve Position (%)"].rolling(window=10).apply(lambda x: max(x) - min(x)).fillna(0)

    features = [
        "Pack Outlet Temp (°C)", "Fan Speed (rpm)", "Temp Deviation (°C)",
        "Valve Position (%)", "Bleed Air Pressure (psi)",
        "dTemp", "Temp_MA10", "Valve_range_10"
    ]
    X = df[features]
    y = df["Failure in 10h"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight='balanced_subsample',
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
    ax_roc.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_test, y_proba):.2f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax_roc.set_title("ROC-кривая модели")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    ax_roc.grid(True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
    fig_shap = plt.figure()
    shap.summary_plot(shap_values_to_plot, X_test, plot_type="bar", show=False)

    df["Predicted Failure"] = model.predict(X)
    fig_temp, ax_temp = plt.subplots(figsize=(10, 4))
    ax_temp.plot(df["Pack Outlet Temp (°C)"].reset_index(drop=True), label="PACK Temp", color="blue")
    ax_temp.scatter(df[df["Predicted Failure"] == 1].index,
                    df.loc[df["Predicted Failure"] == 1, "Pack Outlet Temp (°C)"],
                    color="red", label="Предсказан отказ", s=30)
    ax_temp.set_title("Температура PACK и предсказанные отказы")
    ax_temp.set_xlabel("Индекс")
    ax_temp.set_ylabel("Температура (°C)")
    ax_temp.grid(True)
    ax_temp.legend()

    fig_dtemp, ax_dtemp = plt.subplots(figsize=(10, 4))
    ax_dtemp.plot(df["dTemp"].reset_index(drop=True), label="dTemp", color="orange")
    ax_dtemp.scatter(df[df["Predicted Failure"] == 1].index,
                     df.loc[df["Predicted Failure"] == 1, "dTemp"],
                     color="red", label="Предсказан отказ", s=30)
    ax_dtemp.set_title("Производная температуры и отказы")
    ax_dtemp.set_xlabel("Индекс")
    ax_dtemp.set_ylabel("dTemp (°C/min)")
    ax_dtemp.grid(True)
    ax_dtemp.legend()

    total_failures = df["Predicted Failure"].sum()
    max_dtemp = df["dTemp"].abs().max()
    dtemp_threshold = 3

    if total_failures > 3:
        status = "🔴 Много потенциальных отказов — требуется техническое вмешательство."
    elif total_failures > 0 or max_dtemp > dtemp_threshold:
        status = "🟡 Есть отклонения — рекомендовано наблюдение."
    else:
        status = "🟢 Всё стабильно — эксплуатация допустима."

    print(classification_report(y_test, y_pred, zero_division=0))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    return df, fig_roc, fig_temp, fig_dtemp, fig_shap, status

