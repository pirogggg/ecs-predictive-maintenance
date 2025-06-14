import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

def run_prediction(df):
    df["dTemp"] = df["Pack Outlet Temp (Â°C)"].diff().fillna(0)
    df["Temp_MA10"] = df["Pack Outlet Temp (Â°C)"].rolling(window=10).mean().fillna(method="bfill")
    df["Valve_range_10"] = df["Valve Position (%)"].rolling(window=10).apply(lambda x: max(x) - min(x)).fillna(0)

    features = [
        "Pack Outlet Temp (Â°C)", "Fan Speed (rpm)", "Temp Deviation (Â°C)",
        "Valve Position (%)", "Bleed Air Pressure (psi)",
        "dTemp", "Temp_MA10", "Valve_range_10"
    ]
    X = df[features]
    y = df["Failure in 10h"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight='balanced_subsample',
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    fig_roc, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC-кривая")
    ax.legend()
    ax.grid(True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)

    df["Predicted Failure"] = model.predict(X)

    fig_temp, ax2 = plt.subplots()
    ax2.plot(df["Pack Outlet Temp (Â°C)"].reset_index(drop=True), label="PACK Temp", color="blue")
    ax2.scatter(df[df["Predicted Failure"] == 1].index,
                df.loc[df["Predicted Failure"] == 1, "Pack Outlet Temp (Â°C)"],
                color="red", label="Предсказан отказ", s=30)
    ax2.set_title("Температура PACK с отказами")
    ax2.grid(True)
    ax2.legend()

    fig_dtemp, ax3 = plt.subplots()
    ax3.plot(df["dTemp"].reset_index(drop=True), label="dTemp", color="orange")
    ax3.scatter(df[df["Predicted Failure"] == 1].index,
                df.loc[df["Predicted Failure"] == 1, "dTemp"],
                color="red", label="Предсказан отказ", s=30)
    ax3.set_title("Производная температуры PACK")
    ax3.grid(True)
    ax3.legend()

    total_failures = df["Predicted Failure"].sum()
    max_dtemp = df["dTemp"].abs().max()
    dtemp_threshold = 3

    if total_failures > 3:
        status = "🔴 Необходим ремонт или технический осмотр"
    elif total_failures > 0 or max_dtemp > dtemp_threshold:
        status = "🟡 Рекомендуется наблюдение за системой"
    else:
        status = "🟢 Система функционирует стабильно"

    return df, fig_roc, fig_temp, fig_dtemp, status
