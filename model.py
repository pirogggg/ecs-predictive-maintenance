# Установка SHAP (если не установлено)
!pip install shap

# Импорт библиотек
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# Загрузка данных
df = pd.read_excel("ecs_data.xlsx")

# Создание дополнительных признаков
df["dTemp"] = df["Pack Outlet Temp (Â°C)"].diff().fillna(0)
df["Temp_MA10"] = df["Pack Outlet Temp (Â°C)"].rolling(window=10).mean().fillna(method="bfill")
df["Valve_range_10"] = df["Valve Position (%)"].rolling(window=10).apply(lambda x: max(x) - min(x)).fillna(0)

# Обновлённый список признаков
features = [
    "Pack Outlet Temp (Â°C)", "Fan Speed (rpm)", "Temp Deviation (Â°C)",
    "Valve Position (%)", "Bleed Air Pressure (psi)",
    "dTemp", "Temp_MA10", "Valve_range_10"
]
X = df[features]
y = df["Failure in 10h"]

# Разделение выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Обучение модели
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    class_weight='balanced_subsample',
    random_state=42
)
model.fit(X_train, y_train)

# Метрики
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# ROC-кривая
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.2f})")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривая модели")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# SHAP-анализ
# Explicitly pass training data to the explainer
explainer = shap.TreeExplainer(model, data=X_train)
shap_values = explainer.shap_values(X_test)

# Correct slicing for multi-class output and plotting with feature names
# Ensure shap_values has the correct shape (n_samples, n_features, n_classes)
# and select the slice for the positive class (index 1) across all samples and features.
# If the explainer still outputs with 5 features, there might be an issue with how
# the explainer handles models trained on transformed data.
# Let's try slicing based on the expected output for a multi-class classifier with 2 classes.
if isinstance(shap_values, list):
    # For multi-output models, shap_values is a list of arrays.
    # Assuming it's a list with two arrays, one for each class.
    # We need the array for the positive class (index 1).
    if len(shap_values) > 1:
        shap_values_positive_class = shap_values[1]
    else:
        # If it's a single output (regression or binary with single output array)
        shap_values_positive_class = shap_values[0] # Or handle as appropriate
else:
    # For single output models, shap_values is a single array.
    # For binary classification, this array might be (n_samples, n_features)
    # or (n_samples, n_features, 2).
    if shap_values.ndim == 3 and shap_values.shape[2] == 2:
         # If shape is (n_samples, n_features, 2), slice for the positive class
        shap_values_positive_class = shap_values[:, :, 1]
    elif shap_values.ndim == 2:
        # If shape is (n_samples, n_features) for binary classification
        shap_values_positive_class = shap_values
    else:
        # Handle other unexpected shapes
        print(f"Unexpected shap_values shape: {shap_values.shape}")
        shap_values_positive_class = None # Or raise an error

if shap_values_positive_class is not None:
    # Ensure the number of features in shap_values_positive_class matches X_test
    if shap_values_positive_class.shape[1] == X_test.shape[1]:
        shap.summary_plot(shap_values_positive_class, X_test, plot_type="bar", feature_names=features)
    else:
        print(f"Shape mismatch after slicing: shap_values_positive_class.shape={shap_values_positive_class.shape}, X_test.shape={X_test.shape}")
        # Fallback: try plotting with the sliced shap_values and X_test.values if shapes match after slicing
        if shap_values_positive_class.shape == X_test.values.shape:
             shap.summary_plot(shap_values_positive_class, X_test.values, plot_type="bar", feature_names=features)
        else:
             print("Cannot plot SHAP summary due to unresolvable shape mismatch.")


# Временной ряд с предсказаниями
df["Predicted Failure"] = model.predict(X)

plt.figure(figsize=(12, 5))
plt.plot(df["Pack Outlet Temp (Â°C)"].reset_index(drop=True), label="PACK Temp", color="blue")
plt.scatter(
    df[df["Predicted Failure"] == 1].index,
    df.loc[df["Predicted Failure"] == 1, "Pack Outlet Temp (Â°C)"],
    color="red", label="Предсказан отказ", s=30
)
plt.title("Температура PACK с предсказанными отказами")
plt.xlabel("Индекс временной метки")
plt.ylabel("Температура (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Визуализация dTemp с предсказаниями
plt.figure(figsize=(12, 5))
plt.plot(df["dTemp"].reset_index(drop=True), label="d(PACK Temp)/dt", color="orange")
plt.scatter(
    df[df["Predicted Failure"] == 1].index,
    df.loc[df["Predicted Failure"] == 1, "dTemp"],
    color="red", label="Предсказан отказ", s=30
)
plt.title("Производная температуры PACK и предсказания отказов")
plt.xlabel("Индекс временной метки")
plt.ylabel("dTemp (°C/min)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# ------------------------------
# Интеллектуальный вывод по результатам модели
# ------------------------------

# Условия для оценки состояния
total_failures = df["Predicted Failure"].sum()
max_dtemp = df["dTemp"].abs().max()

# Порог чувствительности для dTemp
dtemp_threshold = 3  # °C за 5 сек, подбирается эмпирически

# Логика принятия решения
if total_failures > 3:
    print("🔴 Вывод: Обнаружены многократные признаки потенциального отказа.")
    print("Рекомендация: Необходим ремонт или технический осмотр.")
elif total_failures > 0 or max_dtemp > dtemp_threshold:
    print("🟡 Вывод: Имеются единичные аномалии или нестабильность.")
    print("Рекомендация: Рекомендуется наблюдение за системой.")
else:
    print("🟢 Вывод: Система функционирует стабильно.")
    print("Рекомендация: Продолжить эксплуатацию и регулярное наблюдение.")
