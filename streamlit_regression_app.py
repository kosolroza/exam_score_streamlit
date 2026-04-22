from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def build_demo_data(seed: int = 42, n_samples: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    score = rng.uniform(30, 95, n_samples)
    study_hours = rng.uniform(0, 10, n_samples)
    noise = rng.normal(0, 0.18, n_samples)
    gpa = 0.018 * score + 0.12 * study_hours + 1.0 + noise
    gpa = np.clip(gpa, 0, 4)

    return pd.DataFrame(
        {
            "Score": np.round(score, 2),
            "Study Hours": np.round(study_hours, 2),
            "GPA": np.round(gpa, 2),
        }
    )


def train_model(data: pd.DataFrame) -> tuple[LinearRegression, np.ndarray, np.ndarray, float, float, float]:
    x_values = data[["Score", "Study Hours"]].to_numpy()
    y_values = data["GPA"].to_numpy()

    model = LinearRegression()
    model.fit(x_values, y_values)
    predictions = model.predict(x_values)

    return (
        model,
        x_values,
        predictions,
        r2_score(y_values, predictions),
        mean_absolute_error(y_values, predictions),
        mean_squared_error(y_values, predictions),
    )


def render_app() -> None:
    st.set_page_config(page_title="Linear Regression Demo", page_icon="📈", layout="wide")

    st.title("📈 Linear Regression Mini Project")
    st.write(
        "This small Streamlit project predicts student GPA using two inputs: "
        "current score and study hours."
    )

    st.sidebar.header("Settings")
    seed = st.sidebar.slider("Random seed", 0, 100, 42)
    n_samples = st.sidebar.slider("Number of samples", 20, 200, 100)

    data = build_demo_data(seed=seed, n_samples=n_samples)
    model, x_values, predictions, r2, mae, mse = train_model(data)

    st.subheader("Dataset Preview")
    st.dataframe(data, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("R² score", f"{r2:.3f}")
    col2.metric("MAE", f"{mae:.2f}")
    col3.metric("MSE", f"{mse:.2f}")

    st.subheader("Model")
    st.write(f"Intercept: **{model.intercept_:.2f}**")
    st.write(f"Coefficient for Score: **{model.coef_[0]:.2f}**")
    st.write(f"Coefficient for Study Hours: **{model.coef_[1]:.2f}**")

    st.subheader("Predict GPA")
    input_score = st.number_input("Enter score", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
    input_hours = st.number_input("Enter study hours", min_value=0.0, max_value=24.0, value=4.0, step=0.5)

    predicted_gpa = model.predict(np.array([[input_score, input_hours]]))[0]
    predicted_gpa = float(np.clip(predicted_gpa, 0, 4))
    st.success(
        f"Predicted GPA for score {input_score:.1f} and study hours {input_hours:.1f}: {predicted_gpa:.2f}"
    )

    st.subheader("Quick Explanation")
    st.markdown(
        "- Linear regression fits a straight line to data.\n"
        "- Here we use two features: score and study hours.\n"
        "- The prediction above estimates GPA from both user inputs."
    )

    st.subheader("Optional Visualization")
    chart_data = data.copy()
    chart_data["Predicted GPA"] = predictions
    chart_data = chart_data.sort_values("Study Hours")
    st.line_chart(chart_data.set_index("Study Hours")[["GPA", "Predicted GPA"]])


if __name__ == "__main__":
    render_app()