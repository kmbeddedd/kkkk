import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Satellite Error Predictor", layout="wide")

# Initialize session state variables to persist data across page navigations
if "page" not in st.session_state:
    st.session_state.page = 1
if "model" not in st.session_state:
    st.session_state.model = None
if "historical_df" not in st.session_state:
    st.session_state.historical_df = None


# Mock model loader/trainer (Replace this with your actual model loading logic)
def load_trained_model(training_data):
    # e.g., model = joblib.load("model.pkl") or model.fit(training_data)
    class DummyModel:
        def predict(self, df):
            # Example placeholder prediction: adding noise to simulate predicted error
            return df.select_dtypes(include=["float64", "int64"]) * 1.05

    return DummyModel()


# -----------------------------------------------------------------------------
# PAGE 1: HISTORICAL DATA UPLOAD & MODEL INITIALIZATION
# -----------------------------------------------------------------------------
if st.session_state.page == 1:
    st.title("🛰️ Satellite Error Build-up Predictor")
    st.subheader("Step 1: Upload 7-Day Historical Data")
    st.write(
        "Upload the CSV file containing the initial 7-day satellite clock and ephemeris error parameters."
    )

    uploaded_hist_file = st.file_uploader(
        "Upload 7-Day Data (CSV)", type=["csv"], key="hist_uploader"
    )

    if uploaded_hist_file is not None:
        try:
            hist_df = pd.read_csv(uploaded_hist_file)
            st.session_state.historical_df = hist_df

            st.success("Historical data successfully loaded!")
            st.write("### Data Preview (First 5 Rows):")
            st.dataframe(hist_df.head())

            if st.button("Initialize Model & Proceed to Prediction →"):
                # Link your ML model with the 7-day dataset
                st.session_state.model = load_trained_model(hist_df)
                st.session_state.page = 2
                st.rerun()

        except Exception as e:
            st.error(f"Error reading the file: {e}")

# -----------------------------------------------------------------------------
# PAGE 2: FUTURE DATA UPLOAD & MODEL COMPARISON
# -----------------------------------------------------------------------------
elif st.session_state.page == 2:
    st.title("📊 Model Prediction vs. Ground Truth Comparison")
    st.subheader("Step 2: Upload Future Day Data")

    # Navigation back to Page 1
    if st.button("← Back to Upload Historical Data"):
        st.session_state.page = 1
        st.rerun()

    uploaded_future_file = st.file_uploader(
        "Upload Future Day Data (CSV)", type=["csv"], key="future_uploader"
    )

    if uploaded_future_file is not None:
        try:
            future_df = pd.read_csv(uploaded_future_file)
            st.write("### Ground Truth Data Preview:")
            st.dataframe(future_df.head())

            # Run prediction using the linked model
            predictions = st.session_state.model.predict(future_df)

            st.markdown("---")
            st.subheader("Results & Comparison")

            # Numerical Evaluation Metrics
            st.write("#### Comparison Metrics")
            numeric_cols = future_df.select_dtypes(
                include=["float64", "int64"]
            ).columns

            if len(numeric_cols) > 0:
                selected_col = st.selectbox(
                    "Select Parameter to Compare:", numeric_cols
                )

                actual_vals = future_df[selected_col]
                predicted_vals = predictions[selected_col]

                mae = (actual_vals - predicted_vals).abs().mean()
                col1, col2 = st.columns(2)
                col1.metric("Mean Absolute Error (MAE)", f"{mae:.6f}")
                col2.metric("Total Records Analyzed", f"{len(future_df)}")

                # Graphical Plotting
                st.write(f"#### Visualization: {selected_col}")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(
                    actual_vals.values,
                    label="Actual Future Data",
                    color="blue",
                    linestyle="-",
                )
                ax.plot(
                    predicted_vals.values,
                    label="Predicted Data",
                    color="orange",
                    linestyle="--",
                )
                ax.set_title(f"Actual vs Predicted Build-up Error ({selected_col})")
                ax.set_xlabel("Time Index / Samples")
                ax.set_ylabel("Error Value")
                ax.legend()
                ax.grid(True)

                st.pyplot(fig)
            else:
                st.warning(
                    "No numeric columns found in the uploaded CSV for comparison."
                )

        except Exception as e:
            st.error(f"Error processing future data: {e}")