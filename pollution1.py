import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

st.title("💧 Water Pollution & Disease Predictor")
st.write("Train a machine learning model to predict **Diarrheal Cases per 100,000 people** based on water quality, environment, and infrastructure data.")

# Upload CSV file
uploaded_file = st.file_uploader("📁 Upload your dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    # Target column
    target = 'Diarrheal Cases per 100,000 people'

    if target not in df.columns:
        st.error(f"❌ Column '{target}' not found in the dataset.")
    else:
        # Features and labels
        X = df.drop(columns=[target])
        y = df[target]

        # Detect column types
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Remove 'Year' if not helpful
        if 'Year' in numerical_cols:
            numerical_cols.remove('Year')

        # Preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )

        # Pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("📊 Model Performance")
        st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
        st.metric("R² Score", f"{r2:.2f}")

        st.success("✅ Model training complete!")

        # Optional: Download predictions
        if st.checkbox("Show predictions on test data"):
            results = X_test.copy()
            results[target] = y_test
            results['Predicted'] = y_pred
            st.dataframe(results.head())

            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime='text/csv'
            )

