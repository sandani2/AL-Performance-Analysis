import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# --- Theme & constants ---
MATTE_PRIMARY = ["#34495E"]  # Bar charts
MATTE_PIE = {"Yes": "#34495E", "No": "#BDC3C7"}
GRADE_MAP = {'A':5, 'B':4, 'C':3, 'S':2, 'F':0}

# --- Page Config ---
st.set_page_config(page_title="2020 A/L Analysis", layout="wide")

# --- Load preprocessed CSV ---
try:
    df = pd.read_csv("cleaned_al_data.csv")
except FileNotFoundError:
    st.error("cleaned_al_data.csv not found! Run preprocess_data.py first.")
    st.stop()

grade_map = {'A':5, 'B':4, 'C':3, 'S':2, 'F':0}

@st.cache_resource
def prepare_models(df):
    df_ml = df.dropna(subset=['Zscore']).copy()
    
    # Convert grades to numeric
    for col in ['sub1_r', 'sub2_r', 'sub3_r', 'ge_r']:
        if df_ml[col].dtype == object:
            df_ml[col] = df_ml[col].map(grade_map)
    
    # Encode gender numerically
    df_ml['gender_num'] = df_ml['gender'].map({'male': 0, 'female': 1})
    
    # One-hot encode streams
    df_model = pd.get_dummies(df_ml, columns=['stream'], drop_first=True)
    stream_cols = [c for c in df_model.columns if c.startswith("stream_")]
    
    # Features and target
    features = ['sub1_r','sub2_r','sub3_r','ge_r','cgt_r','gender_num','age'] + stream_cols
    df_model['age'] = df_model['age'].fillna(df_model['age'].median())
    df_model[features] = df_model[features].fillna(0)
    
    X = df_model[features]
    y = df_model['Zscore']
    
    # --- Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    #----- Scale features-------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #------ Ridge Regression-------
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)
    r2 = r2_score(y_test, ridge_model.predict(X_test_scaled))
    mae = mean_absolute_error(y_test, ridge_model.predict(X_test_scaled))
    
    # --- Clustering ---
    cluster_features = ['sub1_r','sub2_r','sub3_r','cgt_r']
    cluster_df = df_model[cluster_features].fillna(0)
    
    scaler_cluster = StandardScaler()
    scaled_cluster = scaler_cluster.fit_transform(cluster_df)
    
    kmeans_final = MiniBatchKMeans(
        n_clusters=3,
        random_state=42,
        batch_size=10000,
        n_init=10
)
    clusters = kmeans_final.fit_predict(scaled_cluster)
    df_model['Cluster'] = clusters
    
    # Map clusters to performance labels
    cluster_summary = df_model.groupby('Cluster')['Zscore'].mean().sort_values()
    sorted_clusters = cluster_summary.index.tolist()
    tier_labels = {
        sorted_clusters[0]: "At-Risk",
        sorted_clusters[1]: "Average",
        sorted_clusters[2]: "High Performer"
    }
    df_model['Performance_Level'] = df_model['Cluster'].map(tier_labels)
    
    # Summary by performance tier
    cluster_summary = df_model.groupby('Performance_Level')['Zscore'].mean().sort_values()
    
    return (
        ridge_model, scaler, scaler_cluster, kmeans_final,
        r2, mae, df_model, stream_cols, features, tier_labels, cluster_summary
    )

# --- Sidebar Navigation ---
page = st.sidebar.radio("Main Menu", ["Home", "Data Explorer","Modelling"])
# --- 3. HOME PAGE ---
if page == "Home":
    st.markdown("""
        <div style="background-color: #34495E; padding: 40px; border-radius: 15px; text-align: center; margin-bottom: 30px;">
            <h1 style="color: white; margin-bottom: 10px;">GCE A/L 2020 Performance Analytics</h1>
            <p style="color: #BDC3C7; font-size: 18px;">Advanced Student Profiling & Z-Score Prediction System</p>
        </div>
    """, unsafe_allow_html=True)

    # 2. Key Project Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Dataset**\n\nOfficial 2020 Examination Results")
    with col2:
        st.info("**Models**\n\nRidge Regression & K-Means Clustering")
    with col3:
        st.info("**Scope**\n\nNational Level Student Performance")

    st.markdown("---")

    # 3. Project Objectives with Feature Cards
    st.subheader(" Project Objectives...")
    
    obj_col1, obj_col2, obj_col3 = st.columns(3)
    
    with obj_col1:
        st.markdown("""
            <div style="border: 1px solid #D6EAF8; padding: 20px; border-radius: 10px; height: 250px;">
                <h4 style="color: #34495E;">Data Exploration</h4>
                <p style="color: #7F8C8D; font-size: 14px;">
                    Interactive visualization of academic streams, gender distributions, and eligibility rates across the island.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with obj_col2:
        st.markdown("""
            <div style="border: 1px solid #D6EAF8; padding: 20px; border-radius: 10px; height: 250px;">
                <h4 style="color: #34495E;">Z-Score Prediction</h4>
                <p style="color: #7F8C8D; font-size: 14px;">
                    Utilizing Ridge Regression to estimate student Z-scores based on subject grades, age, and general test marks.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with obj_col3:
        st.markdown("""
            <div style="border: 1px solid #D6EAF8; padding: 20px; border-radius: 10px; height: 250px;">
                <h4 style="color: #34495E;">Performance Profiling</h4>
                <p style="color: #7F8C8D; font-size: 14px;">
                    Unsupervised machine learning (K-Means) to categorize students into distinct performance tiers for targeted support.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

# --- DATA EXPLORER ---
elif page == "Data Explorer":
    st.title("Data Exploration Dashboard of A/L 2020")
    st.markdown("---")

    # --- FILTER PANEL ---
    st.sidebar.header("Filter Panel")
    with st.sidebar.expander("Adjust Population Filters", expanded=True):
        stream_options = sorted(df["stream"].unique())
        selected_stream = st.multiselect("Academic Stream", options=stream_options)
        selected_gender = st.multiselect("Gender", options=["male", "female"])

    # Apply Filters
    filtered_df = df.copy()
    if selected_stream:
        filtered_df = filtered_df[filtered_df["stream"].isin(selected_stream)]
    if selected_gender:
        filtered_df = filtered_df[filtered_df["gender"].isin(selected_gender)]

    # --- KEY PERFORMANCE INDICATORS (KPIs) ---
    total_students = len(filtered_df)
    elig_count = len(filtered_df[filtered_df['eligible_for_university_entrance'] == 'Yes'])
    elig_rate = (elig_count / total_students * 100) if total_students > 0 else 0
    avg_z = filtered_df["Zscore"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students Analyzed", f"{total_students:,}")
    col2.metric("University Eligibility Rate", f"{elig_rate:.2f}%")
    col3.metric("Average Z-Score", f"{avg_z:.3f}" if not np.isnan(avg_z) else "N/A")

    st.markdown("---")

    # --- ANALYSIS TABS ---
    tab1, tab2 = st.tabs(["Population Distributions", "Performance Relationships"])

    import matplotlib.pyplot as plt

    # -----TAB 1: POPULATION DISTRIBUTIONS------
    with tab1:
        
        # Two columns: Stream distribution & Eligibility
        col1, col2 = st.columns([2, 1])

        # --- Column 1: Stream Distribution ---
        with col1:
            fig, ax = plt.subplots(figsize=(6,4))

            # Use same ordering as in eligibility chart
            stream_order = filtered_df['stream'].value_counts().index
            stream_counts = filtered_df['stream'].value_counts().reindex(stream_order)

            ax.bar(stream_counts.index, stream_counts.values, color=MATTE_PRIMARY[0])
            ax.set_xlabel("Stream")
            ax.set_ylabel("Number of Students")
            ax.set_title("Student Enrollment by Stream")
            ax.set_xticklabels(stream_counts.index, rotation=45, ha='right')

            st.pyplot(fig)
            st.info("**Interpretation:** This chart shows which academic streams are the most popular. A higher bar indicates more students enrolled in that specific field of study. The most number students of from Art stream and the least number of students from Biosystem Technology stream.")

        # --- Column 2: Pie Chart ---
        with col2:
            fig, ax = plt.subplots(figsize=(4,4))
            elig_counts = filtered_df['eligible_for_university_entrance'].value_counts()
            colors = [MATTE_PIE.get(label, "#BDC3C7") for label in elig_counts.index]
            ax.pie(elig_counts, labels=elig_counts.index, autopct='%1.1f%%', colors=colors,
                   startangle=90, wedgeprops={'edgecolor':'white'}, textprops={'fontsize':12})
            ax.set_title("University Eligibility")
            st.pyplot(fig)
            st.info("**Interpretation:** This represents the 'Success Rate.' The dark section shows students who met the minimum requirements for university entrance, while the light section shows those who did not.")

        # --- Next Row: Z-Score & CGT ---
        col3, col4 = st.columns(2)

        # Z-Score histogram
        with col3:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.hist(filtered_df['Zscore'].dropna(), bins=20, color=MATTE_PRIMARY[0], edgecolor='white')
            ax.set_xlabel("Z-Score")
            ax.set_ylabel("Number of Students")
            ax.set_title("Z-Score Distribution")
            st.pyplot(fig)
            st.info("Interpretation: Shows academic competitiveness. Most students fall in the 'peak.' Right-side values represent top national performers; left-side shows lower relative rankings.")

        # CGT histogram
        with col4:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.hist(filtered_df['cgt_r'].dropna(), bins=20, color=MATTE_PRIMARY[0], edgecolor='white')
            ax.set_xlabel("CGT Score")
            ax.set_ylabel("Number of Students")
            ax.set_title("CGT Performance Distribution")
            st.pyplot(fig)
            st.info("Interpretation: Measures general aptitude. This spread reveals that most of the students are centered in the right middle, showing that higher marks or lower marks for the test are rare.")

    # -----------------------------
    # TAB 2: PERFORMANCE RELATIONSHIPS
    # -----------------------------
    with tab2:

        col1, col2 = st.columns(2)

        # Eligibility by Stream
        with col1:
            cross_tab = pd.crosstab(filtered_df['stream'], filtered_df['eligible_for_university_entrance'])
            fig, ax = plt.subplots(figsize=(6,4))
            cross_tab.plot(kind='bar', stacked=True, ax=ax, color=[MATTE_PIE.get("Yes"), MATTE_PIE.get("No")])
            ax.set_xlabel("Stream")
            ax.set_ylabel("Number of Students")
            ax.set_title("Eligibility Success by Stream")
            ax.legend(title="Eligible")
            st.pyplot(fig)
            st.info("**Interpretation:** This compares passing rates across different subjects. It helps identify which academic streams have the highest or lowest proportions of university-eligible students. passing rate of physical science students is high compare to others")

        # English Grades
        with col2:
            temp_ge = filtered_df.copy()
            if temp_ge['ge_r'].dtype != object:
                inv_grade_map = {v:k for k,v in GRADE_MAP.items()}
                temp_ge['ge_label'] = temp_ge['ge_r'].map(inv_grade_map)
            else:
                temp_ge['ge_label'] = temp_ge['ge_r']

            temp_ge = temp_ge[~temp_ge['ge_label'].astype(str).str.upper().isin(['WH', 'WITHHELD', 'NAN', 'NONE'])]
            ge_counts = temp_ge['ge_label'].value_counts().reindex(['A','B','C','S','F']).fillna(0)

            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(ge_counts.index, ge_counts.values, color=MATTE_PRIMARY[0])
            ax.set_xlabel("English Grade")
            ax.set_ylabel("Number of Students")
            ax.set_title("General English Performance")
            st.pyplot(fig)
            st.info("**Interpretation:** This distribution shows student performance in General English. larger number of students failed General English Test and the students who obtained 'A' pass comparatively low.")

# Move these to the TOP of your script (outside any if/else)
grade_map = {'A': 5, 'B': 4, 'C': 3, 'S': 2, 'F': 0}

 # ---  MODELLING ---

if page =="Modelling":
    @st.cache_resource
    def load_models():
        return prepare_models(df)

    (ridge_model, scaler, scaler_cluster, kmeans_final, r2, mae,
     df_model, stream_cols, features, tier_labels, cluster_summary) = load_models()
    st.subheader(" Prediction and clustering")
    met1, met2 = st.columns(2)
    with met1:
        st.metric("R² Score", f"{r2:.4f}")
        st.markdown(f"""
            <p style='font-size: 13px; color: #5D6D7E;'>
            <b>Predictive Reliability:</b> This model explains <b>{r2*100:.1f}%</b> of the variance in Z-Scores. 
            In academic modeling, an R² above 0.70 is considered highly robust.
            </p>
        """, unsafe_allow_html=True)
        
    with met2:
        st.metric("Mean Absolute Error", f"{mae:.4f}")
        st.markdown(f"""
            <p style='font-size: 13px; color: #5D6D7E;'>
            <b>Error Margin:</b> On average, the predicted Z-Score deviates by only <b>{mae:.4f}</b> 
            units. This allows for high-confidence tier placement.
            </p>
        """, unsafe_allow_html=True)

    # --- PERFORMANCE CLUSTERING VISUALS ---
    st.divider()
    st.subheader("Performance Clustering Analysis")

    # Tables with Integrated Interpretation
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Average Z-Score per Cluster**")
        st.dataframe(cluster_summary.rename("Mean Z-Score"), use_container_width=True)
        st.caption("**Analysis:** Cluster 2 shows the highest performance (mean Z = 1.1541), indicating above-average students. Cluster 1 is close to average (0.1647). Cluster 0 has below-average performance (-0.396).")

    with col_right:
        st.markdown("**Cluster Population Distribution**")
        counts = df_model['Cluster'].value_counts().sort_index().rename("Student Count")
        st.dataframe(counts, use_container_width=True)
        st.caption("**Analysis:** Cluster 1 has the largest number of students (80,108), while Cluster 2 has the smallest (64,442). This shows most students fall into the average-performance group")

    # Scatter Plot with Interpretation on the Right
    st.markdown("---")
    viz_col, desc_col = st.columns([2, 1])

    with viz_col:
        fig, ax = plt.subplots(figsize=(8,6))
    
        # Define colors for each performance level
        color_map = {
    "At-Risk": "#808080",        # Gray
    "Average": "#001F3F",        # Navy Blue
    "High Performer": "#000000"  # Black
}
    
        # Plot points for each performance level
        for level, color in color_map.items():
            subset = df_model[df_model['Performance_Level'] == level]
            ax.scatter(subset['Zscore'], subset['cgt_r'], 
                       label=level, color=color, alpha=0.6, edgecolors='w', s=80)
    
        # Labels and title
        ax.set_xlabel("Academic Z-Score")
        ax.set_ylabel("Aptitude (CGT)")
        ax.set_title("Institutional Academic Mapping")
        ax.legend(title="Performance Level")
        ax.grid(True, linestyle='--', alpha=0.5)
    
        st.pyplot(fig)

    with desc_col:
        st.markdown("**Insights**")
        st.info("""
            Students with higher academic scores (right side of the chart) mostly belong to the High Performer group.
            Students with lower academic scores (left side) are mostly in the At-Risk group.
            The Average group falls in the middle range.
        """)

    # --- 4️PREDICTION UI ---
    st.divider()
    st.subheader("Predict & Classify Your Performance")

    stream_options = sorted(df['stream'].dropna().unique())
    col1, col2 = st.columns(2)

    with col1:
        selected_stream = st.selectbox("Select Stream", stream_options)
        g1 = st.selectbox("Subject 1 Grade", ['A','B','C','S','F'])
        g2 = st.selectbox("Subject 2 Grade", ['A','B','C','S','F'])
        g3 = st.selectbox("Subject 3 Grade", ['A','B','C','S','F'])

    with col2:
        ge = st.selectbox("General English Grade", ['A','B','C','S','F'])
        gender_input = st.selectbox("Gender", ["male","female"])
        age_input = st.slider("Age", 17, 30, 19)
        cgt_input = st.slider("CGT Mark", 0, 100, 50)

    if st.button("Predict & Classify"):
        # Prediction Logic
        input_dict = {
            'sub1_r': grade_map[g1], 'sub2_r': grade_map[g2], 'sub3_r': grade_map[g3],
            'ge_r': grade_map[ge], 'cgt_r': cgt_input, 
            'gender_num': 1 if gender_input == "female" else 0, 'age': age_input
        }
        for col in stream_cols:
            input_dict[col] = 1 if col == f"stream_{selected_stream}" else 0

        input_df = pd.DataFrame([input_dict])[features]
        input_scaled = scaler.transform(input_df)
        prediction = ridge_model.predict(input_scaled)[0]

        # Results Display
        st.metric("Estimated Z-score", f"{prediction:.4f}")
        
        # Cluster Prediction
        cluster_input = [[grade_map[g1], grade_map[g2], grade_map[g3], cgt_input]]
        cluster_input_scaled = scaler_cluster.transform(cluster_input)
        student_cluster = kmeans_final.predict(cluster_input_scaled)[0]
        performance_label = tier_labels[student_cluster]

        st.subheader("Current Performance Status")
        st.markdown(f"### {performance_label}")

        if "At-Risk" in performance_label:
            st.error("Student may need academic support.")
        elif "Average" in performance_label:
            st.warning("Student is performing moderately.")
        else:
            st.success("Student is a high performer.")

        # --- STRATEGIC INTERPRETATION ---
        st.divider()
        st.subheader("Strategic Academic Interpretation")

        if "High Performer" in performance_label:
            status_color = "#2ECC71" 
            insight = "The model identifies this student as highly competitive."
            recommendation = "Maintain current study methodologies and focus on securing high-priority university placements."
            drivers = "Strong core subject results combined with high aptitude indicator."
        elif "Average" in performance_label:
            status_color = "#F1C40F" 
            insight = "The model places this student in the median performance tier."
            recommendation = "Targeted improvement in one core subject could significantly boost the Z-score."
            drivers = "Balanced results but requires a 'competitive edge' in high-weightage subjects."
        else:
            status_color = "#E74C3C"
            insight = "The model flags this profile for academic intervention."
            recommendation = "Recommend structured remedial sessions and a review of foundational concepts."
            drivers = "Significant gaps in core subject performance or low aptitude marks."

        st.markdown(f"""
            <div style="border-left: 5px solid {status_color}; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
                <h4 style="color: {status_color}; margin-top: 0;">Performance Driver Analysis</h4>
                <p><strong>Primary Insight:</strong> {insight}</p>
                <p><strong>Key Drivers:</strong> {drivers}</p>
                <p><strong>Strategic Recommendation:</strong> {recommendation}</p>
            </div>
        """, unsafe_allow_html=True)
# -----------ASSUMPTIONS & LIMITATIONS-----------
    st.divider()
    st.subheader("Assumptions & Limitations")

    with st.expander("Model Assumptions", expanded=False):
        st.markdown("""
    **Ridge Regression Assumptions**
    - Linear relationship between subject grades, CGT, and Z-score.
    - Observations (students) are independent.
    - Regularization (L2 penalty) reduces multicollinearity among predictors.
    
    **Clustering Assumptions**
    - Student performance naturally forms three groups (At-Risk, Average, High Performer).
    - Euclidean distance is appropriate for similarity measurement.
    - Features are standardized to ensure equal importance.
    """)

    with st.expander("Study Limitations", expanded=False):
        st.markdown("""
    - Analysis is based only on 2020 A/L data; results may not generalize to other years.
    - Socioeconomic, school-level, and regional variables were not included.
    - Clustering groups are relative to dataset distribution, not absolute academic standards.
    - Predicted Z-scores are statistical estimates and not official examination outcomes.
    """)