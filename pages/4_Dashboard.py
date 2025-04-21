import streamlit as st
import sys
import os
import re
import time
from random import randint
from collections import Counter

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Data Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Keep sidebar collapsed by default
)

# Set flag to prevent app.py from rendering its UI when imported
os.environ["IMPORTING_ONLY"] = "1"

# Now import other modules that might use Streamlit
from st_aggrid import AgGrid, GridOptionsBuilder
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json
from io import BytesIO
from utils.theme import apply_custom_theme, feature_card
# Import from app.py after setting the environment variable
from app import (
    calculate_column_score,
    calculate_pair_score,
    calculate_triple_score,
    get_visualization_recommendation,
    score_all_columns_and_pairs,
    score_groupby_column,
    score_aggregation_column,
    calculate_groupby_pair_score,
    get_groupby_visualization_recommendation,
    visualize_triple,
    convert_to_datetime,
    visualize_groupby
)
from visualization_decision import (
    is_temporal_column,
    get_vis_type_for_single_column,
    get_vis_type_for_pair,
    get_vis_type_for_triple,
    get_vis_type_for_groupby,
    get_vis_type_for_groupby_pair
)
from utils.translations import get_translation_function

# Apply custom theme
apply_custom_theme()

# Get translation function
t = get_translation_function()

# Define domain keywords dictionary
domain_keywords = {
    "Business & Finance": ["revenue", "profit", "sales", "cost", "margin", "product", "inventory", "customer", "price"],
    "Healthcare": ["patient", "disease", "treatment", "diagnosis", "hospital", "doctor", "medical", "health"],
    "Education": ["student", "course", "grade", "teacher", "school", "university", "education", "learning"],
    "Science & Research": ["experiment", "measurement", "observation", "variable", "correlation", "hypothesis", "sample"],
    "Marketing": ["campaign", "audience", "conversion", "engagement", "channel", "ad", "social", "brand"],
    "Human Resources": ["employee", "salary", "hire", "performance", "department", "role", "position"],
    "Transportation": ["vehicle", "distance", "route", "destination", "travel", "transport", "delivery"],
    "E-commerce": ["product", "order", "customer", "cart", "purchase", "shipping", "review", "item"],
    "Social Media": ["user", "post", "engagement", "follower", "comment", "platform", "share", "like"],
    "Real Estate": ["property", "price", "location", "area", "sale", "rent", "agent", "housing"]
}

# Add CSS for better dropdown formatting
st.markdown("""
<style>
/* Language selector styling - only apply to the language selector */
div.language-selector-container div[data-testid="stSelectbox"] {
    max-width: 200px;
    margin-left: auto;
}

/* Ensure text in buttons is white */
.stButton button {
    color: white !important;
}

.stButton button p, 
.stButton button span, 
.stButton button div {
    color: white !important;
}

/* Tab button styling */
button[data-baseweb="tab"] {
    padding: 10px 15px;
}

/* Make all buttons more visible */
.stButton button {
    margin-bottom: 10px;
}

/* Reduce white space at the top of the page */
.block-container {
    padding-top: 1rem !important;
}

/* Make step indicators more compact */
.step-indicator {
    margin-top: 0 !important;
    margin-bottom: 0.5rem !important;
    padding: 0 !important;
    font-size: 1rem !important;
}

/* Reduce space around progress bar */
.stProgress {
    margin-top: 0.5rem !important;
    margin-bottom: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'dashboard_step' not in st.session_state:
    st.session_state.dashboard_step = 1
    
if 'domain' not in st.session_state:
    st.session_state.domain = None
    
if 'top3' not in st.session_state:
    st.session_state.top3 = []
    
if 'viz_recommendations' not in st.session_state:
    st.session_state.viz_recommendations = None

# Add language selector at the top
language_col1, language_col2 = st.columns([6, 1])
with language_col2:
    # Add CSS to ensure language selector stays properly positioned but doesn't affect other selectboxes
    st.markdown("""
    <style>
    /* Fix language selector positioning */
    div.language-selector-container div[data-testid="stSelectbox"] {
        width: 150px;
        float: right;
        max-width: 100%;
        margin-right: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Wrap the language selector in a div with a specific class for targeting
    st.markdown('<div class="language-selector-container">', unsafe_allow_html=True)
    selected_lang = st.selectbox(
        t('language_selector'),
        options=["English", "Français"],
        index=0 if st.session_state.get("language", "en") == "en" else 1,
        key="language_selector",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Update language when user changes selection
    if (selected_lang == "English" and st.session_state.get("language", "en") != "en") or \
       (selected_lang == "Français" and st.session_state.get("language", "en") != "fr"):
        st.session_state.language = "en" if selected_lang == "English" else "fr"
        st.rerun()

# Add custom CSS for button text
st.markdown("""
    <style>
    /* Force white text in all parts of buttons */
    .stButton button {
        color: white !important;
    }
    
    .stButton button p, 
    .stButton button span, 
    .stButton button div {
        color: white !important;
    }
    
    /* Target button text specifically */
    button[kind="primary"] p,
    [data-testid^="stButton"] p,
    .stButton p {
        color: white !important;
    }
    
    /* Aggregation type styling */
    .mean-bg {
        background-color: #e6f3ff; 
        color: #0066cc;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    
    .sum-bg {
        background-color: #fff2e6; 
        color: #cc6600;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    
    .count-bg {
        background-color: #e6ffe6; 
        color: #006600;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    
    /* Arrow navigation styling */
    .arrow-button {
        background-color: #f0f2f6;
        border: 1px solid #dfe1e6;
        border-radius: 4px;
        color: #36454F;
        cursor: pointer;
        font-size: 18px;
        padding: 8px 12px;
        transition: background-color 0.3s;
    }
    
    .arrow-button:hover:not([disabled]) {
        background-color: #e6e9ef;
    }
    
    .arrow-button[disabled] {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    /* Styling for navigation buttons */
    button[key="prev_alt"], button[key="next_alt"] {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: white !important;
        font-size: 18px !important;
        width: 45px !important;
        height: 40px !important;
        border-radius: 0.5rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        line-height: 1 !important;
        margin: 0 auto !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    
    /* Add spacing for columns containing navigation buttons */
    [data-testid="column"] button[key="prev_alt"] {
        margin-right: 10px !important;
    }
    
    [data-testid="column"] button[key="next_alt"] {
        margin-left: 10px !important;
    }
    
    /* Target the button's text element to ensure proper centering */
    button[key="prev_alt"] p, button[key="next_alt"] p {
        margin: 0 !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        color: white !important;
    }
    
    button[key="prev_alt"]:hover:not([disabled]), button[key="next_alt"]:hover:not([disabled]) {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
    }
    
    /* Alternative display styling */
    .alternative-display {
        margin: 15px auto;
        padding: 15px;
        background-color: #f0f7ff;
        border-radius: 5px;
        text-align: center;
        max-width: 600px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Consistent button styling - updated to match Home.py */
    .stButton > button {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.025em !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
        color: white !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
        color: white !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%) !important;
    }
    
    /* Wide button for alternatives */
    .wide-button {
        min-width: 200px;
    }
    
    /* Metrics table styling */
    .metrics-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        text-align: center;
        table-layout: fixed;
    }
    
    .metrics-table th {
        background-color: #f0f2f6;
        padding: 12px;
        border-bottom: 2px solid #ddd;
        font-weight: bold;
    }
    
    .metrics-table td {
        padding: 10px;
        border-bottom: 1px solid #eee;
    }
    
    /* Center browse alternatives section */
    .browse-alternatives-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        max-width: 600px;
        width: 100%;
    }
    
    /* Flex container for arrows */
    .arrow-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 30px;
        margin: 10px auto;
        width: 120px;
    }
    
    /* Use Alternative button */
    .use-alternative-btn {
        margin: 15px auto;
        display: block;
    }
    
    /* Centered dropdown */
    .centered-dropdown {
        margin: 0 auto;
        max-width: 400px;
        text-align: center;
    }
    
    /* Override Streamlit's default styles to center elements in alternatives section */
    .browse-alternatives .stButton {
        display: flex;
        justify-content: center;
    }
    
    .stExpander {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Ensure elements inside expander are centered */
    .stExpander > div > div:nth-child(2) {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }
    
    /* Fix button alignment in columns */
    [data-testid="column"] .stButton {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    
    /* Clean up obsolete styles and focus on the simplified approach */
    .alternative-nav {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        margin: 10px auto;
    }
    
    /* Use alternative button styling */
    button[key="use_alternative"] {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        margin: 15px auto !important;
        display: block !important;
        padding: 0.75rem 1.5rem !important;
        letter-spacing: 0.025em !important;
        transition: all 0.3s ease !important;
    }
    
    button[key="use_alternative"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
    }
    
    /* Center the table and give it fixed width */
    .styled-table-container {
        margin: 0 auto;
        max-width: 800px;
        width: 100%;
    }
    
    /* New class for centering visualization header and table */
    .viz-recommendations-container {
        max-width: 900px;
        margin: 0 auto;
        text-align: center;
    }
    
    /* Metrics panel styling - horizontal layout */
    .metrics-panel {
        display: flex;
        flex-direction: row;
        flex-wrap: nowrap;
        justify-content: space-between;
        gap: 10px; /* Reduced from 15px */
        background-color: transparent;
        padding: 0;
        margin-bottom: 10px; /* Reduced from 20px */
        width: 100%;
    }
    
    /* Section titles */
    .section-title {
        font-size: 18px;
        font-weight: 600;
        margin: 8px 0; /* Reduced from 10px 0 */
        padding-bottom: 5px; /* Reduced from 8px */
        border-bottom: 2px solid #f0f2f6;
    }
    
    /* Visualization grid layout */
    .viz-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 15px;
        margin-bottom: 20px;
    }
    
    /* Individual visualization container */
    .viz-container {
        background-color: transparent;
        border-radius: 0;
        padding: 0;
        box-shadow: none;
        margin-bottom: 0;
    }
    
    .viz-container:hover {
        transform: none;
        box-shadow: none;
    }
    
    /* Make plot titles centered */
    .js-plotly-plot .plotly .main-svg .infolayer .g-gtitle {
        text-anchor: middle !important;
    }
    
    /* Visualization title and subtitle */
    .viz-title {
        font-size: 16px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 5px;
        color: #333;
    }
    
    .viz-subtitle {
        font-size: 13px;
        color: #666;
        text-align: center;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #f0f2f6;
    }
    
    /* Metric card styling */
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 12px 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
        flex: 1 1 0;
        min-width: 0;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .metric-card.mean-bg {
        border-left-color: #2196F3;
    }
    
    .metric-card.sum-bg {
        border-left-color: #FF9800;
    }
    
    .metric-card.count-bg {
        border-left-color: #9C27B0;
    }
    
    .metric-label {
        font-size: 14px;
        color: #666;
        margin-bottom: 5px;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 600;
        color: #333;
    }
    
    .metric-desc {
        font-size: 12px;
        color: #888;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Domain keywords for context selection
domain_keywords = { 
    "General": {},
    "Sales / Ventes": {
        "sales": 2, "revenue": 2, "profit": 2, "quantity": 2, "cost": 1.5, "amount": 1.5, "turnover": 2, "commission": 1.5,
        "vente": 2, "chiffre d'affaires": 2, "bénéfice": 2, "quantité": 2, "coût": 1.5, "montant": 1.5, "commission": 1.5
    },
    "Logistics / Logistique": {
        "weight": 1.5, "distance": 1.5, "time": 1.5, "shipping": 2, "delivery": 2, "transport": 1.5, "stock": 1.5, "warehouse": 1.5,
        "poids": 1.5, "distance": 1.5, "temps": 1.5, "expédition": 2, "livraison": 2, "transport": 1.5, "stock": 1.5, "entrepôt": 1.5
    },
    "Maintenance": {
        "repair": 1.5, "cost": 2, "hours": 1.5, "downtime": 2, "failure": 1.5, "service": 1.5, "inspection": 1.5, "parts": 1.5,
        "réparation": 1.5, "coût": 2, "heures": 1.5, "temps d'arrêt": 2, "panne": 1.5, "service": 1.5, "inspection": 1.5, "pièces": 1.5
    },
    "Finance": {
        "income": 2, "expense": 2, "investment": 2, "loan": 1.5, "interest": 1.5, "tax": 1.5, "credit": 1.5, "debt": 1.5, "salary": 2,
        "revenu": 2, "dépense": 2, "investissement": 2, "prêt": 1.5, "intérêt": 1.5, "impôt": 1.5, "crédit": 1.5, "dette": 1.5, "salaire": 2
    },
    "Education / Éducation": {
        "score": 2, "grade": 2, "attendance": 1.5, "study": 1.5, "tuition": 2, "exam": 1.5, "homework": 1.5, "learning": 2, "GPA": 2,
        "note": 2, "classement": 2, "présence": 1.5, "étude": 1.5, "frais de scolarité": 2, "examen": 1.5, "devoir": 1.5, "apprentissage": 2
    },
    "Health / Santé": {
        "heart rate": 2, "blood pressure": 2, "cholesterol": 1.5, "bmi": 1.5, "steps": 1.5, "calories": 1.5, "exercise": 1.5, "sleep": 2, "weight": 1.5,
        "rythme cardiaque": 2, "pression artérielle": 2, "cholestérol": 1.5, "IMC": 1.5, "pas": 1.5, "calories": 1.5, "exercice": 1.5, "sommeil": 2, "poids": 1.5
    },
    "Social Media / Réseaux Sociaux": {
        "likes": 2, "shares": 2, "followers": 1.5, "comments": 1.5, "engagement": 2, "posts": 1.5, "views": 1.5, "subscribers": 1.5,
        "mentions j'aime": 2, "partages": 2, "abonnés": 1.5, "commentaires": 1.5, "engagement": 2, "publications": 1.5, "vues": 1.5, "inscrits": 1.5
    },
    "Production": {
        "units": 2, "efficiency": 2, "defects": 1.5, "yield": 1.5, "downtime": 1.5, "productivity": 2, "output": 2, "manufacturing": 2,"production":2, "stocks": 1.5,
        "unités": 2, "efficacité": 2, "défauts": 1.5, "rendement": 1.5, "temps d'arrêt": 1.5, "productivité": 2, "production": 2, "fabrication": 2
    },
    "E-commerce": {
        "sales": 2, "orders": 2, "cart": 1.5, "conversion": 2, "return": 1.5, "discount": 1.5, "customer": 1.5, "rating": 1.5, "reviews": 1.5,
        "ventes": 2, "commandes": 2, "panier": 1.5, "conversion": 2, "retour": 1.5, "réduction": 1.5, "client": 1.5, "évaluation": 1.5, "avis": 1.5
    },
    "Energy / Énergie": {
        "power": 2, "consumption": 2, "fuel": 1.5, "electricity": 2, "gas": 1.5, "efficiency": 1.5, "renewable": 2, "solar": 2, "wind": 2,
        "puissance": 2, "consommation": 2, "carburant": 1.5, "électricité": 2, "gaz": 1.5, "efficacité": 1.5, "renouvelable": 2, "solaire": 2, "éolien": 2
    },
    "Real Estate / Immobilier": {
        "property": 2, "price": 2, "rent": 2, "mortgage": 2, "investment": 2, "square footage": 1.5, "valuation": 1.5,
        "propriété": 2, "prix": 2, "loyer": 2, "hypothèque": 2, "investissement": 2, "superficie": 1.5, "évaluation": 1.5
    },
    "Human Resources / Ressources Humaines": {
        "salary": 2, "bonus": 2, "hiring": 2, "promotion": 2, "benefits": 1.5, "training": 1.5, "recruitment": 2,
        "salaire": 2, "prime": 2, "embauche": 2, "promotion": 2, "avantages": 1.5, "formation": 1.5, "recrutement": 2
    },
    "Technology / Informatique": {
        "server": 2, "cloud": 2, "AI": 2, "algorithm": 1.5, "machine learning": 2, "CPU": 1.5, "GPU": 1.5, "latency": 1.5,
        "serveur": 2, "cloud": 2, "IA": 2, "algorithme": 1.5, "apprentissage automatique": 2, "processeur": 1.5, "graphique": 1.5, "latence": 1.5
    }
}

# Helper functions for metrics recommendation
def detect_id_column(column_name):
    id_keywords = ["id", "code", "number", "uuid", "identifier", "reference", "index", "key"]
    return any(keyword in column_name.lower() for keyword in id_keywords)

def calculate_entropy(series):
    binned = pd.qcut(series, q=min(10, len(series.unique())), duplicates='drop')
    counts = binned.value_counts(normalize=True)
    return entropy(counts, base=2)

def clean_column_name(col_name):
    col_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', col_name)
    col_name = col_name.replace("_", " ")
    return col_name.lower().strip()

def business_relevance_boost(col_name, selected_domain):
    if selected_domain in domain_keywords:
        col_name = clean_column_name(col_name)
        boost = sum(weight for keyword, weight in domain_keywords[selected_domain].items() if keyword in col_name)
        return boost
    return 0

def suggest_aggregation(series, col_name):
    unique_ratio = series.nunique() / len(series)
    col_name_clean = clean_column_name(col_name).lower()

    # Rule 1: ID columns → count
    if detect_id_column(col_name):
        return 'count'
    
    # Rule 2: Force sum if column has 'total' or 'sum'
    if 'total' in col_name_clean or 'sum' in col_name_clean:
        return 'sum'
    
    # Rule 3: Force mean if column has 'rate'
    if 'rate' in col_name_clean:
        return 'mean'

    # Rule 4: Special keywords → mean
    if any(keyword in col_name_clean for keyword in ['age', 'price', 'rating', 'score', 'percent']):
        return 'mean'

    # Rule 5: Unique ratio based decisions
    if unique_ratio < 0.05:
        return 'count'
    elif unique_ratio > 0.95:
        return 'mean'
    else:
        # Rule 6: Mostly positive numeric → sum
        if pd.api.types.is_numeric_dtype(series) and (series > 0).mean() > 0.9:
            return 'sum'
        else:
            return 'mean'

def rank_columns(df, selected_domain):
    # Clean column names in the DataFrame
    df.columns = [clean_column_name(col) for col in df.columns]
    
    numerical_cols = [col for col in df.select_dtypes(include=['number']).columns if not detect_id_column(col)]
    scores = []

    for col in numerical_cols:
        series = df[col].dropna()
        if len(series) < 5:
            continue

        col_cleaned = clean_column_name(col)

        cv = series.std() / series.mean() if series.mean() != 0 else 0
        skw = abs(skew(series))
        krt = abs(kurtosis(series))
        ent = calculate_entropy(series)
        uniq = len(series.unique()) / len(series)
        outlier = np.clip((series > series.mean() + 3 * series.std()).sum() / len(series), 0, 1)

        norm_cv = np.tanh(cv)
        norm_skw = np.tanh(skw / 10)
        norm_krt = np.tanh(krt / 15)
        norm_ent = ent / np.log(len(series.unique()) + 1)
        norm_uniq = np.sqrt(uniq)
        norm_outlier = outlier

        score = (
            1.5 * norm_cv +
            1.2 * norm_skw +
            1.0 * norm_krt +
            0.8 * norm_ent +
            0.7 * norm_uniq +
            0.6 * norm_outlier
        )

        boost = business_relevance_boost(col_cleaned, selected_domain)
        score += boost

        suggested_agg = suggest_aggregation(series, col)

        scores.append({
            'Column': col,
            'CV_Score': round(norm_cv * 1.5, 3),
            'Skew_Score': round(norm_skw * 1.2, 3),
            'Kurtosis_Score': round(norm_krt * 1.0, 3),
            'Entropy_Score': round(norm_ent * 0.8, 3),
            'Uniqueness_Score': round(norm_uniq * 0.7, 3),
            'Outlier_Score': round(norm_outlier * 0.6, 3),
            'Business_Relevance_Boost': boost,
            'Final_Score': round(score, 3),
            'Suggested_Aggregation': suggested_agg
        })

    sorted_scores = sorted(scores, key=lambda x: x['Final_Score'], reverse=True)
    return sorted_scores[:3], sorted_scores

# Initialize all required session state variables
if "language" not in st.session_state:
    st.session_state.language = "en"
if "progress" not in st.session_state:
    st.session_state.progress = {'upload': True, 'process': True, 'clean': True, 'visualize': True}
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "cleaned_dataframes" not in st.session_state:
    st.session_state.cleaned_dataframes = {}
if "data_analyzed" not in st.session_state:
    st.session_state.data_analyzed = False
if "data_quality_scores" not in st.session_state:
    st.session_state.data_quality_scores = {}
if "cleaning_mode" not in st.session_state:
    st.session_state.cleaning_mode = None
if "processing_mode" not in st.session_state:
    st.session_state.processing_mode = None

# Dashboard wizard state variables
if "dashboard_step" not in st.session_state:
    st.session_state.dashboard_step = 1
if "selected_domain" not in st.session_state:
    st.session_state.selected_domain = None
if "top3" not in st.session_state:
    st.session_state.top3 = None
if "full_rank" not in st.session_state:
    st.session_state.full_rank = None
if "alternatives" not in st.session_state:
    st.session_state.alternatives = None
if "dummy" not in st.session_state:
    st.session_state.dummy = 0
if 'viz_recommendations' not in st.session_state:
    st.session_state.viz_recommendations = None
if 'selected_visualization' not in st.session_state:
    st.session_state.selected_visualization = None

def advance_step():
    # Clear recommendations when advancing to step 3
    if st.session_state.dashboard_step == 2:
        # Going from step 2 to step 3, clear any previous viz recommendations
        if 'viz_recommendations' in st.session_state:
            st.session_state.viz_recommendations = None
    
    # Increment the step
    st.session_state.dashboard_step += 1

def domain_step():
    st.title(t("Dashboard Configuration"))
    st.markdown(f"<h3 style='text-align: center;'>{t('Step 1: Choose Data Context')}</h3>", unsafe_allow_html=True)
    
    # First check if data exists in any of the possible session state variables
    has_data = False
    if "cleaned_dataframes" in st.session_state and st.session_state.cleaned_dataframes:
        has_data = True
    elif "dataframes" in st.session_state and st.session_state.dataframes:
        has_data = True
    elif "dashboard_uploaded_df" in st.session_state and st.session_state.dashboard_uploaded_df is not None:
        has_data = True
    
    # If no data is available, display warning and return to Home button
    if not has_data:
        st.error(t("No data available. Please upload a CSV file first."))
        
        # Center the button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(t("Go to Home Page"), use_container_width=True):
                st.switch_page("Home.py")
        return
        
    # If we have data, continue with domain selection
    st.markdown(f"<p style='text-align: center;'>{t('Select the domain that best matches your data to get more relevant visualization recommendations.')}</p>", unsafe_allow_html=True)
    
    # Add custom CSS to make the domain selectbox properly centered and contained
    st.markdown("""
        <style>
        /* Domain selection dropdown styles */
        .domain-selection-container {
            max-width: 100%;
            margin: 0 auto;
        }
        
        .domain-selection-container [data-testid="stSelectbox"] {
            max-width: 100%;
            width: 100%;
        }
        
        /* Make select element contained */
        .domain-selection-container [data-testid="stSelectbox"] > div > div {
            max-width: 100%;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
        }
        
        /* Center the label */
        .domain-selection-container [data-testid="stSelectbox"] label {
            text-align: center;
            width: 100%;
            display: block;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Use columns for better layout - larger middle column
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Wrap in a container for styles
        st.markdown('<div class="domain-selection-container">', unsafe_allow_html=True)
    
        domain_options = [t("Select a domain")] + list(domain_keywords.keys())
        selected_domain = st.selectbox(
            t("Select dataset domain"), 
            domain_options, 
            index=0 if st.session_state.selected_domain is None else domain_options.index(st.session_state.selected_domain)
        )
    
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Only show next button when a domain is selected
    if selected_domain != t("Select a domain"):
        st.session_state.selected_domain = selected_domain
        
        # Center the button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(t("Next"), use_container_width=True, type="primary"):
                # Force immediate step advance without requiring a second click
                st.session_state.dashboard_step = 2
                st.rerun()

def metrics_recommendation_step():
    # This should only run when we're on step 2, so let's validate that first
    if st.session_state.dashboard_step != 2:
        return
    
    # Check for data in various possible session state variables
    if "cleaned_dataframes" in st.session_state and st.session_state.cleaned_dataframes:
        # Use the first cleaned dataframe
        df_key = list(st.session_state.cleaned_dataframes.keys())[0]
        df = st.session_state.cleaned_dataframes[df_key]
        st.success(f"{t('Using cleaned data from previous steps')}: {df_key}")
    elif "dataframes" in st.session_state and st.session_state.dataframes:
        # If no cleaned data, try using the raw dataframe
        df_key = list(st.session_state.dataframes.keys())[0]
        df = st.session_state.dataframes[df_key]
        st.info(f"{t('Using raw data from previous steps')}: {df_key}")
    elif "dashboard_uploaded_df" in st.session_state:
        df = st.session_state.dashboard_uploaded_df
        st.info(t("Using previously uploaded file"))
    else:
        # If no data is available, instruct to return to Home
        st.error(t("No data available. Please go back to process data first."))
        if st.button(t("Back to Home"), use_container_width=True):
            st.switch_page("Home.py")
        return
    
    # Clean column names to avoid KeyError
    df.columns = [clean_column_name(col) for col in df.columns]
    
    # Initialize rankings if not already done
    if st.session_state.full_rank is None or st.session_state.top3 is None or st.session_state.alternatives is None:
        top3_candidates, full_ranking = rank_columns(df, st.session_state.selected_domain)
        st.session_state.full_rank = full_ranking
        st.session_state.top3 = top3_candidates
        st.session_state.alternatives = full_ranking[3:8]
    
    # Show Top Metrics in a horizontal table with just column names and aggregation types
    st.markdown(t("### 🏆 Recommended Metrics"))
    
    # Create a horizontal table with HTML
    metrics_html = """
    <table class="metrics-table">
        <thead>
            <tr>
                <th>Metric 1</th>
                <th>Metric 2</th>
                <th>Metric 3</th>
            </tr>
        </thead>
        <tbody>
            <tr>
    """
    
    # Add metric names
    for item in st.session_state.top3:
        metrics_html += f"<td>{item['Column']}</td>"
    
    metrics_html += """
            </tr>
            <tr>
    """
    
    # Add aggregation types with styling
    for item in st.session_state.top3:
        agg_type = item["Suggested_Aggregation"]
        agg_class = f"{agg_type}-bg"
        metrics_html += f'<td><span class="{agg_class}">{agg_type}</span></td>'
    
    metrics_html += """
            </tr>
        </tbody>
    </table>
    """
    
    # Display the metrics table
    st.write(metrics_html, unsafe_allow_html=True)
    
    # Show Alternatives section in an expander
    with st.expander(t("🔄 Browse Alternatives"), expanded=False):
        st.markdown('<div class="alternatives-section">', unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; margin-bottom:15px;'>{t('Select a metric to replace and browse alternatives.')}</div>", unsafe_allow_html=True)
        
        # Select which top metric to replace
        selected_metric = st.selectbox(
            t("Select metric to replace:"), 
            st.session_state.top3, 
            format_func=lambda x: x['Column'],
            key="selected_metric_to_replace"
        )
        
        # Get the position of selected metric
        pos_idx = st.session_state.top3.index(selected_metric)
        
        # Display current metric
        st.markdown(f"""
        <div style='text-align:center; margin:15px 0;'>
            <span style='font-weight:bold;'>{t('Current Metric')}:</span> 
            <span>{selected_metric['Column']}</span> 
            (<span class='{selected_metric['Suggested_Aggregation']}-bg'>{selected_metric['Suggested_Aggregation']}</span>)
        </div>
        """, unsafe_allow_html=True)
        
        # Add custom CSS to make the alternatives section wider
        st.markdown("""
            <style>
            /* Make alternatives section wider */
            .alternative-display {
                width: 100%;
                max-width: 900px;
            }
            
            /* Make the entire expander wider */
            [data-testid="stExpander"] {
                max-width: 900px !important;
                width: 100% !important;
                margin: 0 auto !important;
            }
            
            /* Make the dropdown wider but keep it centered - updated to be more specific */
            .alternatives-section .stSelectbox {
                min-width: 300px !important;
                max-width: 900px !important;
                width: 100% !important;
                margin: 0 auto !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Initialize navigation counter if not in session state
        if "nav_position" not in st.session_state:
            st.session_state.nav_position = {"metrics_alternatives": 0}
            
        # Get position for this section
        alt_pos = st.session_state.nav_position.get("metrics_alternatives", 0)
        
        # Show the alternatives
        if st.session_state.alternatives and len(st.session_state.alternatives) > 0:
            total_alternatives = len(st.session_state.alternatives)
            
            # Ensure position doesn't go out of bounds
            alt_pos = max(0, min(alt_pos, total_alternatives - 1))
            
            alternative_item = st.session_state.alternatives[alt_pos]
            
            st.markdown(f"<div style='text-align:center; margin-bottom:10px;'>{t('Alternative')} {alt_pos+1} {t('of')} {total_alternatives}</div>", unsafe_allow_html=True)
            
            # Add some space
            st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
            
            # Create navigation controls with columns
            nav_cols = st.columns([1, 1, 2, 1, 1])
            
            # Previous button
            prev_disabled = alt_pos <= 0
            with nav_cols[0]:
                if st.button("◀", key="prev_alt_metrics", disabled=prev_disabled, use_container_width=True):
                    st.session_state.nav_position["metrics_alternatives"] = alt_pos - 1
                    st.rerun()
            
            # Position indicator
            with nav_cols[2]:
                st.markdown(f"<div style='text-align: center; font-weight: bold;'>{alt_pos+1}/{total_alternatives}</div>", unsafe_allow_html=True)
            
            # Next button
            next_disabled = alt_pos >= total_alternatives - 1
            with nav_cols[4]:
                if st.button("▶", key="next_alt_metrics", disabled=next_disabled, use_container_width=True):
                    st.session_state.nav_position["metrics_alternatives"] = alt_pos + 1
                    st.rerun()
            
            st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
            
            # Display alternative
            st.markdown(f"""
            <div style='text-align:center; padding:15px; background-color:#f8f9fa; border-radius:5px;'>
                <div style='font-size:18px; font-weight:bold; margin-bottom:8px;'>{alternative_item['Column']}</div>
                <div style='margin-bottom:5px;'>
                    <span class="{alternative_item['Suggested_Aggregation']}-bg">{alternative_item['Suggested_Aggregation']}</span>
                </div>
                <div style='font-size:14px; color:#666; margin-top:10px;'>{t('Score')}: {alternative_item.get('Score', alternative_item.get('Final_Score', 0)):.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add button to use this alternative
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                if st.button(t("Use This Alternative"), key=f"use_metrics_{pos_idx}_{alt_pos}", use_container_width=True, type="primary"):
                    # Replace the selected metric with this alternative
                    # First check if alternative_item is already a dict or needs conversion
                    if hasattr(alternative_item, 'to_dict'):
                        st.session_state.top3[pos_idx] = alternative_item.to_dict()
                    else:
                        # It's already a dict
                        st.session_state.top3[pos_idx] = alternative_item
                    st.success(t("Metric replaced successfully!"))
                    st.rerun()
        else:
            st.info(t("No alternatives available for this metric."))
            
        st.markdown('</div>', unsafe_allow_html=True) # Close alternatives-section
    
    # Add a next button to advance to step 3
    if st.button(t("Next"), use_container_width=True, type="primary"):
        advance_step()
        st.rerun()  # Add explicit rerun to ensure step advances immediately

def extract_columns_from_name(row, df):
    """
    Extract and validate column names from a recommendation row.
    
    Parameters:
    -----------
    row : pandas.Series
        The recommendation row with Name and Type
    df : pandas.DataFrame
        The dataframe to validate column existence
        
    Returns:
    --------
    list
        List of valid column names
    """
    # Check if Name is present and is a string
    if 'Name' not in row:
        print(f"WARNING: 'Name' not found in row: {row}")
        return []
        
    name = row['Name']
    
    # Handle non-string names
    if not isinstance(name, str):
        print(f"WARNING: 'Name' is not a string, type is {type(name)}")
        if isinstance(name, (int, float)):
            # For numeric values, check if any column name matches this value
            matching_cols = [col for col in df.columns if str(name) == col]
            if matching_cols:
                return matching_cols[:1]
            # Otherwise return empty list
            return []
        # Try converting to string if possible
        try:
            name = str(name)
        except:
            return []
    
    row_type = row['Type']
    columns = []
    
    print(f"DEBUG - Extracting columns for {row_type}: {name}")
    
    # Extract columns based on row type
    if row_type == 'Column':
        # Single column - just return it if it exists
        if name in df.columns:
            return [name]
        return []
    
    elif row_type == 'Pair':
        # Try common separators for pairs
        for sep in [' & ', ', ', ' vs ', ' by ', ' and ', ' with ']:
            if sep in name:
                columns = [col.strip() for col in name.split(sep)]
                if len(columns) >= 2:
                    print(f"DEBUG - Found pair columns using separator '{sep}': {columns}")
                    break
        
        # If still no columns and looks like a tuple, try that format
        if len(columns) < 2 and name.startswith('(') and name.endswith(')'):
            try:
                # Remove parentheses and split by comma
                cols_str = name[1:-1]
                extracted = []
                for col in cols_str.split(','):
                    col = col.strip()
                    # Remove quotes if present
                    if (col.startswith("'") and col.endswith("'")) or (col.startswith('"') and col.endswith('"')):
                        col = col[1:-1]
                    extracted.append(col)
                if len(extracted) >= 2:
                    columns = extracted
                    print(f"DEBUG - Found pair columns from tuple: {columns}")
            except:
                # Fallback if tuple parsing fails
                pass
                
        # Last resort - try a simple split on ampersand
        if len(columns) < 2:
            columns = name.split(' & ')
            
        print(f"DEBUG - Extracted pair columns: {columns}")
    
    elif row_type == 'Triple':
        # Try common separators for triples
        for sep in [' & ', ', ']:
            if sep in name:
                columns = [col.strip() for col in name.split(sep)]
                if len(columns) >= 3:
                    break
                    
        # If that didn't work, try more complex formats like "X vs Y by Z"
        if len(columns) < 3:
            if ' vs ' in name and ' by ' in name:
                parts = name.split(' vs ')
                col1 = parts[0].strip()
                parts2 = parts[1].split(' by ')
                col2 = parts2[0].strip()
                col3 = parts2[1].strip()
                columns = [col1, col2, col3]
                
        # Last resort - split on ampersand
        if len(columns) < 3:
            columns = name.split(' & ')
    
    elif row_type == 'GroupBy':
        # Try extracting GroupBy columns with different formats
        if ' [by] ' in name:
            columns = name.split(' [by] ', 1)
        elif ' grouped by ' in name:
            columns = name.split(' grouped by ', 1)
        elif ' by ' in name:
            columns = name.split(' by ', 1)
        else:
            columns = [name]
    
    # Validate that columns exist in the dataframe
    valid_columns = [col for col in columns if col in df.columns]
    
    # If we don't have enough columns for the type, try fallbacks
    if row_type == 'Pair' and len(valid_columns) < 2 and len(df.columns) >= 2:
        # For pairs, we need at least 2 columns
        if len(valid_columns) == 1:
            # We have one valid column, find another one
            other_cols = [col for col in df.columns if col != valid_columns[0]]
            if other_cols:
                valid_columns.append(other_cols[0])
                print(f"DEBUG - Added complementary numeric column: {other_cols[0]}")
            else:
                # If no numeric column, use any other column
                other_col = next((col for col in df.columns if col != columns[0]), None)
                if other_col:
                    columns.append(other_col)
                    print(f"DEBUG - Added any complementary column: {other_col}")
        elif len(columns) == 0:
            # No valid column, use first two numeric columns as fallback
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            if len(numeric_cols) >= 2:
                columns = numeric_cols[:2]
                print(f"DEBUG - Using first two numeric columns: {columns}")
            else:
                columns = list(df.columns[:2])
                print(f"DEBUG - Using first two columns: {columns}")
            
        print(f"DEBUG - Final columns for pair visualization: {columns}")
    
    # Check if all specified columns exist in the dataframe
    existing_columns = [col for col in columns if col in df.columns]
    if not existing_columns:
        print(f"WARNING: None of the columns {columns} exist in the dataframe")
        # Use first columns as fallback
        if len(df.columns) > 0:
            existing_columns = [df.columns[0]]
            if row_type == 'Pair' and len(df.columns) > 1:
                existing_columns.append(df.columns[1])
            print(f"DEBUG - Using fallback columns: {existing_columns}")
    
    # Final log of columns to be used
    print(f"DEBUG - Final columns for visualization: {existing_columns}")
    
    return existing_columns

def visualization_recommendation_step():
    """Generate and display visualization recommendations based on the data."""
    
    # This should only run when we're on step 3, so let's validate that first
    if st.session_state.dashboard_step != 3:
        return
    
    # Get the dataframe from session state (handle different possible storage locations)
    if "dataframes" in st.session_state and st.session_state.dataframes:
        df_key = list(st.session_state.dataframes.keys())[0]
        df = st.session_state.dataframes[df_key]
    elif "dashboard_uploaded_df" in st.session_state:
        df = st.session_state.dashboard_uploaded_df
    else:
        st.error("No data found. Please return to previous steps.")
        return

    # Display centered header
    st.markdown("""
    <div class="viz-recommendations-container">
      <h3>🏆 Recommended Visualisations</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate the file ID for tracking changes
    current_file_id = None
    if "dataframes" in st.session_state and st.session_state.dataframes:
        df_key = list(st.session_state.dataframes.keys())[0]
        current_file_id = hash(df_key + str(df.shape) + str(list(df.columns)))
    elif "dashboard_uploaded_df" in st.session_state:
        current_file_id = hash("dashboard_df" + str(df.shape) + str(list(df.columns)))
    
    # Initialize file_id in session state if it doesn't exist
    if 'file_id' not in st.session_state:
        st.session_state.file_id = None
    
    # Check if we need to recalculate scores (new file upload)
    new_file_upload = st.session_state.file_id != current_file_id
    
    # Calculate scores only if this is a new file or first run
    if new_file_upload or 'all_scores' not in st.session_state:
        # Update file ID
        st.session_state.file_id = current_file_id
        
        # Calculate scores for all columns, pairs, and triples
        with st.spinner("Analyzing data and calculating visualization scores..."):
            column_scores, pair_scores, triple_scores, groupby_scores = score_all_columns_and_pairs(df)
        
        # Prepare data for display
        column_data = []
        for column, scores in column_scores.items():
            column_data.append({
                'Name': column,
                'Type': 'Column',
                'Distribution': round(scores['distribution_score'], 2),
                'Type Match': round(scores['data_type_score'], 2),
                'Quality': round(scores['data_quality_score'], 2),
                'Predictive': round(scores['predictive_power_score'], 2),
                'Semantic': round(scores['semantic_content_score'], 2),
                'Dimensional': round(scores['dimensional_analysis_score'], 2),
                'Variance': round(scores['variance_info_ratio_score'], 2),
                'Total Score': round(scores['total_score'], 2)
            })
        
        # Column pairs
        pair_data = []
        for pair, scores in pair_scores.items():
            pair_data.append({
                'Name': pair,
                'Type': 'Pair',
                'Association': round(scores['statistical_association'], 2),
                'Complexity': round(scores['visualization_complexity'], 2),
                'Pattern': round(scores['pattern_detection'], 2),
                'Anomaly': round(scores['anomaly_highlighting'], 2),
                'Complementarity': round(scores['information_complementarity'], 2),
                'Redundancy': round(scores['redundancy_penalization'], 2),
                'Utility': round(scores['practical_utility_score'], 2),
                'Total Score': round(scores['total_score'], 2)
            })
        
        # Column triples
        triple_data = []
        for triple, scores in triple_scores.items():
            triple_data.append({
                'Name': triple,
                'Type': 'Triple',
                'Dimensional Balance': round(scores['dimensional_balance'], 2),
                'Information Density': round(scores['information_density'], 2),
                'Visualization Feasibility': round(scores['visualization_feasibility'], 2),
                'Insight Potential': round(scores['insight_potential'], 2),
                'Interaction Synergy': round(scores['interaction_synergy'], 2),
                'Complexity Penalty': round(scores['complexity_penalty'], 2),
                'Total Score': round(scores['total_score'], 2)
            })
        
        # GroupBy pairs
        groupby_data = []
        for pair, scores in groupby_scores.items():
            # For app.py compatibility, format the name with [by] separator
            try:
                groupby_col, agg_col = pair
                pair_name = f"{groupby_col} [by] {agg_col}"
            except:
                # Handle the case where the pair might be formatted differently
                pair_name = str(pair)
            
            groupby_data.append({
                'Name': pair_name,
                'Type': 'GroupBy',
                'Group Differentiation': round(scores['group_differentiation'], 2),
                'Aggregation Meaningfulness': round(scores['aggregation_meaningfulness'], 2),
                'Group Size Balance': round(scores['group_size_balance'], 2),
                'Outlier Robustness': round(scores['outlier_robustness'], 2),
                'Visualization Potential': round(scores['visualization_potential'], 2),
                'Total Score': round(scores['total_score'], 2)
            })
        
        # Combine all scores
        all_scores = pd.DataFrame(column_data + pair_data + triple_data + groupby_data)
        all_scores = all_scores.sort_values('Total Score', ascending=False)
        
        # Create a recommended visualization type column
        all_scores['Recommended Visualization'] = all_scores.apply(
            lambda row: get_visualization_recommendation(row, df, 
                                                       {**column_scores, **pair_scores, **triple_scores}),
            axis=1
        )
        
        # Update visualization recommendations for GroupBy pairs
        for idx, row in all_scores[all_scores['Type'] == 'GroupBy'].iterrows():
            try:
                parts = row['Name'].split(" [by] ")
                if len(parts) == 2:
                    groupby_col, agg_col = parts
                    all_scores.at[idx, 'Recommended Visualization'] = get_groupby_visualization_recommendation(
                        df, groupby_col, agg_col, groupby_scores[(groupby_col, agg_col)]
                    )
            except:
                # Handle cases where the format is different or key error
                continue
        
        # Store all results in session state
        st.session_state.all_scores = all_scores
        st.session_state.column_scores = column_scores
        st.session_state.pair_scores = pair_scores
        st.session_state.triple_scores = triple_scores
        st.session_state.groupby_scores = groupby_scores
    else:
        # Use cached scores from session state
        all_scores = st.session_state.all_scores
        column_scores = st.session_state.column_scores
        pair_scores = st.session_state.pair_scores
        triple_scores = st.session_state.triple_scores
        groupby_scores = st.session_state.groupby_scores
    
    # Only create initial recommendations once when a new file is loaded
    if new_file_upload or 'top_recommendations' not in st.session_state:
        # Get the top items of each type separately
        top_columns = all_scores[all_scores['Type'] == 'Column'].head(2)
        top_pairs = all_scores[all_scores['Type'] == 'Pair'].head(2)
        top_triples = all_scores[all_scores['Type'] == 'Triple'].head(1)  # Limit to 1
        top_groupby = all_scores[all_scores['Type'] == 'GroupBy'].head(1)  # Limit to 1
        
        # Combine and re-sort to get the top 5 overall
        candidates = pd.concat([top_columns, top_pairs, top_triples, top_groupby])
        candidates = candidates.sort_values('Total Score', ascending=False)
        
        # Now create a more balanced selection
        winners = []
        
        # Check if we have each type available
        has_columns = not top_columns.empty
        has_pairs = not top_pairs.empty
        has_triples = not top_triples.empty
        has_groupby = not top_groupby.empty
        
        # Always include highest scoring column and pair if available
        if has_columns:
            best_column = candidates[candidates['Type'] == 'Column'].iloc[0]
            winners.append(best_column)
            
        if has_pairs:
            best_pair = candidates[candidates['Type'] == 'Pair'].iloc[0]
            winners.append(best_pair)
            
        # Include one triple if available
        if has_triples:
            best_triple = candidates[candidates['Type'] == 'Triple'].iloc[0]
            winners.append(best_triple)
            
        # Include one groupby if available
        if has_groupby:
            best_groupby = candidates[candidates['Type'] == 'GroupBy'].iloc[0]
            winners.append(best_groupby)
            
        # Fill remaining slots with highest scored items not already included
        winners_names = [w.name for w in winners]
        remaining = candidates[~candidates.index.isin(winners_names)]
        
        # Add items until we reach 5 total or run out of candidates
        for _, row in remaining.iterrows():
            winners.append(row)
            if len(winners) >= 5:
                break
            
        # Convert to DataFrame
        top_recommendations = pd.DataFrame(winners)
        
        # Ensure we have exactly 5 visualizations
        if len(top_recommendations) < 5:
            print(f"WARNING: Only have {len(top_recommendations)} recommendations, need 5")
            # If we have fewer than 5, duplicate some to reach 5
            while len(top_recommendations) < 5:
                # Add the top scoring visualization again
                top_recommendations = pd.concat([top_recommendations, pd.DataFrame([top_recommendations.iloc[0]])])
                print(f"Added duplicate to reach {len(top_recommendations)} visualizations")
        
        # Ensure we have at most 5 visualizations
        if len(top_recommendations) > 5:
            print(f"WARNING: Have {len(top_recommendations)} recommendations, trimming to 5")
            top_recommendations = top_recommendations.iloc[:5]
        
        # Create recommendations with column information
        top_recommendations['columns'] = top_recommendations.apply(
            lambda row: extract_columns_from_name(row, df), axis=1
        )
        
        # Store in session state
        st.session_state.top_recommendations = top_recommendations.copy()
        st.session_state.current_recommendations = top_recommendations.copy()
        
        # Store only the top alternatives for each type (limit to top 5)
        max_alternatives = 5
        st.session_state.column_candidates = all_scores[all_scores['Type'] == 'Column'].head(max_alternatives+2).iloc[2:].copy() if len(all_scores[all_scores['Type'] == 'Column']) > 2 else pd.DataFrame()
        st.session_state.pair_candidates = all_scores[all_scores['Type'] == 'Pair'].head(max_alternatives+2).iloc[2:].copy() if len(all_scores[all_scores['Type'] == 'Pair']) > 2 else pd.DataFrame()
        st.session_state.triple_candidates = all_scores[all_scores['Type'] == 'Triple'].head(max_alternatives+1).iloc[1:].copy() if len(all_scores[all_scores['Type'] == 'Triple']) > 1 else pd.DataFrame()
        st.session_state.groupby_candidates = all_scores[all_scores['Type'] == 'GroupBy'].head(max_alternatives+1).iloc[1:].copy() if len(all_scores[all_scores['Type'] == 'GroupBy']) > 1 else pd.DataFrame()
    
    # Initialize a counter for retry clicks if it doesn't exist
    if 'retry_counter' not in st.session_state:
        st.session_state.retry_counter = 0
    
    # Use current recommendations from session state
    displayed_recommendations = st.session_state.current_recommendations
    
    # Display the table using Streamlit's native table functionality instead of HTML
    # Create a simplified table for display
    display_df = displayed_recommendations.copy().reset_index(drop=True)
    display_df.index = range(1, len(display_df) + 1)  # 1-based indexing for user
    
    # Create a DataFrame for display with desired columns
    table_data = []
    for idx, row in display_df.iterrows():
        type_val = row['Type']
        # Define type-specific styling similar to metrics step
        type_class = ""
        if type_val == 'Column':
            type_class = "mean-bg"  # Blue styling
        elif type_val == 'Pair':
            type_class = "sum-bg"   # Orange styling
        elif type_val == 'Triple':
            type_class = "count-bg" # Purple styling
        elif type_val == 'GroupBy':
            type_class = "count-bg" # Purple styling (can be changed)
            
        table_data.append({
            "#": idx,
            "Column(s)": row['Name'],
            "Recommended Visualization": row['Recommended Visualization'],
            "Type": f'<span class="{type_class}">{type_val}</span>'  # Apply styling to type
        })
    
    # Convert to DataFrame for display
    vis_table = pd.DataFrame(table_data)
    
    # Display the table using Streamlit with HTML formatting
    st.write(vis_table.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # Add a container for the Browse Alternatives section
    st.markdown("""
    <style>
    .browse-alternatives-container {
        max-width: 900px;
        margin: 20px auto;
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Make visualization alternatives wider */
    .alternative-display {
        width: 100%;
        max-width: 900px;
    }
    
    /* Make the entire expander wider */
    [data-testid="stExpander"] {
        max-width: 900px !important;
        width: 100% !important;
        margin: 0 auto !important;
    }
    
    /* Make the dropdown wider but keep it centered - updated to be more specific */
    .viz-alternatives-section .stSelectbox {
        min-width: 300px !important;
        max-width: 900px !important;
        width: 100% !important;
        margin: 0 auto !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Show Alternatives section in an expander, just like in metrics page
    with st.expander(t("🔄 Browse Alternatives"), expanded=False):
        st.markdown('<div class="viz-alternatives-section">', unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; margin-bottom:15px;'>{t('Select a recommendation to replace and browse alternatives.')}</div>", unsafe_allow_html=True)
        
        # Dropdown to select which recommendation to modify
        selected_idx = st.selectbox(
            t("Select recommendation to replace:"), 
            range(1, len(displayed_recommendations)+1),
            format_func=lambda x: f"#{x}: {displayed_recommendations.iloc[x-1]['Name']}",
            key="selected_recommendation_to_replace"
        )
            
        # Get the type of selected recommendation (using 0-based index)
        row_idx = selected_idx - 1
        row_type = displayed_recommendations.iloc[row_idx]['Type']
        
        # Create a unique key for this recommendation
        nav_key = f"{row_idx}_{row_type}"
        
        current_item = displayed_recommendations.iloc[row_idx]
        
        # Add type-specific styling
        type_color_map = {
            'Column': '#0066cc',
            'Pair': '#cc6600',
            'Triple': '#006600',
            'GroupBy': '#6600cc'
        }
        type_color = type_color_map.get(row_type, 'black')
            
        # Display current metric
        st.markdown(f"""
        <div style='text-align:center; margin:15px 0;'>
            <span style='font-weight:bold;'>{t('Current Metric')}:</span> 
            <span>{current_item['Name']}</span> 
            (<span style='color: {type_color}; font-weight: bold;'>{current_item['Type']}</span>)
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize navigation counter if not in session state
        if "nav_position" not in st.session_state:
            st.session_state.nav_position = {}
            
        # Initialize counter for this recommendation if needed
        if nav_key not in st.session_state.nav_position:
            st.session_state.nav_position[nav_key] = 0
        
        # Get the alternative position for this recommendation
        alt_pos = st.session_state.nav_position[nav_key]
        
        # Determine how many alternatives we have
        if row_type == 'Column':
            candidates = st.session_state.column_candidates
        elif row_type == 'Pair':
            candidates = st.session_state.pair_candidates
        elif row_type == 'Triple':
            candidates = st.session_state.triple_candidates
        elif row_type == 'GroupBy':
            candidates = st.session_state.groupby_candidates
        else:
            candidates = pd.DataFrame()
            
        total_alternatives = len(candidates)
        
        if total_alternatives > 0:
            # Show alternative count 
            st.markdown(f"<div style='text-align:center; margin-bottom:10px;'>Alternative {alt_pos+1} of {total_alternatives}</div>", unsafe_allow_html=True)
            
            # Use custom HTML for spacing
            st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
            
            # Use 5 columns with balanced width for better spacing - make the buttons further apart
            nav_cols = st.columns([1, 1, 2, 1, 1])
            
            # Previous button
            with nav_cols[1]:
                prev_disabled = alt_pos <= 0
                if st.button("◀", key=f"prev_alt_viz", disabled=prev_disabled, use_container_width=True):
                    if alt_pos > 0:
                        st.session_state.nav_position[nav_key] -= 1
                        st.rerun()
            
            # Center column for spacing
            with nav_cols[2]:
                st.markdown(f"<div style='text-align: center; font-weight: bold;'>{alt_pos+1}/{total_alternatives}</div>", unsafe_allow_html=True)
            
            # Next button
            with nav_cols[3]:
                next_disabled = alt_pos >= total_alternatives - 1
                if st.button("▶", key=f"next_alt_viz", disabled=next_disabled, use_container_width=True):
                    if alt_pos < total_alternatives - 1:
                        st.session_state.nav_position[nav_key] += 1
                        st.rerun()
            
            st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
            
            # Make sure alt_pos is within valid range
            alt_pos = min(alt_pos, total_alternatives - 1)
            alternative = candidates.iloc[alt_pos]
                
            # Show the alternative with styling (matching metrics page style)
            st.markdown(f"""
            <div class='alternative-display' style='margin: 15px auto; padding: 20px; background-color: #f0f7ff; border-radius: 8px; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.1);'>
                <span style='font-size: 16px; font-weight: 500;'>{alternative['Name']}</span><br>
                <span style='color: {type_color}; font-weight: 500; margin-top: 8px; display: inline-block;'>
                    {alternative['Type']}
                </span> | 
                <span style='font-style: italic;'>
                    {alternative['Recommended Visualization']}
                </span>
                <span style='font-style: italic; margin-left: 10px;'>
                    (Score: {alternative['Total Score']:.2f})
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Center the Use button (visualization preview removed)
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                if st.button("Use This Alternative", key=f"use_viz_{row_idx}_{alt_pos}", use_container_width=True, type="primary"):
                    # Debug before replacement
                    print(f"\n====== BEFORE ALTERNATIVE SELECTION DEBUG ======")
                    print(f"displayed_recommendations shape: {displayed_recommendations.shape}")
                    print(f"displayed_recommendations index: {displayed_recommendations.index.tolist()}")
                    print(f"Replacing row at index {row_idx}")
                    print(f"Current visualization: {displayed_recommendations.iloc[row_idx]['Name']} (Type: {displayed_recommendations.iloc[row_idx]['Type']})")
                    print(f"New visualization: {alternative['Name']} (Type: {alternative['Type']})")
                    print(f"======= END BEFORE DEBUG =======\n")
                    
                    # More robust replacement approach - create a new DataFrame
                    # Convert alternative to Series if it's a dictionary
                    if not hasattr(alternative, 'name'):
                        alternative = pd.Series(alternative)
                    
                    # Extract columns for the alternative
                    alternative_row = alternative.copy()
                    extracted_columns = extract_columns_from_name(alternative_row, df)
                    alternative_row['columns'] = extracted_columns
                    
                    # Replace the row in displayed_recommendations
                    temp_df = displayed_recommendations.copy()
                    temp_df.iloc[row_idx] = alternative_row
                    displayed_recommendations = temp_df
                    
                    # Print debug info for columns
                    print(f"DEBUG - Alternative selected: {alternative_row['Name']}")
                    print(f"DEBUG - Extracted columns: {extracted_columns}")
                    
                    # Debug after replacement
                    print(f"\n====== AFTER ALTERNATIVE SELECTION DEBUG ======")
                    print(f"displayed_recommendations shape: {displayed_recommendations.shape}")
                    print(f"displayed_recommendations index: {displayed_recommendations.index.tolist()}")
                    print(f"Row at index {row_idx} now contains: {displayed_recommendations.iloc[row_idx]['Name']} (Type: {displayed_recommendations.iloc[row_idx]['Type']})")
                    print(f"======= END AFTER DEBUG =======\n")
                    
                    # Update session_state.viz_recommendations with the new selection
                    st.session_state.viz_recommendations = displayed_recommendations.copy()
                    
                    # Update displayed_recommendations in the current function scope
                    st.session_state.current_recommendations = displayed_recommendations.copy()
                    
                    # Ensure we have at most 5 recommendations
                    if len(st.session_state.current_recommendations) > 5:
                        print(f"WARNING: Trimming current_recommendations from {len(st.session_state.current_recommendations)} to 5")
                        st.session_state.current_recommendations = st.session_state.current_recommendations.iloc[:5]
                        st.session_state.viz_recommendations = st.session_state.current_recommendations.copy()
                    
                    # Force a complete refresh of the display
                    # Add a timestamp to session state to force recomputation
                    st.session_state.last_alternative_update = time.time()
                    
                    # Don't remove from candidates so user can go back to previous choices
                    # Just add a success message and keep the current navigation position
                    st.success("Metric replaced successfully!")
                    
                    # Force a rerun to update the table
                    st.rerun()
        else:
            st.info(t("No alternatives available for this recommendation."))
    
    st.markdown('</div>', unsafe_allow_html=True) # Close viz-alternatives-section
    
    # Store recommendations for dashboard
    if 'viz_recommendations' not in st.session_state or st.session_state.viz_recommendations is None:
        st.session_state.viz_recommendations = displayed_recommendations
    
    # Add a reset button to revert to original recommendations
    if st.button("↻ Reset to Original Recommendations", use_container_width=True):
        # Reset to original top recommendations
        st.session_state.current_recommendations = st.session_state.top_recommendations.copy()
        st.session_state.viz_recommendations = st.session_state.top_recommendations.copy()
        st.session_state.last_alternative_update = time.time()
        st.rerun()
    
    # Add navigation buttons
    cols = st.columns([2, 2, 2])
    with cols[0]:
        if st.button("← Previous", use_container_width=True):
            st.session_state.dashboard_step = 2
            st.rerun()
    with cols[2]:
        if st.button("Approve & View Dashboard", use_container_width=True, type="primary"):
            # Debug before updating
            print(f"\n====== APPROVE BUTTON DEBUG ======")
            print(f"current_recommendations shape: {st.session_state.current_recommendations.shape}")
            print(f"current_recommendations index: {st.session_state.current_recommendations.index.tolist()}")
            if len(st.session_state.current_recommendations) > 0:
                print(f"First recommendation: {st.session_state.current_recommendations.iloc[0]['Name']}")
            print(f"======= END APPROVE DEBUG =======\n")
            
            # Update the final visualizations before advancing
            # Ensure we only have 5 visualizations
            if len(st.session_state.current_recommendations) > 5:
                print(f"WARNING: Trimming visualizations from {len(st.session_state.current_recommendations)} to 5")
                st.session_state.viz_recommendations = st.session_state.current_recommendations.iloc[:5]
            else:
                st.session_state.viz_recommendations = st.session_state.current_recommendations
                
            # Debug after updating
            print(f"\n====== APPROVE BUTTON AFTER DEBUG ======")
            print(f"viz_recommendations shape: {st.session_state.viz_recommendations.shape}")
            print(f"viz_recommendations index: {st.session_state.viz_recommendations.index.tolist()}")
            if len(st.session_state.viz_recommendations) > 0:
                print(f"First recommendation: {st.session_state.viz_recommendations.iloc[0]['Name']}")
            print(f"======= END APPROVE AFTER DEBUG =======\n")
            
            advance_step()
            st.rerun()
    
    # Initialize show_all_viz if not present
    if 'show_all_viz' not in st.session_state:
        st.session_state.show_all_viz = False

def dashboard_layout_step():
    """Show the final dashboard with metrics at the top and visualizations below."""
    # This should only run when we're on step 4
    if st.session_state.dashboard_step != 4:
        return
        
    if 'filtered_df' not in st.session_state or st.session_state.filtered_df is None:
        if "cleaned_dataframes" in st.session_state and st.session_state.cleaned_dataframes:
            df_key = list(st.session_state.cleaned_dataframes.keys())[0]
            df = st.session_state.cleaned_dataframes[df_key]
        elif "dataframes" in st.session_state and st.session_state.dataframes:
            df_key = list(st.session_state.dataframes.keys())[0]
            df = st.session_state.dataframes[df_key]
        elif "dashboard_uploaded_df" in st.session_state:
            df = st.session_state.dashboard_uploaded_df
        else:
            st.error("No data found. Please return to previous steps.")
            return
    else:
        df = st.session_state.filtered_df
    
    # Add enhanced dashboard CSS for a professional look
    st.markdown("""
    <style>
    /* Dashboard container styling */
    .dashboard-container {
        padding: 0;
        background-color: transparent;
        margin-bottom: 10px; /* Reduced from 20px */
    }
    
    /* Metrics panel styling - horizontal layout */
    .metrics-panel {
        display: flex;
        flex-direction: row;
        flex-wrap: nowrap;
        justify-content: space-between;
        gap: 10px; /* Reduced from 15px */
        background-color: transparent;
        padding: 0;
        margin-bottom: 10px; /* Reduced from 20px */
        width: 100%;
    }
    
    /* Section titles */
    .section-title {
        font-size: 18px;
        font-weight: 600;
        margin: 8px 0; /* Reduced from 10px 0 */
        padding-bottom: 5px; /* Reduced from 8px */
        border-bottom: 2px solid #f0f2f6;
    }
    
    /* Individual visualization container */
    .viz-container {
        background-color: transparent;
        border-radius: 0;
        padding: 0;
        box-shadow: none;
        margin-bottom: 0;
    }
    
    .viz-container:hover {
        transform: none;
        box-shadow: none;
    }
    
    /* Make plot titles centered */
    .js-plotly-plot .plotly .main-svg .infolayer .g-gtitle {
        text-anchor: middle !important;
    }
    
    /* Metric card styling */
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 12px 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
        flex: 1 1 0;
        min-width: 0;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .metric-card.mean-bg {
        border-left-color: #2196F3;
    }
    
    .metric-card.sum-bg {
        border-left-color: #FF9800;
    }
    
    .metric-card.count-bg {
        border-left-color: #9C27B0;
    }
    
    .metric-label {
        font-size: 14px;
        color: #666;
        margin-bottom: 5px;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 600;
        color: #333;
    }
    
    .metric-desc {
        font-size: 12px;
        color: #888;
        margin-top: 5px;
    }
    </style>
    
    <div class="dashboard-container">
    """, unsafe_allow_html=True)
    
    # Display metrics in a horizontal panel at the top with no title
    if 'top3' in st.session_state and st.session_state.top3:
        # Create columns for metrics - one column per metric
        metric_cols = st.columns(len(st.session_state.top3))
        
        for i, metric in enumerate(st.session_state.top3):
            column = metric['Column']
            agg_type = metric['Suggested_Aggregation']
            
            # Skip if column doesn't exist in dataframe
            if column not in df.columns:
                continue
                
            # Calculate metric value
            try:
                if agg_type == 'mean':
                    value = df[column].mean()
                    bg_class = "mean-bg"
                    description = f"Average value across {len(df)} records"
                elif agg_type == 'sum':
                    value = df[column].sum()
                    bg_class = "sum-bg"
                    description = f"Total sum across {len(df)} records"
                elif agg_type == 'count':
                    value = df[column].nunique()
                    bg_class = "count-bg"
                    description = f"Number of unique values"
                else:
                    continue
                    
                # Format number for display
                if isinstance(value, (int, float)):
                    if value >= 1000000:
                        formatted_value = f"{value/1000000:.1f}M"
                    elif value >= 1000:
                        formatted_value = f"{value/1000:.1f}K"
                    elif isinstance(value, float):
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                else:
                    formatted_value = str(value)
                
                # Display the metric in its column
                with metric_cols[i]:
                    st.markdown(f"""
                    <div class="metric-card {bg_class}">
                        <div class="metric-label">{agg_type.capitalize()} of {column}</div>
                        <div class="metric-value">{formatted_value}</div>
                        <div class="metric-desc">{description}</div>
                    </div>
                    """, unsafe_allow_html=True)
            except:
                # Skip metrics that can't be calculated
                pass
    
    # Add a divider line under metrics
    st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    
    # Display visualizations from recommendations - show all of them
    if 'viz_recommendations' in st.session_state and not st.session_state.viz_recommendations.empty:
        # More detailed debug for troubleshooting
        print(f"\n====== DETAILED DASHBOARD VIZ RECOMMENDATIONS DEBUG ======")
        print(f"DataFrame shape: {st.session_state.viz_recommendations.shape}")
        print(f"DataFrame index: {st.session_state.viz_recommendations.index.tolist()}")
        print(f"DataFrame columns: {st.session_state.viz_recommendations.columns.tolist()}")
        if len(st.session_state.viz_recommendations) > 0:
            print(f"First row Name: {st.session_state.viz_recommendations.iloc[0].get('Name')}")
            print(f"First row Type: {st.session_state.viz_recommendations.iloc[0].get('Type')}")
            print(f"First row Visualization: {st.session_state.viz_recommendations.iloc[0].get('Recommended Visualization')}")
        print(f"====== END DETAILED DEBUG ======\n")
        
        # DEBUG: Log the recommendations coming from step 3
        print(f"\n---------- BEGIN DASHBOARD VISUALIZATION RECOMMENDATIONS ----------")
        for idx, row in st.session_state.viz_recommendations.iterrows():
            rec_dict = row.to_dict()
            print(f"DEBUG - Recommendation #{idx+1}:")
            print(f"  Name: {rec_dict.get('Name')}")
            print(f"  Type: {rec_dict.get('Type')}")
            print(f"  Recommended Visualization: {rec_dict.get('Recommended Visualization')}")
            print(f"  Columns: {rec_dict.get('columns')}")
        print(f"---------- END DASHBOARD VISUALIZATION RECOMMENDATIONS ----------\n")
        
        # Create the dashboard layout with left and right sections
        left_col, right_col = st.columns([2, 1])
        
        # Convert visualization recommendations to a list of dictionaries
        viz_list = []
        # Ensure we only have 5 recommendations
        if len(st.session_state.viz_recommendations) > 5:
            print(f"WARNING: Trimming viz_recommendations in dashboard_layout_step from {len(st.session_state.viz_recommendations)} to 5")
            viz_recommendations = st.session_state.viz_recommendations.iloc[:5]
        else:
            viz_recommendations = st.session_state.viz_recommendations
            
        for idx, row in viz_recommendations.iterrows():
            viz_dict = row.to_dict()
            print(f"\nDEBUG - Processing recommendation #{idx+1}: {viz_dict.get('Name')} (Type: {viz_dict.get('Type')})")
            
            # Add 'vis_type' key with value from 'Recommended Visualization'
            if 'Recommended Visualization' in viz_dict and 'vis_type' not in viz_dict:
                viz_dict['vis_type'] = viz_dict['Recommended Visualization']
                print(f"DEBUG - Setting vis_type to: {viz_dict['vis_type']}")
            
            # Use columns from step 3 if available, preserve them exactly
            if 'columns' in viz_dict and viz_dict['columns'] is not None:
                # Log the current columns
                print(f"DEBUG - Starting with columns: {viz_dict['columns']} (type: {type(viz_dict['columns'])})")
                
                # Check if columns is iterable before trying to filter it
                if not isinstance(viz_dict['columns'], list):
                    print(f"WARNING: 'columns' is not a list in dashboard_layout_step: {type(viz_dict['columns'])}")
                    # Try to convert to a list if it's a string
                    if isinstance(viz_dict['columns'], str) and viz_dict['columns'] in df.columns:
                        viz_dict['columns'] = [viz_dict['columns']]
                        print(f"DEBUG - Converted string to column list: {viz_dict['columns']}")
                    elif isinstance(viz_dict['columns'], (int, float)):
                        # Try to convert numeric value to column name
                        col_name = str(viz_dict['columns'])
                        if col_name in df.columns:
                            viz_dict['columns'] = [col_name]
                            print(f"DEBUG - Converted numeric to column list: {viz_dict['columns']}")
                        else:
                            # Force regeneration of columns
                            previous_columns = viz_dict['columns']
                            viz_dict['columns'] = extract_columns_from_name(row, df)
                            print(f"DEBUG - Regenerated columns from {previous_columns} to {viz_dict['columns']}")
                    else:
                        # For other types, initialize with extracted columns
                        previous_columns = viz_dict['columns']
                        viz_dict['columns'] = extract_columns_from_name(row, df)
                        print(f"DEBUG - Regenerated columns from {previous_columns} to {viz_dict['columns']}")
                else:
                    # Filter to valid columns but preserve their original order
                    original_columns = viz_dict['columns'].copy()
                    viz_dict['columns'] = [col for col in viz_dict['columns'] if col in df.columns]
                    if len(viz_dict['columns']) != len(original_columns):
                        print(f"DEBUG - Filtered columns from {original_columns} to {viz_dict['columns']}")
                
                # If we don't have enough columns for the type, try regenerating them
                if viz_dict.get('Type') == 'Pair' and len(viz_dict.get('columns', [])) < 2:
                    # Regenerate columns using our helper function
                    print(f"DEBUG - Regenerating columns for pair {viz_dict.get('Name')}")
                    previous_columns = viz_dict.get('columns', [])
                    viz_dict['columns'] = extract_columns_from_name(row, df)
                    print(f"DEBUG - Regenerated columns from {previous_columns} to {viz_dict.get('columns')}")
            else:
                # Add columns if not present (use Name for column reference)
                if 'Name' in viz_dict:
                    # Extract the columns from the name using our helper function
                    viz_dict['columns'] = extract_columns_from_name(row, df)
                    print(f"DEBUG - Generated columns for {viz_dict.get('Type')}: {viz_dict.get('columns')}")
            
            # Ensure visualization type is appropriate for column count
            if viz_dict.get('Type') == 'Triple' and len(viz_dict.get('columns', [])) < 3:
                print(f"WARNING: Triple visualization does not have 3 columns: {viz_dict.get('columns')}")
                # If it's a triple but doesn't have 3 columns, try to make a smart adjustment
                if 'Scatter' in viz_dict['vis_type'] and len(viz_dict.get('columns', [])) == 2:
                    original_vis_type = viz_dict['vis_type']
                    # Scatter with only 2 columns - no color
                    viz_dict['vis_type'] = 'Scatter Plot (px.scatter)'
                    print(f"DEBUG - Changed visualization type from {original_vis_type} to {viz_dict['vis_type']}")
                elif '3D' in viz_dict['vis_type'] and len(viz_dict.get('columns', [])) == 2:
                    original_vis_type = viz_dict['vis_type']
                    # 3D with only 2 columns - fallback to 2D
                    viz_dict['vis_type'] = 'Scatter Plot (px.scatter)'
                    print(f"DEBUG - Changed visualization type from {original_vis_type} to {viz_dict['vis_type']}")
            
            # Special handling for pairs to ensure visualization type is preserved
            if viz_dict.get('Type') == 'Pair':
                # For pairs, ensure the visualization type is handled correctly
                # First ensure vis_type is a string before calling .lower()
                if not isinstance(viz_dict['vis_type'], str):
                    print(f"DEBUG - Converting non-string vis_type to string: {viz_dict['vis_type']} (type: {type(viz_dict['vis_type'])})")
                    viz_dict['vis_type'] = str(viz_dict['vis_type'])
                
                viz_type = viz_dict['vis_type'].lower()
                
                # Check if this is a Scatter Plot Matrix visualization
                if any(splom_term in viz_type for splom_term in ['scatter plot matrix', 'splom', 'scatter matrix']):
                    print(f"DEBUG - Preserving SPLOM visualization type for pair")
                    viz_dict['vis_type'] = 'Scatter Plot Matrix (Splom)'
                
                # Check if we have at least 2 columns
                if len(viz_dict.get('columns', [])) < 2:
                    print(f"WARNING: Pair still doesn't have 2 columns after extraction")
            
            # Print final recommendation before adding to viz_list
            print(f"DEBUG - Final viz configuration:")
            print(f"  Name: {viz_dict.get('Name')}")
            print(f"  Type: {viz_dict.get('Type')}")
            print(f"  Visualization type: {viz_dict.get('vis_type')}")
            print(f"  Columns: {viz_dict.get('columns')}")
            
            viz_list.append(viz_dict)
            
        # Ensure we only have 5 visualizations
        if len(viz_list) > 5:
            print(f"WARNING: Trimming viz_list from {len(viz_list)} to 5 items")
            viz_list = viz_list[:5]
    else:
        st.warning("No visualization recommendations available. Please complete step 3 first.")
    
    # Calculate heights for better proportions
    # Increase left plot heights to better utilize vertical space
    left_viz_height = 350  # Further increased to fill vertical space
    # Keep the right visualization height tall
    right_viz_height = 700  # Much taller to ensure it fills the column
    
    # Add minimal CSS to remove extra padding in viz containers
    st.markdown("""
    <style>
    /* Remove empty boxes over visualizations */
    .viz-container {
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    
    /* Force center plot titles */
    .js-plotly-plot .plotly .main-svg .infolayer .g-gtitle {
        font-size: 14px !important;
        margin-top: -5px !important;
        text-anchor: middle !important;
        dominant-baseline: middle !important;
    }
    
    /* Fix Plotly title alignment */
    .gtitle {
        text-anchor: middle !important;
        transform: none !important;
    }
    
    /* Force title to center of container */
    .svg-container {
        text-align: center !important;
    }
    
    /* Ensure plots are aligned vertically */
    .element-container {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Remove any container that might affect alignment */
    .st-emotion-cache-1r6slb0 {
        padding: 0 !important;
        margin: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Left column - top row with 2 visualizations side by side
    with left_col:
        top_cols = st.columns(2)
        for col_idx, rec_idx in enumerate([0, 1]):
            with top_cols[col_idx]:
                create_viz_container(df, viz_list[rec_idx], height=left_viz_height, chart_id=f"left_top_{col_idx}")
    
        # Left column - bottom row with 2 visualizations side by side
        bottom_cols = st.columns(2)
        for col_idx, rec_idx in enumerate([2, 3]):
            with bottom_cols[col_idx]:
                create_viz_container(df, viz_list[rec_idx], height=left_viz_height, chart_id=f"left_bottom_{col_idx}")
    
    # Right column - single visualization taking full height
    with right_col:
        create_viz_container(df, viz_list[4], height=right_viz_height, chart_id="right_main")
    
    # Close dashboard container
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add navigation buttons
    cols = st.columns([2, 2, 2])
    with cols[0]:
        if st.button("← Previous", use_container_width=True):
            st.session_state.dashboard_step = 3
            st.rerun()
    
    # Export button removed as requested

# Create visualization container based on recommendation
def create_viz_container(df, rec, height=300, chart_id=None):
    """Create a visualization container with title and visualization"""
    try:
        # More detailed debug for diagnosing visualization issues
        print(f"\n---------- BEGIN CREATE_VIZ_CONTAINER ----------")
        print(f"DEBUG - Creating visualization for recommendation: {rec.get('Name')}")
        print(f"DEBUG - Recommendation type: {rec.get('Type')}")
        print(f"DEBUG - Raw recommendation columns: {rec.get('columns')}")
        print(f"DEBUG - Visualization type: {rec.get('vis_type')}")
        
        viz_type = rec['vis_type']
        is_pair = rec.get('Type') == 'Pair'
        
        # Ensure viz_type is a string to avoid AttributeError
        if not isinstance(viz_type, str):
            print(f"DEBUG - Converting non-string viz_type to string: {viz_type} (type: {type(viz_type)})")
            viz_type = str(viz_type)
        
        # Special handling for scatter plot matrix and other specific visualizations for pairs
        if is_pair:
            # Ensure viz_type is a string before calling .lower()
            if not isinstance(viz_type, str):
                print(f"DEBUG - Converting non-string viz_type to string in splom check: {viz_type} (type: {type(viz_type)})")
                viz_type = str(viz_type)
                
            # Check if this is a scatter matrix visualization
            if any(splom_term in viz_type.lower() for splom_term in 
                  ['scatter plot matrix', 'splom', 'scatter matrix']):
                print(f"DEBUG - Forcing Scatter Plot Matrix for pair")
                viz_type = "Scatter Plot Matrix (Splom)"
        
        # If this is a Triple type recommendation, force Triple visualization
        if rec.get('Type') == 'Triple':
            viz_type = 'Triple Visualization'
            print(f"DEBUG - Forcing Triple Visualization for Type=Triple recommendation")
        
        # Get the original columns from the recommendation
        original_columns = rec.get('columns', [])
        if original_columns is None:
            original_columns = []
            print(f"DEBUG - Original columns is None, setting to empty list")
            
        # Ensure columns is a list
        if not isinstance(original_columns, list):
            print(f"WARNING: Original columns is not a list: {type(original_columns)}")
            # Try to convert to list
            if isinstance(original_columns, str) and original_columns in df.columns:
                original_columns = [original_columns]
                print(f"DEBUG - Converted string column to list: {original_columns}")
            else:
                # If conversion failed, extract columns from name
                original_columns = extract_columns_from_name(rec, df)
                print(f"DEBUG - Extracted columns from name: {original_columns}")
        
        # Verify columns exist in the dataframe
        columns = [col for col in original_columns if col in df.columns]
        if len(columns) != len(original_columns):
            print(f"WARNING: Some columns don't exist in dataframe. Original: {original_columns}, Valid: {columns}")
        
        # If we don't have any columns, extract from name
        if not columns and 'Name' in rec:
            print(f"DEBUG - No valid columns found, extracting from name")
            columns = extract_columns_from_name(rec, df)
            print(f"DEBUG - Extracted columns: {columns}")
        
        # For pairs, ensure we have at least 2 columns for visualization
        if is_pair and len(columns) < 2:
            print(f"WARNING: Not enough columns for pair visualization: {columns}")
            
            # Only run this logic if we have exactly 1 column
            if len(columns) == 1 and len(df.columns) > 1:
                first_col = columns[0]
                
                # Try to find appropriate column to pair with based on visualization type
                # Ensure viz_type is a string before checking for visualization terms
                if not isinstance(viz_type, str):
                    print(f"DEBUG - Converting non-string viz_type to string in term check: {viz_type} (type: {type(viz_type)})")
                    viz_type = str(viz_type)
                    
                if any(term in viz_type.lower() for term in ['scatter', 'plot', 'point', 'bubble', 'line']):
                    # For these vis types, prefer numeric columns
                    other_cols = [col for col in df.columns 
                                if col != first_col and pd.api.types.is_numeric_dtype(df[col])]
                    if other_cols:
                        columns.append(other_cols[0])
                        print(f"DEBUG - Added complementary numeric column: {other_cols[0]}")
                    else:
                        # If no numeric, try any column
                        other_col = next((col for col in df.columns if col != first_col), None)
                        if other_col:
                            columns.append(other_col)
                            print(f"DEBUG - Added any complementary column: {other_col}")
                elif any(term in viz_type.lower() for term in ['heatmap', 'density', 'contour']):
                    # For heatmaps, prefer categorical columns to pair with numeric
                    is_numeric = pd.api.types.is_numeric_dtype(df[first_col])
                    if is_numeric:
                        # Pair with categorical
                        other_cols = [col for col in df.columns 
                                     if col != first_col and not pd.api.types.is_numeric_dtype(df[col])]
                    else:
                        # Pair with numeric
                        other_cols = [col for col in df.columns 
                                     if col != first_col and pd.api.types.is_numeric_dtype(df[col])]
                    
                    if other_cols:
                        columns.append(other_cols[0])
                        print(f"DEBUG - Added complementary column for heatmap: {other_cols[0]}")
                    else:
                        # Fallback to any column
                        other_col = next((col for col in df.columns if col != first_col), None)
                        if other_col:
                            columns.append(other_col)
                            print(f"DEBUG - Added any complementary column: {other_col}")
            elif len(columns) == 0 and len(df.columns) >= 2:
                # No columns at all - use the first two appropriate columns
                # Ensure viz_type is a string before checking visualization type
                if not isinstance(viz_type, str):
                    print(f"DEBUG - Converting non-string viz_type to string in scatter/line check: {viz_type} (type: {type(viz_type)})")
                    viz_type = str(viz_type)
                    
                if "scatter" in viz_type.lower() or "line" in viz_type.lower():
                    # For scatter, prefer numeric columns
                    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                    if len(numeric_cols) >= 2:
                        columns = numeric_cols[:2]
                        print(f"DEBUG - Using first two numeric columns for scatter: {columns}")
                    else:
                        # Fallback to any columns
                        columns = list(df.columns[:2])
                        print(f"DEBUG - Using first two columns (fallback): {columns}")
                else:
                    # For other types, use any columns
                    columns = list(df.columns[:2])
                    print(f"DEBUG - Using first two columns for general pair: {columns}")
        
        # Final verification of columns
        if not columns:
            print(f"WARNING: No valid columns for visualization, using first column in dataframe")
            if len(df.columns) > 0:
                columns = [df.columns[0]]
        
        # Print final columns
        print(f"DEBUG - Final columns for visualization: {columns}")
        
        # Create a responsive container for the visualization
        container_id = f"viz_container_{chart_id}" if chart_id else f"viz_container_{randint(1000, 9999)}"
        
        # Create visualization
        viz = create_visualization(df, viz_type, columns, 
                                 is_pair=is_pair,
                                 rec=rec)  # Pass the full recommendation for context
        
        if viz is not None:
            # Check if this is the right side plot and use an increased height
            if 'right_main' in str(container_id):
                # For right plot, set a much taller height and adjust margins
                viz.update_layout(
                    height=height, 
                    margin=dict(t=40, b=30, l=30, r=30),  # Increased top margin
                    autosize=True,
                    title_font_size=16,  # Slightly larger than left plots but still compact
                    title_y=1.0,        # Moved title higher to position outside plot area
                    title={'text': viz.layout.title.text, 'x': 0.5},  # Force center alignment
                    title_xanchor='center'  # Explicitly set the anchor point
                )
            else:
                # For left plots, use tighter margins to maximize vertical space
                viz.update_layout(
                    height=height,
                    margin=dict(t=40, b=20, l=25, r=25),  # Increased top margin
                    autosize=True,
                    title_font_size=14,  # Smaller title font
                    title_y=1.0,        # Moved title higher to position outside plot area
                    title={'text': viz.layout.title.text, 'x': 0.5},  # Force center alignment
                    title_xanchor='center'  # Explicitly set the anchor point
                )
            # Use use_container_width to fill available space
            st.plotly_chart(viz, use_container_width=True, key=container_id)
            print(f"---------- END CREATE_VIZ_CONTAINER: SUCCESS ----------\n")
        else:
            st.error(t(f"Could not create {viz_type} with columns {', '.join(columns)}"))
            print(f"---------- END CREATE_VIZ_CONTAINER: FAILED (NULL VISUALIZATION) ----------\n")
            
        # Removed the closing div container
        return True
    except Exception as e:
        # More detailed error information
        import traceback
        print(f"ERROR in create_viz_container: {str(e)}")
        print(f"Recommendation object: {rec}")
        traceback.print_exc()
        st.error(t(f"Error displaying visualization: {str(e)}"))
        print(f"---------- END CREATE_VIZ_CONTAINER: ERROR ----------\n")
        return False

def create_visualization(df, viz_type, columns, is_pair=False, rec=None):
    """
    Create a plotly visualization based on visualization type and columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to visualize
    viz_type : str
        The visualization type to create
    columns : list
        List of column names to use for the visualization
    is_pair : bool, optional
        Whether this visualization is for a pair of columns
    rec : dict, optional
        The full recommendation dictionary with additional context
        
    Returns:
    --------
    plotly.graph_objects.Figure or None
        The created visualization, or None if creation failed
    """
    try:
        # Clear debug headers
        print(f"\n---------- BEGIN CREATE_VISUALIZATION ----------")
        print(f"DEBUG - Creating visualization: type='{viz_type}', is_pair={is_pair}")
        print(f"DEBUG - Original columns: {columns}")
        print(f"DEBUG - Recommendation type: {rec.get('Type') if rec else 'None'}")
        
        # Store original viz_type for logging purposes
        original_viz_type = viz_type
        
        # Ensure columns is a list
        if not isinstance(columns, list):
            print(f"WARNING: columns parameter is not a list: {type(columns)}")
            if isinstance(columns, str) and columns in df.columns:
                # Convert string to a single-item list
                columns = [columns]
                print(f"DEBUG - Converted string column to list: {columns}")
            else:
                # For any other type, initialize as empty list
                columns = []
                print(f"DEBUG - Created empty columns list")
        
        # Make a shallow copy of columns to avoid modifying the original
        columns = columns.copy()
        
        # Ensure all columns exist in the dataframe
        original_columns = columns.copy()
        columns = [col for col in columns if col in df.columns]
        if len(columns) != len(original_columns):
            missing = set(original_columns) - set(columns)
            print(f"WARNING: Some columns don't exist in dataframe: {missing}")
            
        # Return early if we have no valid columns
        if not columns and len(df.columns) > 0:
            print(f"WARNING: No valid columns for visualization. Using fallback.")
            # For simple visualizations, try to use the first column(s)
            if is_pair and len(df.columns) >= 2:
                columns = list(df.columns[:2])
                print(f"DEBUG - Using first two columns as fallback: {columns}")
            else:
                columns = [df.columns[0]]
                print(f"DEBUG - Using first column as fallback: {columns[0]}")
        
        # Add comprehensive mapping for all visualization types
        exact_viz_map = {
            # Basic visualizations
            "Scatter Plot Matrix (Splom)": "scatter_matrix",
            "Scatter Plot Matrix": "scatter_matrix",
            "Splom": "scatter_matrix",
            "Scatter Matrix": "scatter_matrix",
            "Scatter Plot (px.scatter)": "scatter",
            "Scatter Plot": "scatter",
            "Scatter Plot with Colors (px.scatter)": "scatter_color",
            "Scatter Plot with Color (px.scatter)": "scatter_color",
            
            # Bar charts
            "Bar Chart (px.bar)": "bar",
            "Bar Chart": "bar",
            "Grouped Bar Chart (px.bar)": "bar_grouped",
            "Grouped Bar Chart": "bar_grouped",
            "Stacked Bar Chart (px.bar)": "bar_stacked",
            "Stacked Bar Chart": "bar_stacked",
            
            # Pie and related
            "Pie Chart (px.pie)": "pie",
            "Pie Chart": "pie",
            "Sunburst Chart (px.sunburst)": "sunburst",
            "Sunburst (px.sunburst)": "sunburst",
            "Sunburst Chart": "sunburst",
            "Treemap (px.treemap)": "treemap",
            "Treemap": "treemap",
            
            # Line and area charts
            "Line Chart (px.line)": "line",
            "Line Chart": "line",
            "Line Plot (px.line)": "line",
            "Multi-line Chart (px.line)": "line_multi",
            "Area Chart (px.area)": "area",
            "Area Chart": "area",
            "Stacked Area Chart (px.area)": "area_stacked",
            
            # Box and distribution plots
            "Box Plot (px.box)": "box",
            "Box Plot": "box",
            "Strip Plot (px.box with points)": "box_strip",
            "Violin Plot (px.violin)": "violin",
            "Violin Plot": "violin",
            "Histogram (px.histogram)": "histogram",
            "Histogram": "histogram",
            
            # Heatmaps
            "Heatmap (px.imshow)": "heatmap",
            "Heatmap (px.heatmap)": "heatmap",
            "Heatmap": "heatmap",
            "Density Heatmap (px.density_heatmap)": "density_heatmap",
            "Calendar Heatmap (px.heatmap)": "calendar_heatmap",
            "Calendar Heatmap": "calendar_heatmap",
            
            # Maps
            "Choropleth Map (px.choropleth)": "choropleth",
            "Choropleth Map": "choropleth",
            
            # Advanced plots
            "Parallel Categories (px.parallel_categories)": "parallel_categories",
            "Parallel Categories": "parallel_categories",
            "Candlestick Chart (px.candlestick)": "candlestick",
            "Candlestick Chart": "candlestick",
            "Table (go.Table)": "table",
            "Timeline (px.scatter)": "timeline",
            "Timeline": "timeline",
            
            # 3D and animations
            "3D Scatter Plot (px.scatter3d)": "scatter3d",
            "3D Scatter Plot": "scatter3d",
            "Animated Scatter (px.scatter with frames)": "scatter_animated",
            "Animated Scatter": "scatter_animated",
            "Faceted Scatter Plots (px.subplots)": "scatter_faceted",
            
            # Special types
            "Triple Visualization": "triple"
        }
        
        # Check exact match first
        if viz_type in exact_viz_map:
            simple_viz_type = exact_viz_map[viz_type]
            print(f"DEBUG - Using exact mapping for '{viz_type}' -> '{simple_viz_type}'")
        else:
            # Try to extract from representation of visualization types like "Visualization Type (px.function)"
            # Ensure viz_type is a string before calling .lower()
            if not isinstance(viz_type, str):
                print(f"WARNING: viz_type is not a string when extracting type: {viz_type} (type: {type(viz_type)})")
                viz_type = str(viz_type)
                
            simple_viz_type = viz_type.lower()
            if '(' in simple_viz_type and ')' in simple_viz_type:
                # Extract the code between parentheses
                extracted_type = simple_viz_type.split('(')[1].split(')')[0]
                # If it starts with px. or go., remove that prefix
                if extracted_type.startswith('px.') or extracted_type.startswith('go.'):
                    simple_viz_type = extracted_type.split('.')[1]
                else:
                    simple_viz_type = extracted_type
            print(f"DEBUG - Extracted viz type: '{simple_viz_type}' from '{viz_type}'")
            
            # Check for common types in the original string if extraction didn't yield a useful result
            if simple_viz_type == viz_type.lower():
                viz_lower = viz_type.lower()
                if 'scatter' in viz_lower:
                    if 'matrix' in viz_lower or 'splom' in viz_lower:
                        simple_viz_type = 'scatter_matrix'
                    elif '3d' in viz_lower:
                        simple_viz_type = 'scatter3d'
                    elif 'animate' in viz_lower or 'frame' in viz_lower:
                        simple_viz_type = 'scatter_animated'
                    elif 'facet' in viz_lower:
                        simple_viz_type = 'scatter_faceted'
                    elif 'color' in viz_lower:
                        simple_viz_type = 'scatter_color'
                    else:
                        simple_viz_type = 'scatter'
                elif 'bar' in viz_lower:
                    if 'grouped' in viz_lower:
                        simple_viz_type = 'bar_grouped'
                    elif 'stacked' in viz_lower:
                        simple_viz_type = 'bar_stacked'
                    else:
                        simple_viz_type = 'bar'
                elif 'line' in viz_lower:
                    if 'multi' in viz_lower:
                        simple_viz_type = 'line_multi'
                    else:
                        simple_viz_type = 'line'
                elif 'area' in viz_lower:
                    if 'stacked' in viz_lower:
                        simple_viz_type = 'area_stacked'
                    else:
                        simple_viz_type = 'area'
                elif 'heatmap' in viz_lower:
                    if 'calendar' in viz_lower:
                        simple_viz_type = 'calendar_heatmap'
                    elif 'density' in viz_lower:
                        simple_viz_type = 'density_heatmap'
                    else:
                        simple_viz_type = 'heatmap'
                elif 'pie' in viz_lower:
                    simple_viz_type = 'pie'
                elif 'box' in viz_lower:
                    if 'strip' in viz_lower or 'point' in viz_lower:
                        simple_viz_type = 'box_strip'
                    else:
                        simple_viz_type = 'box'
                elif 'histogram' in viz_lower or 'hist' in viz_lower:
                    simple_viz_type = 'histogram'
                elif 'violin' in viz_lower:
                    simple_viz_type = 'violin'
                elif 'tree' in viz_lower or 'treemap' in viz_lower:
                    simple_viz_type = 'treemap'
                elif 'sunburst' in viz_lower:
                    simple_viz_type = 'sunburst'
                elif 'table' in viz_lower:
                    simple_viz_type = 'table'
                elif 'parallel' in viz_lower or 'categor' in viz_lower:
                    simple_viz_type = 'parallel_categories'
                elif 'choropleth' in viz_lower or 'map' in viz_lower:
                    simple_viz_type = 'choropleth'
                elif 'candlestick' in viz_lower:
                    simple_viz_type = 'candlestick'
                elif 'timeline' in viz_lower:
                    simple_viz_type = 'timeline'
                elif 'triple' in viz_lower:
                    simple_viz_type = 'triple'
                
                print(f"DEBUG - Inferred viz type from keywords: '{simple_viz_type}'")
                
        # For common simple visualization types, implement directly
        if len(columns) >= 1:
            col = columns[0]
            
            # Single column types
            if simple_viz_type == "histogram" and len(columns) >= 1:
                print(f"DEBUG - Creating histogram for: {col}")
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                print(f"---------- END CREATE_VISUALIZATION: CREATED HISTOGRAM ----------\n")
                return fig
                
            elif simple_viz_type == "bar" and len(columns) == 1:
                print(f"DEBUG - Creating bar chart for single column: {col}")
                # For categorical columns, show value counts
                if not pd.api.types.is_numeric_dtype(df[col]):
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = ['category', 'count']
                    fig = px.bar(value_counts, x='category', y='count', title=f"Count by {col}")
                    print(f"---------- END CREATE_VISUALIZATION: CREATED BAR CHART ----------\n")
                    return fig
                else:
                    # For numeric columns with few unique values, treat as categories
                    if df[col].nunique() <= 20:
                        value_counts = df[col].value_counts().reset_index()
                        value_counts.columns = ['category', 'count']
                        fig = px.bar(value_counts, x='category', y='count', title=f"Count by {col}")
                        print(f"---------- END CREATE_VISUALIZATION: CREATED BAR CHART ----------\n")
                        return fig
                    else:
                        # For many unique values, show histogram
                        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                        print(f"---------- END CREATE_VISUALIZATION: CREATED HISTOGRAM (FALLBACK) ----------\n")
                        return fig
                        
            elif simple_viz_type == "pie" and len(columns) >= 1:
                print(f"DEBUG - Creating pie chart for: {col}")
                # Pie charts work best with limited categories
                if df[col].nunique() <= 20:  # Limit categories for readability
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = ['category', 'count']
                    fig = px.pie(value_counts, names='category', values='count', title=f"Pie Chart of {col}")
                    print(f"---------- END CREATE_VISUALIZATION: CREATED PIE CHART ----------\n")
                    return fig
                else:
                    # Too many categories for pie chart, use bar chart instead
                    print(f"WARNING: Too many categories for pie chart. Using bar chart.")
                    value_counts = df[col].value_counts().nlargest(20).reset_index()  # Top 20 categories
                    value_counts.columns = ['category', 'count']
                    fig = px.bar(value_counts, x='category', y='count', title=f"Top 20 Categories of {col}")
                    print(f"---------- END CREATE_VISUALIZATION: CREATED BAR CHART (FALLBACK) ----------\n")
                    return fig
                    
            elif simple_viz_type == "box" and len(columns) >= 1:
                print(f"DEBUG - Creating box plot for: {col}")
                if pd.api.types.is_numeric_dtype(df[col]):
                    fig = px.box(df, y=col, title=f"Box Plot of {col}")
                    print(f"---------- END CREATE_VISUALIZATION: CREATED BOX PLOT ----------\n")
                    return fig
                else:
                    # Can't create box plot for non-numeric data
                    print(f"WARNING: Cannot create box plot for non-numeric column. Using bar chart.")
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = ['category', 'count']
                    fig = px.bar(value_counts, x='category', y='count', title=f"Count by {col}")
                    print(f"---------- END CREATE_VISUALIZATION: CREATED BAR CHART (FALLBACK) ----------\n")
                    return fig
                    
            elif simple_viz_type == "violin" and len(columns) >= 1:
                print(f"DEBUG - Creating violin plot for: {col}")
                if pd.api.types.is_numeric_dtype(df[col]):
                    fig = px.violin(df, y=col, title=f"Violin Plot of {col}")
                    print(f"---------- END CREATE_VISUALIZATION: CREATED VIOLIN PLOT ----------\n")
                    return fig
                else:
                    # Can't create violin plot for non-numeric data
                    print(f"WARNING: Cannot create violin plot for non-numeric column. Using bar chart.")
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = ['category', 'count']
                    fig = px.bar(value_counts, x='category', y='count', title=f"Count by {col}")
                    print(f"---------- END CREATE_VISUALIZATION: CREATED BAR CHART (FALLBACK) ----------\n")
                    return fig
                    
            elif simple_viz_type == "treemap" and len(columns) >= 1:
                print(f"DEBUG - Creating treemap for: {col}")
                # Treemap works best with categorical data
                if not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() <= 30:
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = ['category', 'count']
                    # Use path as a list with single column name and values from counts
                    fig = px.treemap(value_counts, path=['category'], values='count', title=f"Treemap of {col}")
                    print(f"---------- END CREATE_VISUALIZATION: CREATED TREEMAP ----------\n")
                    return fig
                else:
                    # Too many numeric values for treemap, use bar chart
                    print(f"WARNING: Too many unique numeric values for treemap. Using bar chart.")
                    value_counts = df[col].value_counts().nlargest(30).reset_index()  # Top 30 values
                    value_counts.columns = ['category', 'count']
                    fig = px.bar(value_counts, x='category', y='count', title=f"Top 30 Values of {col}")
                    print(f"---------- END CREATE_VISUALIZATION: CREATED BAR CHART (FALLBACK) ----------\n")
                    return fig
        
        # For two-column visualizations
        if len(columns) >= 2:
            col1, col2 = columns[0], columns[1]
            
            # Get column types
            is_numeric1 = pd.api.types.is_numeric_dtype(df[col1])
            is_numeric2 = pd.api.types.is_numeric_dtype(df[col2])
            is_temporal1 = pd.api.types.is_datetime64_any_dtype(df[col1]) or is_temporal_column(df, col1)
            is_temporal2 = pd.api.types.is_datetime64_any_dtype(df[col2]) or is_temporal_column(df, col2)
            
            if simple_viz_type == "scatter" and is_numeric1 and is_numeric2:
                print(f"DEBUG - Creating scatter plot for: {col1}, {col2}")
                fig = px.scatter(df, x=col1, y=col2, title=f"{col2} vs {col1}")
                print(f"---------- END CREATE_VISUALIZATION: CREATED SCATTER PLOT ----------\n")
                return fig
                
            elif simple_viz_type == "line" and len(columns) >= 2:
                print(f"DEBUG - Creating line chart for: {col1}, {col2}")
                # Line charts work best with x as temporal or ordered numeric
                if is_temporal1:
                    # Sort data by temporal column
                    sorted_df = df.sort_values(col1)
                    fig = px.line(sorted_df, x=col1, y=col2, title=f"{col2} over {col1}")
                elif is_temporal2:
                    # Use col2 as x if it's temporal
                    sorted_df = df.sort_values(col2)
                    fig = px.line(sorted_df, x=col2, y=col1, title=f"{col1} over {col2}")
                elif is_numeric1:
                    # For numeric x, sort it
                    sorted_df = df.sort_values(col1)
                    fig = px.line(sorted_df, x=col1, y=col2, title=f"{col2} vs {col1}")
                else:
                    # Otherwise just use the columns as is
                    fig = px.line(df, x=col1, y=col2, title=f"{col2} vs {col1}")
                print(f"---------- END CREATE_VISUALIZATION: CREATED LINE CHART ----------\n")
                return fig
                
            elif simple_viz_type == "bar" and len(columns) >= 2:
                print(f"DEBUG - Creating bar chart for pair: {col1}, {col2}")
                
                # For bar charts, typically one axis is categorical and one is numeric
                if is_numeric1 and not is_numeric2:
                    # col2 is categorical, col1 is numeric
                    fig = px.bar(df, x=col2, y=col1, title=f"{col1} by {col2}")
                elif is_numeric2 and not is_numeric1:
                    # col1 is categorical, col2 is numeric
                    fig = px.bar(df, x=col1, y=col2, title=f"{col2} by {col1}")
                elif not is_numeric1 and not is_numeric2:
                    # Both categorical, create count-based bar chart
                    grouped = df.groupby(col1)[col2].count().reset_index()
                    fig = px.bar(grouped, x=col1, y=col2, title=f"Count of {col2} by {col1}")
                else:
                    # Both numeric, use binning for x-axis
                    fig = px.bar(df, x=col1, y=col2, title=f"Sum of {col2} by {col1}")
                print(f"---------- END CREATE_VISUALIZATION: CREATED BAR CHART FOR PAIR ----------\n")
                return fig
                
            elif simple_viz_type == "heatmap" and len(columns) >= 2:
                print(f"DEBUG - Creating heatmap for: {col1}, {col2}")
                
                if is_numeric1 and is_numeric2:
                    # For numeric columns, create correlation heatmap
                    corr_df = df[[col1, col2]].corr()
                    fig = px.imshow(corr_df, title=f"Correlation between {col1} and {col2}")
                else:
                    # For categorical columns, create crosstab
                    crosstab = pd.crosstab(df[col2], df[col1])
                    fig = px.imshow(crosstab, title=f"Heatmap of {col2} vs {col1}")
                print(f"---------- END CREATE_VISUALIZATION: CREATED HEATMAP ----------\n")
                return fig
                
            elif simple_viz_type == "box" and len(columns) >= 2:
                print(f"DEBUG - Creating box plot for pair: {col1}, {col2}")
                
                # For box plots, typically one is categorical and one is numeric
                if is_numeric1 and not is_numeric2:
                    # col1 is numeric, col2 is categorical
                    fig = px.box(df, x=col2, y=col1, title=f"Distribution of {col1} by {col2}")
                elif is_numeric2 and not is_numeric1:
                    # col2 is numeric, col1 is categorical
                    fig = px.box(df, x=col1, y=col2, title=f"Distribution of {col2} by {col1}")
                elif is_numeric1 and is_numeric2:
                    # Both numeric, use col1 as y
                    fig = px.box(df, y=col1, title=f"Distribution of {col1}")
                else:
                    # Both categorical, count occurrences
                    print(f"WARNING: Cannot create box plot with two categorical columns. Using bar chart.")
                    crosstab = pd.crosstab(df[col1], df[col2])
                    fig = px.bar(crosstab.reset_index(), x=col1, y=col2, title=f"Count of {col2} by {col1}")
                print(f"---------- END CREATE_VISUALIZATION: CREATED BOX PLOT FOR PAIR ----------\n")
                return fig
                
            elif simple_viz_type == "treemap" and len(columns) >= 2:
                print(f"DEBUG - Creating treemap for pair: {col1}, {col2}")
                
                if is_numeric2 and not is_numeric1:
                    # col1 is categorical, col2 is numeric - this is the ideal case
                    # Group by the categorical column and sum the numeric column
                    grouped = df.groupby(col1)[col2].sum().reset_index()
                    # Use path as a list with single column name (categorical) and values from the numeric column
                    fig = px.treemap(grouped, path=[col1], values=col2, title=f"Sum of {col2} by {col1}")
                    print(f"---------- END CREATE_VISUALIZATION: CREATED TREEMAP FOR PAIR ----------\n")
                    return fig
                elif is_numeric1 and not is_numeric2:
                    # col2 is categorical, col1 is numeric
                    grouped = df.groupby(col2)[col1].sum().reset_index()
                    fig = px.treemap(grouped, path=[col2], values=col1, title=f"Sum of {col1} by {col2}")
                    print(f"---------- END CREATE_VISUALIZATION: CREATED TREEMAP FOR PAIR ----------\n")
                    return fig
                elif not is_numeric1 and not is_numeric2:
                    # Both categorical - use counts
                    crosstab = pd.crosstab(df[col1], df[col2])
                    # Reshape for treemap
                    stacked = crosstab.stack().reset_index()
                    stacked.columns = [col1, col2, 'count']
                    fig = px.treemap(stacked, path=[col1, col2], values='count', 
                                    title=f"Hierarchical view of {col1} and {col2}")
                    print(f"---------- END CREATE_VISUALIZATION: CREATED TREEMAP FOR CATEGORICAL PAIR ----------\n")
                    return fig
                else:
                    # Both numeric - not ideal for treemap, try binning one column
                    print(f"WARNING: Both columns numeric, binning the first column for treemap")
                    # Create bins for the first column
                    df_binned = df.copy()
                    bin_count = min(10, df[col1].nunique())
                    df_binned[f'{col1}_binned'] = pd.cut(df[col1], bins=bin_count)
                    # Group by the binned column and sum the second column
                    grouped = df_binned.groupby(f'{col1}_binned')[col2].sum().reset_index()
                    # Convert bins to strings for treemap path
                    grouped[f'{col1}_binned'] = grouped[f'{col1}_binned'].astype(str)
                    fig = px.treemap(grouped, path=[f'{col1}_binned'], values=col2, 
                                   title=f"Sum of {col2} by binned {col1}")
                    print(f"---------- END CREATE_VISUALIZATION: CREATED TREEMAP WITH BINNING ----------\n")
                    return fig
        
        # Special handling for pair visualizations with 2+ columns
        if is_pair and len(columns) >= 2:
            print(f"DEBUG - Handling pair visualization with {len(columns)} columns")
            # Implement logic for pair visualizations with more than 2 columns
            # This might involve creating a scatter plot matrix or a different type of visualization
            # For now, we'll just use a scatter plot matrix
            print(f"DEBUG - Creating scatter plot matrix for pair with {len(columns)} columns")
            fig = px.scatter_matrix(df, dimensions=columns, title=f"Scatter Plot Matrix of {', '.join(columns)}")
            print(f"---------- END CREATE_VISUALIZATION: CREATED SCATTER PLOT MATRIX ----------\n")
            return fig
        
        # For other non-pair, non-triple visualizations, continue with existing code
        print(f"DEBUG - Handling general visualization case for type: '{simple_viz_type}'")
        
        # If no valid columns remain, use fallbacks
        if not columns:
            print(f"WARNING: No valid columns remain for visualization")
            # Try to use the first column as fallback
            if len(df.columns) > 0:
                columns = [df.columns[0]]
                print(f"DEBUG - Using first column as fallback: {columns[0]}")
        
        # Based on different visualization types with basic fallbacks
        if len(columns) == 1:
            col = columns[0]
            print(f"DEBUG - Processing single column visualization for: {col}")
            
            # Create different visualizations based on the column type
            if simple_viz_type in ["histogram", "dist"]:
                print(f"DEBUG - Creating histogram for: {col}")
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                print(f"---------- END CREATE_VISUALIZATION: CREATED HISTOGRAM ----------\n")
                return fig
            elif simple_viz_type in ["box", "boxplot"]:
                print(f"DEBUG - Creating box plot for: {col}")
                fig = px.box(df, y=col, title=f"Box Plot of {col}")
                print(f"---------- END CREATE_VISUALIZATION: CREATED BOX PLOT ----------\n")
                return fig
            elif simple_viz_type in ["violin"]:
                print(f"DEBUG - Creating violin plot for: {col}")
                fig = px.violin(df, y=col, title=f"Violin Plot of {col}")
                print(f"---------- END CREATE_VISUALIZATION: CREATED VIOLIN PLOT ----------\n")
                return fig
            elif simple_viz_type in ["pie", "donut"]:
                print(f"DEBUG - Creating pie chart for: {col}")
                # Only works for categorical data
                if not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() < 10:
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = ['category', 'count']
                    fig = px.pie(value_counts, names='category', values='count', title=f"Pie Chart of {col}")
                    print(f"---------- END CREATE_VISUALIZATION: CREATED PIE CHART ----------\n")
                    return fig
                else:
                    # Fallback for numeric columns with many unique values
                    print(f"WARNING: Cannot create pie chart for numeric column with many values. Creating histogram instead.")
                    fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                    print(f"---------- END CREATE_VISUALIZATION: CREATED HISTOGRAM (FALLBACK) ----------\n")
                    return fig
            elif simple_viz_type in ["bar", "count"]:
                print(f"DEBUG - Creating bar chart for: {col}")
                # For categorical columns, show value counts
                if not pd.api.types.is_numeric_dtype(df[col]):
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = ['category', 'count']
                    fig = px.bar(value_counts, x='category', y='count', title=f"Count by {col}")
                    print(f"---------- END CREATE_VISUALIZATION: CREATED BAR CHART ----------\n")
                    return fig
                else:
                    # For numeric columns, show histogram
                    fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                    print(f"---------- END CREATE_VISUALIZATION: CREATED HISTOGRAM (FALLBACK) ----------\n")
                    return fig
            else:
                # Default to histogram for any other type with single column
                print(f"DEBUG - Creating default histogram for: {col}")
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                print(f"---------- END CREATE_VISUALIZATION: CREATED DEFAULT HISTOGRAM ----------\n")
                return fig
        
        # Generic fallback visualization based on column types
        if len(columns) >= 2:
            col1, col2 = columns[0], columns[1]
            print(f"DEBUG - Creating generic visualization with columns: {col1}, {col2}")
            
            # Determine column types
            is_numeric1 = pd.api.types.is_numeric_dtype(df[col1])
            is_numeric2 = pd.api.types.is_numeric_dtype(df[col2])
            
            # Create different visualizations based on column types
            if is_numeric1 and is_numeric2:
                # Both numeric - scatter plot
                print(f"DEBUG - Both columns numeric, creating scatter plot")
                fig = px.scatter(df, x=col1, y=col2, title=f"{col2} vs {col1}")
                print(f"---------- END CREATE_VISUALIZATION: CREATED SCATTER PLOT (FALLBACK) ----------\n")
                return fig
            elif is_numeric1 and not is_numeric2:
                # Numeric + categorical - bar chart
                print(f"DEBUG - First column numeric, second categorical, creating bar chart")
                fig = px.bar(df, x=col2, y=col1, title=f"{col1} by {col2}")
                print(f"---------- END CREATE_VISUALIZATION: CREATED BAR CHART (FALLBACK) ----------\n")
                return fig
            elif not is_numeric1 and is_numeric2:
                # Categorical + numeric - bar chart
                print(f"DEBUG - First column categorical, second numeric, creating bar chart")
                fig = px.bar(df, x=col1, y=col2, title=f"{col2} by {col1}")
                print(f"---------- END CREATE_VISUALIZATION: CREATED BAR CHART (FALLBACK) ----------\n")
                return fig
            else:
                # Both categorical - heatmap or crosstab
                print(f"DEBUG - Both columns categorical, creating heatmap")
                try:
                    # Try to create a crosstab heatmap
                    crosstab = pd.crosstab(df[col2], df[col1])
                    fig = px.imshow(crosstab, title=f"Heatmap of {col2} vs {col1}")
                    print(f"---------- END CREATE_VISUALIZATION: CREATED HEATMAP (FALLBACK) ----------\n")
                    return fig
                except:
                    # If that fails, try a stacked bar chart
                    print(f"WARNING: Could not create heatmap, trying stacked bar")
                    value_counts = df.groupby([col1, col2]).size().reset_index(name='count')
                    fig = px.bar(value_counts, x=col1, y='count', color=col2, title=f"Stacked Counts of {col1} by {col2}")
                    print(f"---------- END CREATE_VISUALIZATION: CREATED STACKED BAR (FALLBACK) ----------\n")
                    return fig
        
        # Last resort - create empty figure with message
        print(f"WARNING: Could not create any visualization, creating empty figure")
        fig = go.Figure()
        fig.add_annotation(
            text="Could not create appropriate visualization for the selected columns",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Visualization Not Available")
        print(f"---------- END CREATE_VISUALIZATION: CREATED EMPTY FIGURE ----------\n")
        return fig
        
    except Exception as e:
        print(f"ERROR in create_visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        # Create an empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Visualization error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        print(f"---------- END CREATE_VISUALIZATION: ERROR ----------\n")
        return fig

def main():
    # Display step progress at the top
    step_col1, step_col2, step_col3, step_col4 = st.columns(4)
    with step_col1:
        st.markdown(f"### {t('🔵' if st.session_state.dashboard_step == 1 else '✅')} {t('Step 1: Domain')}")
    with step_col2:
        if st.session_state.dashboard_step >= 2:
            st.markdown(f"### {t('🔵' if st.session_state.dashboard_step == 2 else '✅')} {t('Step 2: Metrics')}")
        else:
            st.markdown(f"### {t('⚪')} {t('Step 2: Metrics')}")
    with step_col3:
        if st.session_state.dashboard_step >= 3:
            st.markdown(f"### {t('🔵' if st.session_state.dashboard_step == 3 else '✅')} {t('Step 3: Visualization')}")
        else:
            st.markdown(f"### {t('⚪')} {t('Step 3: Visualization')}")
    with step_col4:
        if st.session_state.dashboard_step >= 4:
            st.markdown(f"### {t('🔵')} {t('Step 4: Dashboard')}")
        else:
            st.markdown(f"### {t('⚪')} {t('Step 4: Dashboard')}")
    
    # Simple progress bar
    st.progress(st.session_state.dashboard_step / 4)
    
    # Display the current step
    if st.session_state.dashboard_step == 1:
        domain_step()
    elif st.session_state.dashboard_step == 2:
        metrics_recommendation_step()
    elif st.session_state.dashboard_step == 3:
        visualization_recommendation_step()
    elif st.session_state.dashboard_step == 4:
        dashboard_layout_step()

if __name__ == "__main__":
    main() 