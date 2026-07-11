import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Fashion Recommender", layout="centered")

# ---------------------------------------------------------------------
# Load + prep data (cached so it only runs once, not on every click)
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("styles.csv", on_bad_lines="skip")
    df_clean = df.dropna(subset=["baseColour", "season", "year", "usage", "productDisplayName"])
    return df_clean

@st.cache_resource
def train_models(df_clean):
    features = ["gender", "masterCategory", "subCategory", "baseColour", "season", "usage"]
    target = "articleType"

    X = df_clean[features]
    y = df_clean[target]
    X_encoded = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42
    )

    clf_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_forest.fit(X_train, y_train)
    accuracy_forest = accuracy_score(y_test, clf_forest.predict(X_test))

    recommender = NearestNeighbors(n_neighbors=5, metric="euclidean")
    recommender.fit(X_encoded)

    return clf_forest, accuracy_forest, recommender, X_encoded

@st.cache_resource
def train_outfit_models(df_clean):
    """
    Stores each outfit slot's raw data (Top / Bottom / Shoes). Matching happens
    at request time in build_outfit(), which filters by season FIRST, then finds
    the closest match on gender/colour/usage within that season. This avoids
    picking an item that matches colour but is the wrong season (e.g. shorts
    for a winter outfit).
    """
    slot_filters = {
        "Top": df_clean["subCategory"] == "Topwear",
        "Bottom": df_clean["subCategory"] == "Bottomwear",
        "Shoes": df_clean["masterCategory"] == "Footwear",
    }
    return {slot: df_clean[mask].reset_index(drop=True) for slot, mask in slot_filters.items()}

df_clean = load_data()
clf_forest, accuracy_forest, recommender, X_encoded = train_models(df_clean)
slot_data = train_outfit_models(df_clean)

def build_outfit(gender, baseColour, season, usage):
    """
    Returns one matching item per slot (Top, Bottom, Shoes).
    Filters to the correct season first (falling back to the full slot only
    if nothing in that season exists), then finds the closest match on
    gender/colour/usage within that season-correct pool. This guarantees
    season correctness is never sacrificed just to match colour.
    """
    outfit = {}
    for slot, sub_df in slot_data.items():
        season_matched = sub_df[sub_df["season"] == season]
        pool = season_matched if len(season_matched) > 0 else sub_df

        feats_encoded = pd.get_dummies(pool[["gender", "baseColour", "usage"]])
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(feats_encoded)

        user_input = pd.DataFrame([{"gender": gender, "baseColour": baseColour, "usage": usage}])
        user_encoded = pd.get_dummies(user_input).reindex(columns=feats_encoded.columns, fill_value=0)
        _, idx = nn.kneighbors(user_encoded, n_neighbors=1)

        outfit[slot] = pool.reset_index(drop=True).iloc[idx[0][0]]
    return outfit

# ---------------------------------------------------------------------
# Recommender function (same logic tested earlier in the notebook)
# ---------------------------------------------------------------------
def recommend(gender, masterCategory, subCategory, baseColour, season, usage, n=5):
    user_input = pd.DataFrame([{
        "gender": gender, "masterCategory": masterCategory,
        "subCategory": subCategory, "baseColour": baseColour,
        "season": season, "usage": usage
    }])
    user_encoded = pd.get_dummies(user_input).reindex(columns=X_encoded.columns, fill_value=0)
    distances, indices = recommender.kneighbors(user_encoded, n_neighbors=n)
    return df_clean.iloc[indices[0]][
        ["id", "productDisplayName", "articleType", "baseColour", "season", "usage"]
    ]

def predict_category(gender, masterCategory, subCategory, baseColour, season, usage):
    user_input = pd.DataFrame([{
        "gender": gender, "masterCategory": masterCategory,
        "subCategory": subCategory, "baseColour": baseColour,
        "season": season, "usage": usage
    }])
    user_encoded = pd.get_dummies(user_input).reindex(columns=X_encoded.columns, fill_value=0)
    return clf_forest.predict(user_encoded)[0]

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("👕 Fashion Recommendation System")
st.caption(
    f"Trained on {len(df_clean):,} products · Random Forest classifier accuracy: {accuracy_forest*100:.2f}%"
)

st.write(
    "Use **Item Recommender** to find products similar to a specific preference, "
    "or **Outfit Builder** to assemble a complete, colour-coordinated outfit."
)

tab1, tab2 = st.tabs(["🔎 Item Recommender", "👔 Outfit Builder"])

# =======================================================================
# TAB 1 - Item Recommender + Classifier (existing feature)
# =======================================================================
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", sorted(df_clean["gender"].unique()), key="t1_gender")
        masterCategory = st.selectbox("Master Category", sorted(df_clean["masterCategory"].unique()), key="t1_master")
        sub_options = sorted(df_clean[df_clean["masterCategory"] == masterCategory]["subCategory"].unique())
        subCategory = st.selectbox("Sub Category", sub_options, key="t1_sub")

    with col2:
        baseColour = st.selectbox("Colour", sorted(df_clean["baseColour"].unique()), key="t1_colour")
        season = st.selectbox("Season", sorted(df_clean["season"].unique()), key="t1_season")
        usage = st.selectbox("Usage", sorted(df_clean["usage"].unique()), key="t1_usage")

    if st.button("Get Recommendations", type="primary"):
        st.subheader("🔎 Recommended Items")
        results = recommend(gender, masterCategory, subCategory, baseColour, season, usage)

        # Show each recommendation as a photo card instead of a plain table.
        # Requires an images/ folder (download "Fashion Product Images Small" from
        # Kaggle) containing files named images/{id}.jpg
        cols = st.columns(len(results))
        for col, (_, row) in zip(cols, results.iterrows()):
            with col:
                image_path = f"images/{row['id']}.jpg"
                try:
                    st.image(image_path, width='stretch')
                except Exception:
                    st.write("📷 (image not found)")
                st.caption(f"**{row['productDisplayName']}**\n\n{row['baseColour']} · {row['season']} · {row['usage']}")

        st.subheader("🤖 Classifier Prediction")
        predicted = predict_category(gender, masterCategory, subCategory, baseColour, season, usage)
        st.info(f"Based on these attributes, the classifier predicts this item is most likely a: **{predicted}**")

# =======================================================================
# TAB 2 - Outfit Builder (new feature)
# =======================================================================
with tab2:
    st.write(
        "Pick a colour, season, and usage - we'll assemble a complete outfit "
        "(top, bottom, shoes) that all match those preferences."
    )

    oc1, oc2 = st.columns(2)
    with oc1:
        o_gender = st.selectbox("Gender", sorted(df_clean["gender"].unique()), key="t2_gender")
        o_colour = st.selectbox("Colour", sorted(df_clean["baseColour"].unique()), key="t2_colour")
    with oc2:
        o_season = st.selectbox("Season", sorted(df_clean["season"].unique()), key="t2_season")
        o_usage = st.selectbox("Usage", sorted(df_clean["usage"].unique()), key="t2_usage")

    if st.button("Build My Outfit", type="primary"):
        st.subheader("👔 Your Outfit")
        outfit = build_outfit(o_gender, o_colour, o_season, o_usage)

        slot_cols = st.columns(3)
        for col, (slot, item) in zip(slot_cols, outfit.items()):
            with col:
                st.markdown(f"**{slot}**")
                image_path = f"images/{item['id']}.jpg"
                try:
                    st.image(image_path, width='stretch')
                except Exception:
                    st.write("📷 (image not found)")
                st.caption(f"{item['productDisplayName']}\n\n{item['baseColour']} · {item['season']} · {item['usage']}")

st.divider()
with st.expander("ℹ️ About this project"):
    st.write(
        "Built for BSDC-DEVDP-28A · Introduction to Artificial Intelligence. "
        "The recommender uses K-Nearest Neighbors similarity search over one-hot "
        "encoded product attributes to return comparable items. The classifier is a "
        "separate Random Forest model predicting a single articleType label, included "
        "to demonstrate supervised classification performance and its limitations "
        "(see report for the subCategory-hierarchy finding)."
    )
