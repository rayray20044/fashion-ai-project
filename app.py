import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ATELIER · Fashion Recommender",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design system ───────────────────────────────────────────────────────────────
# Palette: parchment cream / ink / warm stone / faded gold
# Type: DM Serif Display (editorial) + DM Sans (utility)
# Signature element: a thin horizontal rule with a centred diamond ✦ — used
# as a section divider. Quiet, precise, unmistakably fashion-editorial.

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #F5F0E8;
    color: #1A1A1A;
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
}

[data-testid="stAppViewContainer"] {
    background-color: #F5F0E8;
}

[data-testid="stHeader"] {
    background-color: #F5F0E8;
    border-bottom: 1px solid #D4C9B0;
}

/* Hide Streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

/* ── Masthead ── */
.masthead {
    text-align: center;
    padding: 3.5rem 0 1rem;
    letter-spacing: 0.35em;
    font-size: 0.7rem;
    font-weight: 500;
    color: #7A6E5F;
    text-transform: uppercase;
}

.brand {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    letter-spacing: 0.08em;
    color: #1A1A1A;
    line-height: 1;
    margin: 0.25rem 0 0.5rem;
}

.tagline {
    font-size: 0.78rem;
    letter-spacing: 0.22em;
    color: #7A6E5F;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* ── Diamond divider (the signature element) ── */
.divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1.5rem 0 2rem;
    color: #B5A88A;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
}
.divider::before, .divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #D4C9B0;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    gap: 0;
    border-bottom: 1px solid #D4C9B0;
    background: transparent;
    justify-content: center;
    padding-bottom: 0;
}

[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #7A6E5F;
    padding: 0.75rem 2.5rem;
    border: none;
    background: transparent;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #1A1A1A;
    border-bottom: 2px solid #1A1A1A;
    background: transparent;
}

/* ── Selectboxes ── */
[data-testid="stSelectbox"] label {
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #7A6E5F;
    font-weight: 500;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSelectbox"] > div > div {
    background-color: transparent;
    border: 1px solid #D4C9B0;
    border-radius: 0;
    color: #1A1A1A;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
}

[data-testid="stSelectbox"] > div > div:hover {
    border-color: #1A1A1A;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    background-color: #1A1A1A;
    color: #F5F0E8;
    border: none;
    border-radius: 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    padding: 0.85rem 3rem;
    width: 100%;
    transition: background 0.2s ease;
}

[data-testid="stButton"] > button:hover {
    background-color: #3D3530;
    color: #F5F0E8;
    border: none;
}

/* ── Product cards ── */
.product-card {
    border: 1px solid #D4C9B0;
    padding: 1.2rem;
    background: #FAF7F2;
    text-align: center;
    transition: border-color 0.2s;
}

.product-card:hover {
    border-color: #1A1A1A;
}

.product-name {
    font-family: 'DM Serif Display', serif;
    font-size: 0.9rem;
    color: #1A1A1A;
    margin: 0.75rem 0 0.3rem;
    line-height: 1.3;
}

.product-meta {
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #7A6E5F;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.62rem;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: #7A6E5F;
    font-weight: 500;
    margin-bottom: 1.5rem;
}

/* ── Prediction chip ── */
.prediction-box {
    border: 1px solid #D4C9B0;
    padding: 1rem 1.5rem;
    background: #FAF7F2;
    text-align: center;
    margin: 1.5rem auto;
    max-width: 320px;
}

.prediction-label {
    font-size: 0.6rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #7A6E5F;
    margin-bottom: 0.35rem;
}

.prediction-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #1A1A1A;
}

/* ── Outfit builder slots ── */
.outfit-slot-label {
    font-size: 0.6rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #7A6E5F;
    text-align: center;
    margin-bottom: 0.5rem;
    font-weight: 500;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #EDE8DE;
    border-right: 1px solid #D4C9B0;
}

/* ── Image placeholder ── */
.img-placeholder {
    background: #EDE8DE;
    height: 180px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    color: #B5A88A;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #7A6E5F;
    font-size: 0.78rem;
    letter-spacing: 0.1em;
}
</style>
""", unsafe_allow_html=True)


# ── Data & models ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("styles.csv", on_bad_lines="skip")
    df = df.dropna(subset=["baseColour", "season", "year", "usage", "productDisplayName"])
    return df.reset_index(drop=True)



def build_outfit(df, gender, colour, season, usage):
    slots = {
        "Top":    ("Apparel", "Topwear"),
        "Bottom": ("Apparel", "Bottomwear"),
        "Shoes":  ("Footwear", "Shoes"),
    }
    results = {}
    for slot, (master, sub) in slots.items():
        pool = df[
            (df["masterCategory"] == master) &
            (df["season"] == season)
        ]
        if pool.empty:
            pool = df[df["masterCategory"] == master]
        if pool.empty:
            continue
        pool_reset = pool.reset_index(drop=True)
        feats = ["gender", "baseColour", "usage"]
        Xp = pd.get_dummies(pool_reset[feats])
        user = pd.DataFrame([{"gender": gender, "baseColour": colour, "usage": usage}])
        user_enc = pd.get_dummies(user).reindex(columns=Xp.columns, fill_value=0)
        nn_slot = NearestNeighbors(n_neighbors=1, metric="hamming")
        nn_slot.fit(Xp)
        _, idx = nn_slot.kneighbors(user_enc)
        results[slot] = pool_reset.iloc[idx[0][0]]
    return results


def show_product(row, label=None):
    """Render a single product card."""
    if label:
        st.markdown(f'<div class="outfit-slot-label">{label}</div>', unsafe_allow_html=True)
    image_path = f"images/{row['id']}.jpg"
    try:
        st.image(image_path, use_container_width=True)
    except Exception:
        st.markdown('<div class="img-placeholder">✦</div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="product-name">{row['productDisplayName']}</div>
        <div class="product-meta">{row['baseColour']} · {row['season']} · {row['usage']}</div>
    """, unsafe_allow_html=True)


# ── Load ───────────────────────────────────────────────────────────────────────
df = load_data()

# ── Masthead ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">SRH Berlin · AI Fashion Project</div>
<div class="brand">ATELIER</div>
<div class="tagline">Personal Style Intelligence</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider">✦</div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["DISCOVER", "OUTFIT"])

# ────────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)

    # Row 1 — Gender, Category, SubCategory (dynamic)
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", sorted(df["gender"].unique()))
    with col2:
        # Only show useful categories
        useful_masters = ["Apparel", "Footwear", "Accessories"]
        master = st.selectbox("Category", useful_masters)
    with col3:
        sub_options = sorted(df[df["masterCategory"] == master]["subCategory"].unique())
        subcat = st.selectbox("Type", sub_options)

    # Row 2 — Colour, Season, Occasion
    col4, col5, col6 = st.columns(3)
    with col4:
        colour = st.selectbox("Colour", sorted(df["baseColour"].unique()))
    with col5:
        season = st.selectbox("Season", sorted(df["season"].unique()))
    with col6:
        usage = st.selectbox("Occasion", sorted(df["usage"].dropna().unique()))

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        search = st.button("Discover Pieces")

    st.markdown('<div class="divider">✦</div>', unsafe_allow_html=True)

    if search:
        # Hard filter: type + season + occasion
        pool = df[
            (df["subCategory"] == subcat) &
            (df["season"] == season) &
            (df["usage"] == usage)
        ].reset_index(drop=True)

        # Relax to type only if nothing matched
        if pool.empty:
            pool = df[df["subCategory"] == subcat].reset_index(drop=True)

        if pool.empty:
            st.markdown('<div class="empty-state">No items found for this combination.</div>',
                        unsafe_allow_html=True)
        else:
            # Rank by colour closeness: exact match → partial string match → rest
            c = colour.lower()
            col_lower = pool["baseColour"].str.lower()
            exact_mask   = col_lower == c
            partial_mask = ~exact_mask & (
                col_lower.str.contains(c, regex=False, na=False) |
                col_lower.apply(lambda v: v in c)
            )
            results = pd.concat([
                pool[exact_mask],
                pool[partial_mask],
                pool[~exact_mask & ~partial_mask],
            ]).head(5)

            st.markdown('<div class="section-label" style="text-align:center">Selected For You</div>',
                        unsafe_allow_html=True)

            cols = st.columns(5)
            for col, (_, row) in zip(cols, results.iterrows()):
                with col:
                    st.markdown('<div class="product-card">', unsafe_allow_html=True)
                    show_product(row)
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="empty-state">
                Select your preferences above<br>and discover your pieces.
            </div>
        """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)

    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        o_gender = st.selectbox("Gender ", sorted(df["gender"].unique()), key="og")
    with oc2:
        o_colour = st.selectbox("Colour ", sorted(df["baseColour"].unique()), key="oc")
    with oc3:
        o_season = st.selectbox("Season ", sorted(df["season"].unique()), key="os")

    o_usage = st.selectbox("Occasion ", sorted(df["usage"].dropna().unique()), key="ou")

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    _, obtn_col, _ = st.columns([2, 1, 2])
    with obtn_col:
        build = st.button("Build Outfit")

    st.markdown('<div class="divider">✦</div>', unsafe_allow_html=True)

    if build:
        outfit = build_outfit(df, o_gender, o_colour, o_season, o_usage)

        if outfit:
            st.markdown('<div class="section-label" style="text-align:center">Your Outfit</div>',
                        unsafe_allow_html=True)
            slot_cols = st.columns(len(outfit))
            for col, (slot, item) in zip(slot_cols, outfit.items()):
                with col:
                    st.markdown('<div class="product-card">', unsafe_allow_html=True)
                    show_product(item, label=slot)
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="empty-state">No outfit could be assembled for this combination.</div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="empty-state">
                Tell us your style and we'll put the look together.
            </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<div style="height:3rem"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; font-size:0.6rem; letter-spacing:0.3em;
     text-transform:uppercase; color:#B5A88A; border-top:1px solid #D4C9B0;
     padding-top:1.5rem; margin-top:1rem;">
    BSDC-DEVDP-28A · Introduction to Artificial Intelligence · SRH Berlin University of Applied Sciences
</div>
""", unsafe_allow_html=True)
