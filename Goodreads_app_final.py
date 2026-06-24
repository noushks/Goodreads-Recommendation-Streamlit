# =============================================================================
# Goodreads Book Recommender — Streamlit App
# =============================================================================
# To run: streamlit run Goodreads_app_final.py
# Required files in the same folder:
#   - Books.csv, Ratings.csv, best_model.pkl
#
# LLM: gemini-2.5-flash-lite via Google Generative AI (google-genai SDK)
# CF Model: BaselineOnly (best RMSE and Precision@10 in model comparison)
# =============================================================================

import pickle
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field
from google import genai

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Saxa Group 9",
    page_icon="📚",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Cream hero background */
    .stApp { background-color: #f5f0e8; }

    /* Hide default streamlit header padding */
    .block-container { padding-top: 0rem !important; }

    /* ── Hero Banner ── */
    .hero {
        background-color: #f5f0e8;
        padding: 48px 40px 32px 40px;
        border-bottom: 1.5px solid #d6cfc4;
        margin-bottom: 32px;
    }
    .hero-brand {
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: #1a1a1a;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 3.4rem;
        font-weight: 900;
        color: #1a1a1a;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .hero-sub {
        font-size: 1rem;
        color: #555;
        margin-top: 8px;
    }

    /* ── Input Section ── */
    .input-section {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 32px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    .input-section label, .input-section .stSelectbox label {
        font-weight: 600;
        color: #1a1a1a !important;
        font-size: 0.92rem;
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 4px;
        margin-top: 8px;
    }
    .section-caption {
        font-size: 0.85rem;
        color: #555;
        margin-bottom: 16px;
    }

    /* ── CF Results Box — dark sage green ── */
    .cf-box {
        background-color: #4a6741;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 32px;
    }
    .cf-box .section-header { color: #1a2a50 !important; }
    .cf-box .section-caption { color: #2c3e6b !important;  }

    /* CF table rows */
    .cf-row {
        display: flex;
        align-items: center;
        background-color: rgba(255,255,255,0.10);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        color: #f5f0e8;
        font-size: 0.92rem;
    }
    .cf-rank {
        font-weight: 700;
        font-size: 1rem;
        color: ##1a2a50;
        min-width: 32px;
    }
    .cf-title { flex: 1; font-weight: 500; }
    .cf-rating {
        font-size: 0.85rem;
        color: #2c3e6b ;
        margin-left: 12px;
    }

    /* ── AI Re-ranking Box — dark dusty rose ── */
    .ai-box {
        background-color: #7a4a5e;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 32px;
    }
    .ai-box .section-header { color: #f5f0e8; }
    .ai-box .section-caption { color: #e0c4ce; }

    /* Book card inside AI box */
    .book-card {
        background-color: rgba(255,255,255,0.12);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
        display: flex;
        align-items: flex-start;
        gap: 14px;
    }
    .book-cover img { border-radius: 6px; }
    .book-info { flex: 1; }
    .book-title-text {
        font-size: 0.98rem;
        font-weight: 700;
        color: #f5f0e8;
        margin-bottom: 3px;
    }
    .book-meta {
        font-size: 0.80rem;
        color: #e0c4ce;
        margin-bottom: 5px;
    }
    .book-reason {
        font-size: 0.85rem;
        color: #f0e0e6;
        font-style: italic;
    }
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background-color: #e8a0b4;
        color: #4a1a2e;
        border-radius: 50%;
        width: 26px;
        height: 26px;
        font-weight: 700;
        font-size: 0.82rem;
        margin-right: 8px;
        flex-shrink: 0;
        margin-top: 2px;
    }

    /* ── Button ── */
    .stButton > button {
        background-color: #1a1a1a !important;
        color: #f5f0e8 !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 10px 24px !important;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #333 !important;
    }

    /* ── Radio buttons ── */
    .stRadio label { color: #1a1a1a !important; font-size: 0.9rem; }

    /* ── Alerts ── */
    .stAlert { border-radius: 10px; }

    /* ── Model info tag ── */
    .model-tag {
        display: inline-block;
        background-color: #e8e0d5;
        color: #444;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 1 — Load Data & Model
# =============================================================================

@st.cache_data
def load_data():
    books   = pd.read_csv("Books.csv")
    ratings = pd.read_csv("Ratings.csv")

    title_of = dict(zip(books["book_id"], books["title"]))

    meta_of = {
        row.title: {
            "authors":   row.authors,
            "year":      int(row.original_publication_year)
                         if pd.notna(row.original_publication_year) else "N/A",
            "rating":    row.average_rating,
            "image_url": row.image_url if pd.notna(row.image_url) else None,
        }
        for row in books.itertuples()
    }

    counts        = ratings["book_id"].value_counts()
    popular_books = set(counts[counts >= 5].index)
    all_users     = sorted(ratings["user_id"].unique().tolist())

    return books, ratings, title_of, meta_of, popular_books, all_users


@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)


books, ratings, title_of, meta_of, popular_books, all_users = load_data()
model = load_model()


# =============================================================================
# SECTION 2 — Helper Functions
# =============================================================================

def top_n_for_user(model, user_id, top_n=10):
    seen = set(ratings.loc[ratings["user_id"] == user_id, "book_id"])
    scored = [
        (title_of[b], model.predict(user_id, b).est)
        for b in books["book_id"]
        if b not in seen and b in popular_books
    ]
    return sorted(scored, key=lambda x: -x[1])[:top_n]


class Pick(BaseModel):
    title:  str = Field(description="Exact book title from the candidate list.")
    reason: str = Field(description="One sentence on why it fits the reader's mood.")


PROMPT_STYLES = {
    "🎩 Concierge (formal)": (
        "You are a book recommendation concierge. From the candidate list, "
        "re-rank the books based on their suitability for the reader's mood, "
        "with a one-sentence reason. Use the exact titles from the list."
    ),
    "👯 Friend (casual)": (
        "You are a well-read friend who knows the reader personally. "
        "They told you their current mood. From the list below, pick the best "
        "matches and explain each in one casual sentence, as if recommending to "
        "a friend. Use the exact titles from the list."
    ),
}


def llm_rerank(candidates, user_mood, api_key, system_prompt, top_n=10):
    catalog = "\n".join(
        "- {title} | Author: {authors} | Year: {year} | Avg Rating: {rating}".format(
            title=title,
            **{k: v for k, v in meta_of.get(
                title, {"authors": "Unknown", "year": "N/A", "rating": "N/A"}
            ).items() if k != "image_url"}
        )
        for title, _ in candidates
    )
    client   = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"Reader's mood: {user_mood}\n\nCandidate books:\n{catalog}",
        config={
            "system_instruction":  system_prompt,
            "response_mime_type":  "application/json",
            "response_schema":     list[Pick],
            "temperature":         0.2 if "Concierge" in system_prompt else 0.7,
            "max_output_tokens":   1000,
        },
    )
    return response.parsed[:top_n]


# =============================================================================
# SECTION 3 — Hero Banner
# =============================================================================

st.markdown("""
<div class="hero">
    <div class="hero-brand">📚 Saxa Group 9</div>
    <div class="hero-title">Find your next<br>favorite book.</div>
    <div class="hero-sub">Powered by collaborative filtering + AI personalization.</div>
    <div class="model-tag">CF: BaselineOnly &nbsp;·&nbsp; LLM: gemini-2.5-flash-lite · Google Generative AI</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 4 — Input Panel
# =============================================================================

st.markdown('<div class="input-section">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.2, 1.5, 1.5])

with col1:
    selected_user = st.selectbox(
        "👤 Select User ID",
        options=all_users,
        help="Choose a user to generate recommendations for."
    )

with col2:
    api_key = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        placeholder="Paste your key here...",
        help="Used this session only — never stored."
    )

with col3:
    user_mood = st.text_input(
        "💭 What are you in the mood for?",
        placeholder="e.g. adventurous and fast-paced",
    )

col_radio, col_btn = st.columns([2, 1])
with col_radio:
    prompt_style_label = st.radio(
        "AI Prompt Style",
        options=list(PROMPT_STYLES.keys()),
        horizontal=True,
        help="Concierge: formal & precise. Friend: casual & personal."
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    rerank_button = st.button("✨ Re-rank with AI")

st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# SECTION 5 — CF Top-10 (dark sage green box)
# =============================================================================

with st.spinner("Generating recommendations..."):
    cf_results = top_n_for_user(model, selected_user, top_n=10)

st.markdown("""
<div class="cf-box">
    <div class="section-header">🔵 Step 1 — Collaborative Filtering Top-10</div>
    <div class="section-caption">
        Books scored by the Baseline CF model based on global user &amp; item biases.
        Updates automatically when you change the User ID.
    </div>
</div>
""", unsafe_allow_html=True)

for i, (title, score) in enumerate(cf_results, 1):
    st.markdown(
        f"""
        <div class="cf-row">
            <span class="cf-rank">{i}</span>
            <span class="cf-title">{title}</span>
            <span class="cf-rating">⭐ {score:.2f}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# SECTION 6 — AI Re-ranking (dark dusty rose box)
# =============================================================================

st.markdown("""
<div class="ai-box">
    <div class="section-header">✨ Step 2 — AI-Powered Re-ranking</div>
    <div class="section-caption">
        The LLM re-ranks the Top-15 CF candidates to match your stated mood,
        using title, author, year, and avg rating as context.
        Cover images are sourced from the Books dataset.
    </div>
""", unsafe_allow_html=True)

if rerank_button:
    if not api_key:
        st.warning("Please paste your Gemini API key above.")
    elif not user_mood:
        st.warning("Please enter a mood or vibe above.")
    else:
        with st.spinner("Asking Gemini to re-rank your books..."):
            cf_candidates_15 = top_n_for_user(model, selected_user, top_n=15)
            try:
                reranked = llm_rerank(
                    cf_candidates_15, user_mood, api_key,
                    system_prompt=PROMPT_STYLES[prompt_style_label],
                    top_n=10
                )

                st.success(
                    f"Re-ranked for: *\"{user_mood}\"* · Style: **{prompt_style_label}**"
                )

                for i, pick in enumerate(reranked, 1):
                    meta      = meta_of.get(pick.title, {})
                    image_url = meta.get("image_url")
                    author    = meta.get("authors", "Unknown")
                    year      = meta.get("year", "N/A")
                    avg_rating = meta.get("rating", "N/A")

                    col_img, col_text = st.columns([1, 6])
                    with col_img:
                        if image_url:
                            st.image(image_url, width=75)
                        else:
                            st.markdown("📕")
                    with col_text:
                        st.markdown(
                            f"""
                            <div style="padding:6px 0;">
                                <div class="book-title-text">
                                    <span class="rank-badge">{i}</span>{pick.title}
                                </div>
                                <div class="book-meta">{author} · {year} · ⭐ {avg_rating}</div>
                                <div class="book-reason">"{pick.reason}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            except Exception as e:
                st.error(f"Gemini API error: {e}")
else:
    st.markdown(
        '<div style="color:#7e1e9c; font-size:0.92rem; padding:8px 0;">'
        'Fill in your mood above and click <b style="color:#;7e1e9c>✨ Re-rank with AI</b> '
        'to personalize your picks. Try switching prompt styles to compare results!'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)