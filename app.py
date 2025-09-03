import streamlit as st
from importlib import reload
import backend

backend = reload(backend)
search_profiles = backend.search_profiles

st.title("🔍 AI Profile Search (Optional Filters)")

# --- Query input ---
query = st.text_input("Enter requirement (e.g., 'React developer with AWS in Bangalore')")

# --- Optional Filters ---
with st.expander("⚙️ Apply Filters (optional)"):
    c1, c2 = st.columns(2)
    with c1:
        min_rating_opt = st.selectbox("⭐ Exact Rating (optional)", ["No filter", 1, 2, 3, 4, 5], index=0)
        min_rating = None if min_rating_opt == "No filter" else int(min_rating_opt)
    with c2:
        use_exp = st.checkbox("Filter by exact experience?", value=False)
        min_exp = st.slider("💼 Exact Experience (years)", 0, 20, 3, step=1) if use_exp else None

# --- Run search ---
if query:
    kwargs = {}
    if min_rating is not None:
        kwargs["min_rating"] = min_rating
    if min_exp is not None:
        kwargs["min_exp"] = int(min_exp)

    matches = search_profiles(query, top_k=10, **kwargs)

    # Results
    if matches:
        st.subheader("Top Matches")
        for score, meta in matches:
            st.markdown(f"""
            ### 👤 {meta.get('name','')}  
            **Skills:** {meta.get('skills','')}  
            **Description:** {meta.get('description','')}  
            **Experience:** {meta.get('experience','')}  
            **Location:** {meta.get('location','')}  
            **⭐ Rating:** {meta.get('rating', 0)}  
            _(Boosted Score: {score:.2f})_
            """)
            st.markdown("---")
    else:
        st.warning("No profiles found that meet your requirements.")
