import streamlit as st
from backend import search_profiles

st.set_page_config(page_title="🔍 AI Profile Search", layout="wide")

st.title("🔍 AI Profile Search")
query = st.text_input("Enter requirement (e.g., 'React developer with AWS in Bangalore')")

with st.expander("⚙️ Apply Filters (optional)"):
    min_rating = st.number_input("⭐ Exact Rating", min_value=1, max_value=5, step=1, value=None, placeholder="Select rating")
    min_exp = st.number_input("💼 Exact Experience (years)", min_value=0, max_value=50, step=1, value=None, placeholder="Select years")

if st.button("Search"):
    if not query.strip():
        st.warning("⚠️ Please enter a query to search.")
    else:
        matches = search_profiles(query, top_k=10, min_rating=min_rating if min_rating else None, min_exp=min_exp if min_exp else None)

        if not matches:
            st.error("🚫 No matching profiles found for your query and filters.")
        else:
            st.subheader("✨ Top Matches")
            for idx, (score, meta) in enumerate(matches, start=1):
                st.markdown(f"""
                <div style="
                    padding:15px;
                    border-radius:10px;
                    background-color:#f9f9f9;
                    margin-bottom:12px;
                    box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
                    color:#000000;  /* ✅ force black text */
                    font-family:Arial, sans-serif;
                ">
                    <h4 style="color:#000000;">👤 {meta.get('name','')}</h4>
                    <p><b>🛠 Skills:</b> {meta.get('skills','')}</p>
                    <p><b>📄 Description:</b> {meta.get('description','')}</p>
                    <p><b>💼 Experience:</b> {meta.get('experience','')}</p>
                    <p><b>📍 Location:</b> {meta.get('location','')}</p>
                    <p><b>⭐ Rating:</b> {meta.get('rating', 0)}</p>
                    <small style="color:gray;">Boosted Score: {score:.2f}</small>
                </div>
                """, unsafe_allow_html=True)
                # st.markdown("---")


