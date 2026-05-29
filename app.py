import streamlit as st
from inference import SASRecInference
from load_movies import load_movie_titles

st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("Movie Recommendations")
st.markdown("Tell us what movies you liked, get personalized suggestions")

# ============================================================================
# LOAD MODEL & MOVIES
# ============================================================================
@st.cache_resource
def load_model():
    return SASRecInference(device="cpu")

@st.cache_resource
def load_titles():
    return load_movie_titles()

try:
    model = load_model()
    movie_titles = load_titles()
    
    # Create reverse mapping: title -> id
    title_to_id = {title: mid for mid, title in movie_titles.items()}
    
    ready = True
except Exception as e:
    st.error(f"Error: {e}")
    ready = False

# ============================================================================
# MAIN UI - USER FRIENDLY
# ============================================================================
if ready:
    st.markdown("### Search & Select Movies You've Watched")
    
    # Multiselect with search capability
    selected_titles = st.multiselect(
        "Start typing a movie name...",
        options=sorted(movie_titles.values()),
        default=[],
        placeholder="e.g., Toy Story, The Matrix, Inception...",
        help="Type to search for movies"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        n_recs = st.number_input("How many recommendations?", min_value=1, max_value=20, value=10)
    
    with col2:
        st.write("")  # Spacing
        st.write("")
        submit = st.button("Get Recommendations", type="primary", use_container_width=True)
    
    if submit:
        if len(selected_titles) == 0:
            st.warning("Please select at least one movie")
        else:
            # Convert titles to IDs
            watched_ids = [title_to_id[title] for title in selected_titles]
            
            with st.spinner("Finding movies you'll love..."):
                recommendations = model.recommend(watched_ids, int(n_recs))
            
            st.success("Done!")
            
            # Show what user selected
            with st.expander("You selected:"):
                for title in selected_titles:
                    st.write(f"• {title}")
            
            st.divider()
            
            st.markdown("### Recommended For You")
            
            for rank, (item_id, score) in enumerate(recommendations, 1):
                title = movie_titles.get(item_id, f"Movie {item_id}")
                normalized = min(abs(score) / 10, 1.0)
                
                col_rank, col_title, col_bar = st.columns([0.5, 2, 2])
                
                with col_rank:
                    st.write(f"**#{rank}**")
                with col_title:
                    st.write(f"**{title}**")
                with col_bar:
                    st.progress(normalized)
    
    st.divider()
    st.caption("Built with SASRec • Self-Attentive Sequential Recommendation")

else:
    st.error("System not ready")