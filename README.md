# AI_Profile_Search
This project is an AI-powered profile search tool built with Streamlit, ChromaDB, and Sentence Transformers. It indexes candidate profiles with skills, experience, ratings, and location, enabling semantic and filter-based search. Users can query requirements, apply filters, and receive ranked matches with boosted scores for relevance and quality.

The app allows recruiters to search for candidate profiles by entering job requirements.

(The user searches for "React developer with AWS in Bangalore". The system applies optional filters for ⭐ rating and 💼 experience, then returns the top matching profiles with details and boosted relevance scores.)

⚙️ Filters Panel

⭐ Exact Rating – Filter candidates by specific ratings (1–5).

💼 Exact Experience – Enable and select specific years of experience.

🏆 Top Matches Output

Results show:

Candidate Name

Skills

Description

Experience

Location

⭐ Rating

Boosted Score (AI similarity + rating + experience weight)
