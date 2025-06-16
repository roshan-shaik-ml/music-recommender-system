"""
Song Finder UI Application

This Streamlit application provides a user interface for searching songs,
fetching lyrics, and finding similar songs using the lyrics.ovh API and ChromaDB.
"""

import streamlit as st
import lyricsgenius
import os
import urllib.parse
import requests
import chromadb
from typing import Dict, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import time
from difflib import SequenceMatcher
import webbrowser
import logging
import json
from datetime import datetime
from rapidfuzz import fuzz

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("song_finder.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="Song Finder UI", layout="wide", initial_sidebar_state="collapsed"
)

# --- Environment Variable Loading ---
load_dotenv()
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")

# --- Custom CSS for a Modern "Gen-Z" Dark Theme ---
st.markdown(
    """
    <style>
    /* --- Base App Styling --- */
    .stApp {
        background-color: #121212;
    }

    /* --- Main Content Container --- */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 900px; /* Center and constrain the content width for readability */
        margin: auto;
    }

    /* --- Typography --- */
    h1, h2, h3 {
        color: #FFFFFF;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }

    h1 {
        background: -webkit-linear-gradient(45deg, #FF1493, #00FFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-bottom: 1rem;
    }

    h2 {
        border-bottom: 2px solid #FF1493;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    
    p, .stMarkdown {
        color: #B3B3B3;
        font-size: 16px;
    }

    /* --- Search Input Box --- */
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: #FFFFFF;
        padding: 12px 20px;
        font-size: 16px;
        transition: all 0.3s ease;
    }

    .stTextInput>div>div>input:focus {
        border: 1px solid #FF1493;
        box-shadow: 0 0 15px rgba(255, 20, 147, 0.2);
    }

    /* --- Lyrics Text Area --- */
    .stTextArea>div>div>textarea {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: #FFFFFF !important;
        min-height: 350px;
    }
    
    .stTextArea>div>div>textarea:disabled {
        background-color: rgba(255, 255, 255, 0.05);
        color: #FFFFFF !important;
    }

    /* --- Primary Action Buttons (with visible text) --- */
    .stButton>button {
        background: linear-gradient(135deg, #FF1493 0%, #00FFFF 100%);
        color: #FFFFFF;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 20, 147, 0.3);
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #00FFFF 0%, #FF1493 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 255, 255, 0.4);
    }
    
    .stButton>button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 10px rgba(255, 20, 147, 0.4);
    }

    /* --- Card for Similar Songs --- */
    .song-card {
        background: rgba(30, 30, 30, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }

    .song-card:hover {
        border: 1px solid #FF1493;
        transform: translateY(-3px);
    }
    
    /* --- Clickable Suggestion Card --- */
    .suggestion-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid transparent;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease-in-out;
    }
    
    .suggestion-card:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(255,255,255,0.2);
    }

    .song-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #FFFFFF;
    }
    
    .song-artist {
        font-size: 0.9rem;
        color: #B3B3B3;
    }

    /* --- Custom Footer --- */
    .footer {
        text-align: center;
        padding: 20px 0;
        margin-top: 40px;
        color: #888888;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Hide default Streamlit footer */
    .stApp > footer {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- ChromaDB Initialization ---
CHROMA_CLIENT = chromadb.PersistentClient(path="./chroma_db")
COLLECTION_NAME = "lyric_embeddings"

# --- Data Structures and Interfaces ---


@dataclass
class SongData:
    """Data class to store song information."""

    id: int
    title: str
    artist: str
    full_title: str
    lyrics: Optional[str] = None


class LyricsAPI(ABC):
    """Abstract base class for lyrics API implementations."""

    @abstractmethod
    def get_lyrics(self, artist: str, title: str) -> Dict:
        pass

    @abstractmethod
    def get_suggestions(self, query: str) -> List[Dict]:
        pass


class ARLLyricsAPI(LyricsAPI):
    """Implementation of the lyrics.ovh API."""

    def __init__(self):
        self.base_url = "https://api.lyrics.ovh/v1"
        self.suggest_url = "https://api.lyrics.ovh/suggest"
        logger.info("Initialized ARLLyricsAPI")

    def get_lyrics(self, artist: str, title: str) -> Dict:
        url = (
            f"{self.base_url}/{urllib.parse.quote(artist)}/{urllib.parse.quote(title)}"
        )
        logger.info(f"Fetching lyrics for: {title} by {artist}")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            result = response.json()
            if result.get("lyrics"):
                logger.info(f"Successfully fetched lyrics for: {title}")
            else:
                logger.warning(f"No lyrics found for: {title}")
            return result
        except requests.exceptions.Timeout:
            logger.error(f"Timeout while fetching lyrics for: {title}")
            st.error("😢 Lyrics not found.")
            return {"lyrics": None}
        except requests.exceptions.RequestException as e:
            logger.error(f"API Error for {title}: {str(e)}")
            st.error(f"API Error: Could not fetch lyrics. {e}")
            return {"lyrics": None}

    def get_suggestions(self, query: str) -> List[Dict]:
        url = f"{self.suggest_url}/{urllib.parse.quote(query)}"
        logger.info(f"Fetching suggestions for query: {query}")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            suggestions = response.json().get("data", [])
            logger.info(f"Found {len(suggestions)} suggestions for query: {query}")
            return suggestions
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching suggestions: {str(e)}")
            st.warning(f"Could not fetch suggestions: {e}")
            return []


class ChromaDBManager:
    """Manager for ChromaDB operations."""

    def __init__(self):
        self.client = CHROMA_CLIENT
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        logger.info("Initialized ChromaDBManager")

    def _normalize_lyrics(self, lyrics: str) -> str:
        """
        Normalize lyrics text for better comparison:
        - Convert to lowercase
        - Remove punctuation
        - Remove extra whitespace
        - Remove common words (the, and, etc.)
        - Sort words to handle different word orders
        """
        if not lyrics:
            return ""

        # Convert to lowercase and remove punctuation
        lyrics = lyrics.lower()
        lyrics = "".join(char for char in lyrics if char.isalnum() or char.isspace())

        # Split into words and remove common words
        common_words = {
            "the",
            "and",
            "a",
            "an",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
        }
        words = [word for word in lyrics.split() if word not in common_words]

        # Sort words to handle different word orders
        return " ".join(sorted(words))

    def _calculate_lyrics_similarity(self, lyrics1: str, lyrics2: str) -> float:
        """
        Calculate similarity between two lyrics using RapidFuzz's token_sort_ratio.
        This handles word order differences and provides better fuzzy matching.
        Returns a score between 0 and 100, which we normalize to 0-1.
        """
        if not lyrics1 or not lyrics2:
            return 0.0

        # Calculate raw similarity using token_sort_ratio
        # This handles word order differences and provides better fuzzy matching
        raw_similarity = fuzz.token_sort_ratio(lyrics1, lyrics2) / 100.0

        # Calculate normalized similarity
        normalized_lyrics1 = self._normalize_lyrics(lyrics1)
        normalized_lyrics2 = self._normalize_lyrics(lyrics2)
        normalized_similarity = (
            fuzz.token_sort_ratio(normalized_lyrics1, normalized_lyrics2) / 100.0
        )

        # Log both scores for comparison
        logger.info(f"Raw lyrics similarity score: {raw_similarity:.2f}")
        logger.info(f"Normalized lyrics similarity score: {normalized_similarity:.2f}")
        logger.debug(f"Raw lyrics 1 sample: {lyrics1[:100]}...")
        logger.debug(f"Raw lyrics 2 sample: {lyrics2[:100]}...")
        logger.debug(f"Normalized lyrics 1 sample: {normalized_lyrics1[:100]}...")
        logger.debug(f"Normalized lyrics 2 sample: {normalized_lyrics2[:100]}...")

        # Return the normalized similarity for filtering
        return normalized_similarity

    def find_similar_songs(
        self, lyrics: str, current_title: str, current_artist: str, n_results: int = 10
    ) -> List[Dict]:
        """
        Finds similar songs based on lyrics using a direct ChromaDB query.
        """
        if not lyrics:
            logger.warning("Cannot find similar songs without lyrics.")
            return []

        try:
            logger.info(
                f"Finding similar songs for: {current_title} by {current_artist}"
            )
            # Get more results than needed to account for filtering
            results = self.collection.query(
                query_texts=[lyrics],
                n_results=n_results * 2,  # Get more results to filter
            )

            if not results or not results.get("metadatas"):
                logger.info("Could not find any similar songs in the database.")
                return []

            similar_songs = []
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            documents = results["documents"][0]  # Get the documents containing lyrics

            min_dist = min(distances) if distances else 0
            max_dist = max(distances) if distances else 1
            logger.info(f"Distance range - Min: {min_dist}, Max: {max_dist}")

            for i, metadata in enumerate(metadatas):
                title = metadata.get("title", "Unknown Title")
                artist = metadata.get("artist", "Unknown Artist")
                distance = distances[i]
                song_lyrics = documents[i]  # Get lyrics from documents

                # Skip if it's the same song
                if (
                    title.lower() == current_title.lower()
                    and artist.lower() == current_artist.lower()
                ):
                    logger.info(f"Skipping current song: {title}")
                    continue

                # Calculate lyrics similarity
                lyrics_similarity = self._calculate_lyrics_similarity(
                    lyrics, song_lyrics
                )

                # Skip if lyrics are too similar (threshold: 0.7 or 70% similar)
                if lyrics_similarity > 0.7:
                    logger.info(
                        f"Skipping lyrically similar song: {title} (similarity: {lyrics_similarity:.2f})"
                    )
                    continue

                if max_dist == min_dist:
                    similarity_score = 1.0
                else:
                    similarity_score = 1 - (
                        (distance - min_dist) / (max_dist - min_dist)
                    )

                logger.info(
                    f"Found similar song: {title} by {artist} (Score: {similarity_score:.2f}, Lyrics similarity: {lyrics_similarity:.2f})"
                )

                similar_songs.append(
                    {
                        "title": title,
                        "artist": artist,
                        "similarity_score": similarity_score,
                        "lyrics_similarity": lyrics_similarity,
                        "youtube_url": f"https://music.youtube.com/search?q={urllib.parse.quote(f'{title} {artist}')}",
                    }
                )

            # Sort by similarity score to ensure the best matches are first
            similar_songs.sort(key=lambda x: x["similarity_score"], reverse=True)
            logger.info(f"Returning {len(similar_songs)} similar songs after filtering")
            return similar_songs[:n_results]

        except Exception as e:
            logger.error(f"Error finding similar songs: {str(e)}")
            st.error(f"An error occurred while finding similar songs: {e}")
            return []


# --- Main Application UI Class ---


class SongFinderUI:
    """Main UI class for the Song Finder application."""

    def __init__(self):
        self.lyrics_api = ARLLyricsAPI()
        self.chroma_manager = ChromaDBManager()
        self._initialize_session_state()
        logger.info("Initialized SongFinderUI")

    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        defaults = {
            "selected_song_id": None,
            "current_lyrics": None,
            "current_lyrics_song_id": None,
            "suggestions": [],
            "last_query": "",
            "last_suggestion_time": 0,
            "similar_songs": [],
            "selected_song_title": "",
            "selected_song_artist": "",
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        logger.info("Initialized session state")

    def _get_suggestions(self, query: str):
        """Get song suggestions with debouncing."""
        current_time = time.time()
        if query and (
            query != st.session_state["last_query"]
            or current_time - st.session_state["last_suggestion_time"] > 0.5
        ):
            logger.info(f"Getting suggestions for query: {query}")
            st.session_state["last_query"] = query
            st.session_state["last_suggestion_time"] = current_time
            st.session_state["suggestions"] = self.lyrics_api.get_suggestions(query)[:5]
        elif not query:
            st.session_state["suggestions"] = []

    def _display_search_section(self):
        """Display and handle the search section of the UI."""
        st.header("Search for a Song")
        search_query = st.text_input(
            "Enter song title or artist:",
            key="search_input",
            placeholder="e.g., Blinding Lights The Weeknd",
        )

        self._get_suggestions(search_query)

        if st.session_state["suggestions"]:
            for song in st.session_state["suggestions"]:
                if st.button(
                    f"{song['title']} by {song['artist']['name']}",
                    key=f"song_{song['id']}",
                    use_container_width=True,
                ):
                    self._update_session_state(song)
                    st.rerun()

    def _update_session_state(self, song_data: Dict):
        """Update the session state with selected song information."""
        logger.info(
            f"Updating session state with song: {song_data['title']} by {song_data['artist']['name']}"
        )
        st.session_state["selected_song_id"] = song_data["id"]
        st.session_state["selected_song_title"] = song_data["title"]
        st.session_state["selected_song_artist"] = song_data["artist"]["name"]
        st.session_state["current_lyrics"] = None
        st.session_state["current_lyrics_song_id"] = None
        st.session_state["similar_songs"] = []

    def _display_lyrics_section(self):
        """Display the lyrics section of the UI."""
        if not st.session_state["selected_song_id"]:
            return

        st.divider()
        st.header(f"Lyrics for '{st.session_state['selected_song_title']}'")

        # Create a button to listen to the original song on YouTube Music
        title = st.session_state["selected_song_title"]
        artist = st.session_state["selected_song_artist"]
        youtube_url = f"https://music.youtube.com/search?q={urllib.parse.quote(f'{title} {artist}')}"

        # We place the button in a column to control its width and alignment
        _, col_btn, _ = st.columns([1, 2, 1])
        with col_btn:
            if st.button(
                "🎵 Listen on YouTube Music",
                key="listen_original",
                use_container_width=True,
            ):
                webbrowser.open(youtube_url, new=2)

        st.markdown("<br>", unsafe_allow_html=True)  # Add some vertical space

        if (
            st.session_state["current_lyrics_song_id"]
            != st.session_state["selected_song_id"]
        ):
            self._fetch_and_display_lyrics()

        lyrics_to_display = st.session_state.get("current_lyrics", "")
        if lyrics_to_display:
            st.text_area("Lyrics", lyrics_to_display, disabled=True)

    def _fetch_and_display_lyrics(self):
        """Fetch and display lyrics for the selected song."""
        title = st.session_state["selected_song_title"]
        artist = st.session_state["selected_song_artist"]
        logger.info(f"Fetching lyrics for: {title} by {artist}")

        with st.spinner(f"Fetching lyrics for {title}..."):
            lyrics_response = self.lyrics_api.get_lyrics(
                artist=artist,
                title=title,
            )
            raw_lyrics = lyrics_response.get("lyrics")
            if raw_lyrics:
                cleaned_lyrics = raw_lyrics.replace("\\n", "\n").strip()
                st.session_state["current_lyrics"] = cleaned_lyrics
                logger.info(f"Successfully fetched and cleaned lyrics for: {title}")
            else:
                st.session_state["current_lyrics"] = "Lyrics not found for this song."
                logger.warning(f"No lyrics found for: {title}")

            st.session_state["current_lyrics_song_id"] = st.session_state[
                "selected_song_id"
            ]

    def _display_similar_songs_section(self):
        """Display the similar songs section, now with better state handling."""
        lyrics_available = (
            st.session_state.get("current_lyrics")
            and "could not be found" not in st.session_state["current_lyrics"]
        )

        if not lyrics_available:
            return

        st.divider()
        st.header("Discover Similar Songs")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✨ Analyze Lyrics & Find Matches", use_container_width=True):
            with st.spinner("Analyzing lyrics and finding similar tracks..."):
                similar_songs = self.chroma_manager.find_similar_songs(
                    st.session_state["current_lyrics"],
                    st.session_state["selected_song_title"],
                    st.session_state["selected_song_artist"],
                )
                st.session_state["similar_songs"] = similar_songs

        if st.session_state["similar_songs"]:
            st.subheader("Similar Songs Found:")
            cols = st.columns(2, gap="large")
            for idx, song in enumerate(st.session_state["similar_songs"]):
                with cols[idx % 2]:
                    with st.container():
                        st.markdown(
                            f"""
                        <div class="song-card">
                            <div>
                                <div class="song-title">{song['title']}</div>
                                <div class="song-artist">👤 {song['artist']}</div>
                                <p style="font-size: 0.9rem; color: #00FFFF; margin-top: 8px;">🎯 Similarity: {song['similarity_score'] * 100:.1f}%</p>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            "🎵 Listen on YouTube Music",
                            key=f"youtube_{idx}",
                            use_container_width=True,
                        ):
                            webbrowser.open(song["youtube_url"], new=2)

    def run(self):
        """Run the main application using a clean, single-column layout."""
        logger.info("Starting SongFinderUI application")
        st.title("🎵 Song Finder")
        st.markdown(
            "<p style='text-align: center; font-size: 18px;'>Discover lyrics and find songs with a similar vibe.</p>",
            unsafe_allow_html=True,
        )

        self._display_search_section()
        self._display_lyrics_section()
        self._display_similar_songs_section()

        st.markdown(
            """
            <div class="footer">
                Made by Uyen and Roshan<br>
                Inspired by May
            </div>
        """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    app = SongFinderUI()
    app.run()
