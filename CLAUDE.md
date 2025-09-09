# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
- **Run the application**: `streamlit run streamlit_ui.py`
- **Install dependencies**: `pip install -r requirements.txt`
- **Create virtual environment**: `python -m venv venv && source venv/bin/activate` (On Windows: `venv\Scripts\activate`)

### Environment Setup
- **Required environment variables**: 
  - `GENIUS_ACCESS_TOKEN`: Your Genius API token (optional, used for lyrics.ovh API)
  - `TOKENIZERS_PARALLELISM`: Set to "false" to avoid warnings

## Architecture

### Core Components

**SongFinderUI** (`streamlit_ui.py`): Main application class handling the Streamlit interface, session state management, and coordinating between API and database layers.

**ARLLyricsAPI**: Handles communication with lyrics.ovh API for song suggestions and lyrics fetching with timeout handling and error management.

**ChromaDBManager**: Manages vector database operations including similarity searches using ChromaDB. Implements fuzzy logic comparison with RapidFuzz to filter out songs with overly similar lyrics (>70% similarity threshold).

### Data Processing Pipeline

The project uses Jupyter notebooks for data preparation and ChromaDB embedding:
- **Data preparation**: `notebooks/chroma_embedding_preparation.ipynb` - Processes song datasets and prepares them for embedding
- **ChromaDB setup**: Uses Ollama embedding function with `nomic-embed-text` model for vector embeddings
- **Dataset files**: Multiple pickle files (`embedding_model_dataset_*.pkl`) containing song metadata and cleaned lyrics

### Key Features

**Similarity Search Algorithm**:
- Vector similarity using ChromaDB embeddings
- RapidFuzz token_sort_ratio for lyrics comparison
- Automatic filtering to avoid duplicate content (songs with >70% lyrical similarity)
- Normalized similarity scores for ranking

**Session State Management**:
- Tracks selected songs, current lyrics, and similar song results
- Implements debounced API calls for search suggestions
- Maintains state across UI interactions

### Database Structure

**ChromaDB Collection**: `lyric_embeddings`
- **Documents**: Cleaned song lyrics (newlines removed, extra spaces normalized)
- **Metadata**: Song title, artist, year, views, features, tag, language
- **IDs**: Song ID converted to string
- **Embeddings**: Generated using Ollama's nomic-embed-text model

### Data Flow

1. User searches for song → ARLLyricsAPI fetches suggestions
2. User selects song → ARLLyricsAPI fetches lyrics
3. User requests similar songs → ChromaDBManager performs vector similarity search
4. Results filtered by lyrics similarity and ranked by vector distance
5. YouTube Music links generated for playback

The application persists ChromaDB data in `./chroma_db` directory and logs operations to `song_finder.log`.