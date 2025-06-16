# 🎵 Song Finder

A modern, Gen Z-themed Streamlit application that helps users discover songs, fetch lyrics, and find similar tracks based on lyrical content. Built with Python, Streamlit, and ChromaDB.

## ✨ Features

- **Modern UI**: Vibrant, Gen Z-inspired interface with neon gradients and smooth animations
- **Song Search**: Real-time search with suggestions powered by lyrics.ovh API
- **Lyrics Display**: Clean, readable lyrics presentation with YouTube Music integration
- **Similar Song Discovery**: Find similar songs using:
  - Vector similarity search via ChromaDB
  - Fuzzy logic comparison using RapidFuzz
  - Smart filtering to avoid duplicate content
- **YouTube Integration**: Direct links to listen to songs on YouTube Music
- **Responsive Design**: Works seamlessly on both desktop and mobile devices

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/roshan-shaik-ml/music-recommender-system.git
   cd music-recommender-system
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_ui.py
   ```

## 📋 Requirements

- Python 3.8+
- Streamlit
- ChromaDB
- RapidFuzz
- lyricsgenius
- python-dotenv
- requests

## 🛠️ Technical Details

### Architecture

The application is built using a modular architecture:

1. **UI Layer** (`SongFinderUI` class):
   - Handles user interface and interactions
   - Manages session state
   - Coordinates between different components

2. **API Layer** (`ARLLyricsAPI` class):
   - Interfaces with lyrics.ovh API
   - Handles song suggestions and lyrics fetching
   - Implements error handling and timeouts

3. **Database Layer** (`ChromaDBManager` class):
   - Manages vector database operations
   - Handles similarity searches
   - Implements fuzzy logic comparison

### Key Components

1. **Song Search**:
   - Real-time suggestions as you type
   - Debounced API calls to prevent rate limiting
   - Clean presentation of results

2. **Lyrics Processing**:
   - Automatic cleaning and formatting
   - Error handling for missing lyrics
   - Read-only display with copy functionality

3. **Similarity Search**:
   - Vector similarity using ChromaDB
   - Fuzzy logic comparison using RapidFuzz
   - Smart filtering to avoid duplicates
   - Normalized similarity scores

4. **UI/UX Features**:
   - Modern, responsive design
   - Smooth animations and transitions
   - Clear visual hierarchy
   - Intuitive navigation

## 🔧 Configuration

### Environment Variables

- `GENIUS_ACCESS_TOKEN`: Your Genius API token
- `TOKENIZERS_PARALLELISM`: Set to "false" to avoid warnings

### Customization

The application's appearance can be customized by modifying the CSS in `streamlit_ui.py`:
- Color schemes
- Animations
- Layout
- Typography

## 📝 Logging

The application includes comprehensive logging:
- File logging to `song_finder.log`
- Console output
- Debug information for similarity calculations
- Error tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Credits

- Made by Uyen and Roshan
- Inspired by May

## 🐛 Known Issues

- API timeouts may occur with slow internet connections
- Some songs may not have available lyrics
- Similarity scores may vary based on lyrics quality

## 🔮 Future Improvements

- Add more music platforms integration
- Implement caching for better performance
- Add user preferences and history
- Enhance similarity algorithms
- Add more visualization options

## 📞 Support

For support, please open an issue in the repository or contact the maintainers.

---

Made with ❤️ by Uyen and Roshan 
