# 🎙️ ExactTranscriber

**Precision audio transcription powered by Google's Gemini AI with intelligent agent-based architecture**

ExactTranscriber is a modern, agent-powered audio transcription application that leverages Google's Gemini 2.5 Flash multimodal models to deliver high-quality, accurate transcriptions. Built with a sophisticated agent architecture, it offers smart editing, quality assurance, and multiple export formats.

## ✨ Key Features

### 🤖 Agent-Based Architecture
- **FileProcessingAgent**: Handles audio validation, chunking, and preprocessing
- **TranscriptionAgent**: Orchestrates Gemini API calls with caching and retry logic
- **QualityAssuranceAgent**: Analyzes transcript quality and suggests improvements
- **EditingAssistantAgent**: Provides smart search, replace, and formatting tools
- **ExportAgent**: Supports multiple export formats with customization options

### 🎯 Core Capabilities
- **High-Accuracy Transcription**: Powered by Gemini 2.5 Flash multimodal models
- **Smart Chunking**: Automatically splits large files for optimal processing
- **Quality Scoring**: Real-time transcript quality assessment (0-100 score)
- **Intelligent Editing**: AI-powered suggestions and auto-formatting
- **Multiple Export Formats**: TXT, SRT, VTT, JSON, XML, CSV, and more
- **Progress Tracking**: Real-time updates during processing
- **Error Recovery**: Automatic retry with exponential backoff

### 🛠️ Smart Editing Tools
- Find & replace with regex support
- Auto-formatting (capitalization, punctuation, filler words)
- Quality check with issue detection
- Edit history with undo/redo
- Context-aware suggestions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- FFmpeg (for audio processing)
- Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ExactTranscriber.git
   cd ExactTranscriber
   ```

2. **Install dependencies**
   ```bash
   pip install -e .
   ```

3. **Install FFmpeg**
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

4. **Configure secrets**
   
   Copy the example secrets file and update with your values:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   
   Edit `.streamlit/secrets.toml`:
   ```toml
   # App password
   app_password = "your-secure-password"
   
   # Google API Key
   GOOGLE_API_KEY = "your-gemini-api-key"
   ```
   
   **⚠️ Security Note**: Never commit `secrets.toml` to version control. The `.gitignore` file is already configured to exclude it.

5. **Run the application**
   ```bash
   streamlit run main.py
   ```

   The app will be available at `http://localhost:5000`

## 📋 Configuration

### Available Models
- **Gemini 2.5 Flash**: Fast, high-quality transcription
- **Gemini 2.5 Flash Lite**: Lighter model for faster processing

### Environment Variables
- `GOOGLE_API_KEY`: Your Gemini API key (alternative to secrets.toml)
- `APP_PASSWORD`: Application password (alternative to secrets.toml)

## 🎯 Usage

1. **Login**: Enter the password configured in secrets.toml
2. **Upload Audio**: Select your audio file (supports MP3, WAV, M4A, FLAC, OGG, MP4, WEBM)
3. **Configure Settings**:
   - Select transcription model
   - Set number of speakers
   - Add optional context (topic, description)
4. **Transcribe**: Click "Transcribe Audio" and monitor progress
5. **Review Quality**: Check the quality score and metrics
6. **Edit**: Use smart editing tools to refine the transcript
7. **Export**: Choose format and download

## 🏗️ Architecture

### Agent System
```
┌─────────────────────┐
│ WorkflowCoordinator │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │   Supervisor │
    └──────┬──────┘
           │
    ┌──────┴────────────────────────────┐
    │                                   │
┌───▼────────┐  ┌──────────────┐  ┌───▼──────────┐
│FileProcessor│  │ Transcriber  │  │QualityAssure │
└────────────┘  └──────────────┘  └──────────────┘
                                           │
                        ┌──────────────────┴─────────────┐
                        │                                │
                  ┌─────▼──────┐              ┌─────────▼──┐
                  │EditAssistant│              │  Exporter  │
                  └─────────────┘              └────────────┘
```

### Message Flow
1. User uploads audio → FileProcessingAgent validates and chunks
2. TranscriptionAgent processes chunks with Gemini API
3. QualityAssuranceAgent analyzes transcript quality
4. EditingAssistantAgent provides editing capabilities
5. ExportAgent handles format conversion

## 📊 Quality Metrics

The app provides comprehensive quality analysis:
- **Readability Score**: Based on sentence length and structure
- **Sentence Variety**: Measures variation in sentence patterns
- **Vocabulary Richness**: Analyzes word diversity
- **Punctuation Density**: Checks proper punctuation usage
- **Timestamp Coverage**: For timestamped transcripts

## 🔧 Advanced Features

### Batch Processing
Process multiple audio files simultaneously with progress tracking for each file.

### Smart Editing
- **Auto-format**: Fix capitalization, punctuation, remove filler words
- **Find & Replace**: Support for regex and case-sensitive search
- **Quality Check**: Detect repeated words, missing punctuation, inconsistencies

### Export Options
- **TXT**: Plain text with optional formatting
- **SRT/VTT**: Subtitle formats with timing
- **JSON**: Structured data with metadata
- **CSV**: Tabular format for analysis

## 🐛 Troubleshooting

### Common Issues

1. **"API key not found"**
   - Ensure GOOGLE_API_KEY is set in secrets.toml or environment

2. **"File too large"**
   - Maximum file size is 200MB
   - Large files are automatically chunked

3. **"FFmpeg not found"**
   - Install FFmpeg and ensure it's in your PATH

4. **Import errors**
   - Run `pip install -r requirements.txt` to install all dependencies

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini team for the powerful multimodal AI
- Streamlit for the excellent web framework
- The open-source community for various dependencies

---

**Note**: This application requires a valid Google Gemini API key. Get yours at [Google AI Studio](https://makersuite.google.com/app/apikey).