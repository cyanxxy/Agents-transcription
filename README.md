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
- **Multiple Export Formats**: TXT, SRT, JSON with structured metadata
- **Progress Tracking**: Real-time updates during processing
- **Error Recovery**: Automatic retry with exponential backoff
- **Async Processing**: Non-blocking operations with background task management
- **State Optimization**: Intelligent state management to reduce unnecessary reruns
- **Structured Error Handling**: Comprehensive error categorization and recovery

### 🛠️ Smart Editing Tools
- Find & replace with regex support
- Auto-formatting (capitalization, punctuation, filler words)
- Quality check with issue detection
- Edit history with undo/redo
- Context-aware suggestions

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
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

   The app will be available at `http://localhost:8501`

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

### Modern Agent System with Async Support
```
┌─────────────────────┐    ┌──────────────────┐
│ WorkflowCoordinator │    │   AsyncHandler   │
└──────────┬──────────┘    └────────┬─────────┘
           │                        │
    ┌──────┴──────┐          ┌──────┴──────┐
    │   Supervisor │◄─────────┤ErrorHandler │
    └──────┬──────┘          └─────────────┘
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
                                                      │
              ┌───────────────────────────────────────┘
              │
        ┌─────▼──────┐
        │StateOptimizer│
        └────────────┘
```

### Core Components
- **AsyncHandler**: Manages non-blocking async operations without event loop conflicts
- **ErrorHandler**: Provides structured error handling with automatic retry and recovery
- **StateOptimizer**: Intelligent state management with batching to minimize reruns
- **Agent Supervisor**: Coordinates message passing between specialized agents

### Message Flow
1. User uploads audio → FileProcessingAgent validates and chunks (with async validation)
2. TranscriptionAgent processes chunks with Gemini API (with retry logic)
3. Background tasks handle long operations without blocking UI
4. QualityAssuranceAgent analyzes transcript quality
5. EditingAssistantAgent provides smart editing with error recovery
6. ExportAgent handles format conversion
7. StateOptimizer batches updates to prevent unnecessary reruns

## 📊 Quality Metrics

The app provides comprehensive quality analysis:
- **Readability Score**: Based on sentence length and structure
- **Sentence Variety**: Measures variation in sentence patterns
- **Vocabulary Richness**: Analyzes word diversity
- **Punctuation Density**: Checks proper punctuation usage
- **Timestamp Coverage**: For timestamped transcripts

## 🔧 Advanced Features

### Async Architecture
- **Non-blocking Operations**: Background task management prevents UI freezing
- **Event Loop Management**: Proper async/await handling without conflicts
- **Progress Tracking**: Real-time updates during long-running operations

### Intelligent Error Handling
- **Structured Errors**: Categorized error types (API, Network, File, etc.)
- **Automatic Retry**: Exponential backoff with configurable retry limits
- **Recovery Strategies**: Context-aware error recovery based on error type
- **Error Persistence**: Failed operations can be resumed

### State Optimization
- **Batch Updates**: Multiple state changes grouped to reduce reruns
- **Smart Caching**: Computation results cached with TTL and invalidation
- **State Persistence**: User preferences saved across sessions
- **Minimal Reruns**: Intelligent checks prevent unnecessary page reloads

### Smart Editing
- **Auto-format**: Fix capitalization, punctuation, remove filler words
- **Find & Replace**: Support for regex and case-sensitive search
- **Quality Check**: Detect repeated words, missing punctuation, inconsistencies
- **Edit History**: Track changes with undo/redo capabilities

### Export Options
- **TXT**: Plain text with speaker labels and timestamps
- **SRT**: Subtitle format with precise timing for video playback
- **JSON**: Structured data with metadata, quality metrics, and processing info

## 🛠️ Technical Architecture

### Utility Modules
- **`utils/async_handler.py`**: Manages async operations in Streamlit without blocking
- **`utils/error_handler.py`**: Provides structured error handling with retry mechanisms  
- **`utils/state_optimizer.py`**: Optimizes state management to reduce reruns

### Key Improvements (Latest Version)
1. **Async/Sync Fix**: Replaced `asyncio.run()` with proper async handlers
2. **Error Recovery**: Added comprehensive error categorization and recovery
3. **Performance**: Optimized state management with intelligent batching
4. **Background Tasks**: Long operations run in background with progress callbacks
5. **Retry Logic**: Automatic retry with exponential backoff for API failures

### Dependencies
- **Core**: Streamlit, Google GenerativeAI, PyDub
- **Audio**: FFmpeg (system dependency)
- **Development**: pytest, ruff, black, mypy (optional)

## 🐛 Troubleshooting

### Common Issues

1. **"API key not found"**
   - Ensure GOOGLE_API_KEY is set in secrets.toml or environment
   - Check that secrets.toml file exists in .streamlit/ directory

2. **"File too large"**
   - Maximum file size is 200MB
   - Large files are automatically split into 2-minute chunks

3. **"FFmpeg not found"**
   - Install FFmpeg and ensure it's in your PATH
   - Verify installation: `ffmpeg -version`

4. **"Event loop errors"**
   - The app now uses proper async handling to prevent these issues
   - If encountered, try refreshing the page

5. **"Frequent page reloads"**
   - The new state optimizer reduces unnecessary reruns
   - Check browser console for any JavaScript errors

6. **"Processing stuck"**
   - Background task management handles long operations
   - Check the progress indicators and wait for completion

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