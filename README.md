# UNDER DEVELOPMENT
# 🤖 Jarvis — Voice-Controlled AI Assistant for macOS

A free, local, voice-controlled AI assistant for macOS. Say **"Jarvis"** and give a command — it listens, understands, and controls your Mac.

## Features

- 🎙️ **Wake word detection** — say "Jarvis" to activate (free, no paid API)
- 🧠 **Local transcription** — OpenAI Whisper runs fully offline on your Mac
- 🌐 **Multi-provider AI** — OpenAI, Anthropic, OpenRouter (free models), Google Gemini, NVIDIA
- 🖥️ **Mac control** — open apps, search web, type, click, volume, close windows
- 🔊 **macOS TTS** — speaks back using the built-in `say` command

## Voice Commands

| Say | What happens |
|-----|-------------|
| "Jarvis open Chrome" | Opens Google Chrome |
| "Jarvis open Safari" | Opens Safari |
| "Jarvis open YouTube" | Opens youtube.com |
| "Jarvis search for lofi music" | Google search |
| "Jarvis volume up" | Volume +10% |
| "Jarvis volume down" | Volume -10% |
| "Jarvis close window" | Closes current window |
| "Jarvis type hello world" | Types text |
| "Jarvis screenshot" | Takes a screenshot |
| "Jarvis click" | Clicks mouse |
| "Jarvis stop" | Shuts down Jarvis |
| Anything else | Sent to AI for a response |

## Setup

### 1. Install dependencies

```bash
pip install openai-whisper speechrecognition pyaudio pyautogui mss requests sounddevice numpy
```

### 2. Configure

On first run, Jarvis creates a `config.json`. Edit it with your AI provider details:

```json
{
  "ai_provider": "openai",
  "ai_model": "gpt-4o-mini",
  "ai_api_key": "your-key-here"
}
```

#### Free option — OpenRouter

Sign up at [openrouter.ai](https://openrouter.ai) for a free key and free models:

```json
{
  "ai_provider": "openai",
  "ai_base_url": "https://openrouter.ai/api/v1",
  "ai_model": "meta-llama/llama-3.3-70b-instruct:free",
  "ai_api_key": "sk-or-v1-..."
}
```

#### Other providers

| Provider | `ai_provider` | Example model |
|----------|--------------|---------------|
| OpenAI | `openai` | `gpt-4o-mini` |
| Anthropic | `anthropic` | `claude-3-haiku-20240307` |
| Google Gemini | `google` | `gemini-2.5-flash` |
| NVIDIA | `nvidia` | `meta/llama-3.1-8b-instruct` |
| OpenRouter | `openai` + `ai_base_url` | any free model |

### 3. Run

```bash
python jarvis.py
```

Grant microphone permission when macOS asks. Then say **"Jarvis"** followed by your command.

## Config Reference

| Field | Default | Description |
|-------|---------|-------------|
| `wake_word` | `jarvis` | Word to activate Jarvis |
| `whisper_model` | `small` | Whisper model size (tiny/base/small/medium/large) |
| `ai_provider` | `openai` | AI provider to use |
| `ai_model` | `gpt-4o-mini` | Model name |
| `ai_api_key` | `""` | Your API key |
| `ai_base_url` | `null` | Override base URL (for OpenRouter etc.) |
| `tts_voice` | `Samantha` | macOS voice |
| `tts_rate` | `180` | Speech rate (words per minute) |

## Requirements

- macOS (uses `say` command for TTS)
- Python 3.10+
- Microphone access

## Built with

- [OpenAI Whisper](https://github.com/openai/whisper) — local speech-to-text
- [SpeechRecognition](https://github.com/Uberi/speech_recognition) — wake word detection
- [PyAutoGUI](https://github.com/asweigart/pyautogui) — Mac control
- [mss](https://github.com/BoboTiG/python-mss) — screenshots

## License

MIT
