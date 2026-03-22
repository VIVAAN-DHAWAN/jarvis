#!/usr/bin/env python3
"""
Jarvis - A Voice-Controlled AI Assistant for macOS

Features:
- Wake word detection using SpeechRecognition (free, no API key needed)
- Audio recording after wake word
- Local transcription using OpenAI Whisper
- Multi-provider AI API support (OpenAI, Anthropic, OpenRouter, Google, NVIDIA)
- Text-to-speech using macOS `say` command
- Mac control: open apps, search web, volume, type, click, close windows
"""

import os
import sys
import json
import queue
import threading
import subprocess
import tempfile
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime

# Audio handling
import pyaudio
import wave
import speech_recognition as sr
import time

# Whisper for transcription
import whisper

# Mac control
import pyautogui
import mss

# HTTP for API calls
import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Configuration for Jarvis."""
    # Wake word settings
    wake_word: str = "jarvis"
    porcupine_model_path: Optional[str] = None
    porcupine_keyword_path: Optional[str] = None
    porcupine_access_key: str = ""

    # Audio settings
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_format: int = pyaudio.paInt16
    audio_chunk_size: int = 512

    # Recording settings
    recording_threshold: float = 0.01
    recording_timeout: float = 30.0
    silence_duration: float = 2.0

    # Whisper settings
    whisper_model: str = "small"  # tiny, base, small, medium, large
    whisper_language: Optional[str] = None

    # AI Provider settings
    ai_provider: str = "openai"  # openai, anthropic, openrouter, google, nvidia
    ai_model: str = "gpt-4o-mini"
    ai_api_key: str = ""
    ai_base_url: Optional[str] = None  # Override base URL (e.g. for OpenRouter)
    ai_temperature: float = 0.7
    ai_max_tokens: int = 500

    # TTS settings
    tts_voice: str = "Samantha"  # macOS voice
    tts_rate: int = 180

    # AI behavior
    system_prompt: str = (
        "You are Jarvis, a voice-controlled AI assistant for macOS. "
        "Help the user control their Mac. Keep responses short and clear."
    )

    # System settings
    log_level: str = "INFO"


# Provider configs
AI_PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "chat_endpoint": "/chat/completions",
        "default_model": "gpt-4o-mini",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "chat_endpoint": "/v1/messages",
        "default_model": "claude-3-haiku-20240307",
        "auth_header": "x-api-key",
        "auth_prefix": "",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "chat_endpoint": "/chat/completions",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "chat_endpoint": "/models/{model}:generateContent",
        "default_model": "gemini-2.5-flash",
        "auth_header": "",
        "auth_prefix": "",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "chat_endpoint": "/chat/completions",
        "default_model": "meta/llama-3.1-8b-instruct",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
}


# =============================================================================
# LOGGER
# =============================================================================

class Logger:
    """Simple logger for Jarvis."""

    def __init__(self, name: str = "Jarvis", level: str = "INFO"):
        self.name = name
        self.log_levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        self.current_level = self.log_levels.get(level.upper(), 1)

    def _log(self, level_name: str, message: str):
        if self.log_levels.get(level_name.upper(), 0) >= self.current_level:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{self.name}] [{level_name.upper()}] {message}")

    def debug(self, msg): self._log("DEBUG", msg)
    def info(self, msg): self._log("INFO", msg)
    def warning(self, msg): self._log("WARNING", msg)
    def error(self, msg): self._log("ERROR", msg)


# =============================================================================
# UTILITIES
# =============================================================================

def speak(text: str, config: Config):
    """Uses macOS `say` command to speak text."""
    subprocess.run(["say", "-v", config.tts_voice, "--rate", str(config.tts_rate), text])


def record_audio(config: Config, audio_queue: queue.Queue, stop_event: threading.Event, logger: Logger):
    """Listens for wake word using SpeechRecognition and puts audio in queue."""
    r = sr.Recognizer()
    source = sr.Microphone(sample_rate=config.audio_sample_rate)

    logger.info("Calibrating microphone for ambient noise...")
    with source as src:
        r.adjust_for_ambient_noise(src, duration=1)
    logger.info(f"Listening for wake word '{config.wake_word}' via SpeechRecognition...")

    def callback(recognizer, audio):
        try:
            text = recognizer.recognize_google(audio)
            logger.debug(f"Heard: {text}")
            if config.wake_word.lower() in text.lower():
                logger.info(f"Wake word '{config.wake_word}' detected in: '{text}'")
                audio_queue.put(audio.get_raw_data())
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            logger.error(f"SpeechRecognition service error: {e}")

    stop_listening = r.listen_in_background(source, callback, phrase_time_limit=config.recording_timeout)

    while not stop_event.is_set():
        time.sleep(0.1)

    stop_listening()
    logger.info("Stopped audio listening.")


def transcribe_audio(config: Config, audio_queue: queue.Queue, command_queue: queue.Queue, logger: Logger):
    """Transcribes audio from the queue using OpenAI Whisper."""
    model = whisper.load_model(config.whisper_model)
    logger.info(f"Whisper model '{config.whisper_model}' loaded.")

    while True:
        audio_data = audio_queue.get()
        if audio_data is None:
            break

        logger.info("Transcribing audio...")
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            wav_file = wave.open(tmp_path, 'wb')
            wav_file.setnchannels(config.audio_channels)
            pa = pyaudio.PyAudio()
            try:
                sample_width = pa.get_sample_size(config.audio_format)
            finally:
                pa.terminate()
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(config.audio_sample_rate)
            wav_file.writeframes(audio_data)
            wav_file.close()

            result = model.transcribe(tmp_path, language=config.whisper_language, fp16=False)
            os.unlink(tmp_path)

            command = result["text"].strip()
            logger.info(f"Transcribed: '{command}'")
            command_queue.put(command)

        except Exception as e:
            logger.error(f"Error during transcription: {e}")
        finally:
            audio_queue.task_done()


def get_ai_response(config: Config, command: str, logger: Logger) -> Optional[str]:
    """Sends the transcribed command to the configured AI API and returns the response."""
    provider_config = AI_PROVIDERS.get(config.ai_provider)
    if not provider_config:
        logger.error(f"Unknown AI provider: {config.ai_provider}")
        return None

    headers = {"Content-Type": "application/json"}
    if config.ai_api_key:
        auth_header = provider_config["auth_header"]
        if auth_header:
            headers[auth_header] = f"{provider_config['auth_prefix']}{config.ai_api_key}"

    # Use config base_url override if set (e.g. OpenRouter via openai provider)
    base_url = config.ai_base_url if config.ai_base_url else provider_config["base_url"]
    endpoint = provider_config["chat_endpoint"]
    model = config.ai_model or provider_config["default_model"]

    payload: Dict[str, Any] = {}

    if config.ai_provider in ("openai", "openrouter", "nvidia"):
        messages = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        messages.append({"role": "user", "content": command})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": config.ai_temperature,
            "max_tokens": config.ai_max_tokens,
        }
    elif config.ai_provider == "anthropic":
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": command}],
            "max_tokens": config.ai_max_tokens,
            "temperature": config.ai_temperature,
        }
        if config.system_prompt:
            payload["system"] = config.system_prompt
    elif config.ai_provider == "google":
        endpoint = endpoint.format(model=model)
        payload = {
            "contents": [{"parts": [{"text": command}]}],
            "generationConfig": {
                "temperature": config.ai_temperature,
                "maxOutputTokens": config.ai_max_tokens,
            },
        }
    else:
        logger.error(f"Unsupported AI provider: {config.ai_provider}")
        return None

    try:
        url = f"{base_url}{endpoint}"
        request_kwargs = {"headers": headers, "json": payload, "timeout": 15}
        if config.ai_provider == "google" and config.ai_api_key:
            request_kwargs["params"] = {"key": config.ai_api_key}

        response = requests.post(url, **request_kwargs)
        response.raise_for_status()
        response_json = response.json()

        if config.ai_provider in ("openai", "openrouter", "nvidia"):
            content = response_json["choices"][0]["message"]["content"]
            return content.strip() if content else None
        elif config.ai_provider == "anthropic":
            return response_json["content"][0]["text"].strip()
        elif config.ai_provider == "google":
            return response_json["candidates"][0]["content"]["parts"][0]["text"].strip()

    except requests.exceptions.RequestException as e:
        logger.error(f"AI API request failed: {e}")
    except (KeyError, TypeError) as e:
        logger.error(f"Unexpected AI API response format: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during AI API call: {e}")

    return None


def handle_command(config: Config, command: str, logger: Logger, stop_event: threading.Event) -> str:
    """Handles the transcribed command — runs direct Mac actions or falls back to AI."""
    command_lower = command.lower()

    # Stop command
    if "jarvis stop" in command_lower:
        logger.info("Stop command received.")
        stop_event.set()
        return "Stopping Jarvis. Goodbye!"

    # Strip leading wake word
    wake_word_lower = config.wake_word.lower()
    if command_lower.startswith(wake_word_lower):
        command = command[len(wake_word_lower):].strip()
        command_lower = command.lower()
    else:
        idx = command_lower.find(wake_word_lower)
        if idx != -1:
            command = command[idx + len(wake_word_lower):].strip()
            command_lower = command.lower()

    # --- Direct Mac Commands ---

    # Open apps (fuzzy keyword matching)
    if "open" in command_lower or any(app in command_lower for app in ["chrome", "safari", "youtube", "music", "spotify", "finder", "terminal"]):
        if "chrome" in command_lower:
            subprocess.run(["open", "-a", "Google Chrome"])
            return "Opening Chrome."
        elif "safari" in command_lower:
            subprocess.run(["open", "-a", "Safari"])
            return "Opening Safari."
        elif "youtube" in command_lower:
            subprocess.run(["open", "https://www.youtube.com"])
            return "Opening YouTube."
        elif "music" in command_lower or "spotify" in command_lower:
            app = "Spotify" if "spotify" in command_lower else "Music"
            subprocess.run(["open", "-a", app])
            return f"Opening {app}."
        elif "finder" in command_lower:
            subprocess.run(["open", "-a", "Finder"])
            return "Opening Finder."
        elif "terminal" in command_lower:
            subprocess.run(["open", "-a", "Terminal"])
            return "Opening Terminal."
        elif command_lower.startswith("open "):
            app_name = command[5:].strip()
            subprocess.run(["open", "-a", app_name])
            return f"Opening {app_name}."

    # Web search
    if "search" in command_lower:
        query = command_lower.replace("search", "").replace("for", "").strip()
        if query:
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            subprocess.run(["open", url])
            return f"Searching for {query}."

    # Volume control
    if "volume up" in command_lower:
        subprocess.run(["osascript", "-e", "set volume output volume ((output volume of (get volume settings)) + 10)"])
        return "Volume up."
    if "volume down" in command_lower:
        subprocess.run(["osascript", "-e", "set volume output volume ((output volume of (get volume settings)) - 10)"])
        return "Volume down."

    # Close window
    if "close" in command_lower and ("window" in command_lower or "this" in command_lower):
        pyautogui.hotkey('command', 'w')
        return "Closed the window."

    # Type text
    if "type" in command_lower:
        text_to_type = command.split("type", 1)[-1].strip()
        if text_to_type:
            pyautogui.write(text_to_type)
            return f"Typed: {text_to_type}"

    # Screenshot
    if "screenshot" in command_lower:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = str(Path.home() / f"screenshot_{timestamp}.png")
        with mss.mss() as sct:
            sct.shot(output=path)
        return f"Screenshot saved to {path}"

    # Click
    if "click" in command_lower:
        pyautogui.click()
        return "Clicked."

    # --- Fallback to AI ---
    logger.info("Sending command to AI for response...")
    ai_response = get_ai_response(config, command, logger)
    return ai_response if ai_response else "Sorry, I couldn't process that. Please try again."


def main():
    parser = argparse.ArgumentParser(description="Jarvis - Voice-Controlled AI Assistant for macOS")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    args = parser.parse_args()

    logger = Logger(level="INFO")
    logger.info("Jarvis is starting up...")

    config_path = Path(args.config)
    config = Config()

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            # Filter to only known fields
            known_fields = {f.name for f in config.__dataclass_fields__.values()}
            config_dict = {k: v for k, v in config_dict.items() if k in known_fields}
            config = Config(**config_dict)
            logger.current_level = logger.log_levels.get(config.log_level.upper(), 1)
        except json.decoder.JSONDecodeError:
            logger.warning(f"Malformed config file. Using defaults.")
    else:
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=4)
        logger.info(f"Created default config at {config_path}. Edit it with your API keys.")
        sys.exit(0)

    logger = Logger(level=config.log_level)
    logger.info("Jarvis is starting up...")
    logger.info("Wake word detection using SpeechRecognition (no Porcupine).")

    audio_queue = queue.Queue()
    command_queue = queue.Queue()
    stop_event = threading.Event()

    audio_thread = threading.Thread(
        target=record_audio,
        args=(config, audio_queue, stop_event, logger),
        daemon=True
    )
    transcription_thread = threading.Thread(
        target=transcribe_audio,
        args=(config, audio_queue, command_queue, logger),
        daemon=True
    )

    audio_thread.start()
    transcription_thread.start()

    try:
        while not stop_event.is_set():
            command = None
            try:
                command = command_queue.get(timeout=1)
                if command:
                    logger.info(f"Processing command: '{command}'")
                    response = handle_command(config, command, logger, stop_event)
                    if response:
                        logger.info("Speaking response...")
                        speak(response, config)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing command: {e}")
            finally:
                if command is not None:
                    command_queue.task_done()

    except KeyboardInterrupt:
        logger.info("Jarvis stopped by user (Ctrl+C).")
    finally:
        logger.info("Shutting down Jarvis...")
        stop_event.set()
        audio_queue.put(None)
        audio_thread.join(timeout=3)
        transcription_thread.join(timeout=3)
        logger.info("Jarvis shutdown complete.")


if __name__ == "__main__":
    main()
