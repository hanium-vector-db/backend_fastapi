import os
import io
import tempfile
import logging
from typing import Optional, Dict, Any
import speech_recognition as sr
from gtts import gTTS
import whisper
from pydub import AudioSegment
# from pydub.playback import play  # 플레이백 기능 제외
import numpy as np

logger = logging.getLogger(__name__)

class SpeechService:
    """음성 인식(STT)과 음성 합성(TTS) 서비스"""

    def __init__(self):
        # Whisper 모델 로드 (더 정확한 음성 인식)
        try:
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper 모델 로드 완료")
        except Exception as e:
            logger.error(f"Whisper 모델 로드 실패: {e}")
            self.whisper_model = None

        # SpeechRecognition 초기화 (fallback)
        self.recognizer = sr.Recognizer()

        # 마이크 초기화 (PyAudio 없이는 제한적)
        try:
            self.microphone = sr.Microphone()
            # 노이즈 조정
            try:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.info("마이크 노이즈 조정 완료")
            except Exception as e:
                logger.warning(f"마이크 노이즈 조정 경고: {e}")
        except Exception as e:
            logger.warning(f"마이크 초기화 실패 (PyAudio 없음): {e}")
            self.microphone = None

    def speech_to_text_whisper(self, audio_file_path: str) -> Dict[str, Any]:
        """Whisper를 사용한 음성 인식 (더 정확함)"""
        try:
            if not self.whisper_model:
                raise Exception("Whisper 모델이 로드되지 않았습니다")

            # Whisper로 음성 인식
            result = self.whisper_model.transcribe(audio_file_path)

            return {
                "success": True,
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "confidence": 1.0,  # Whisper는 confidence 점수를 직접 제공하지 않음
                "method": "whisper"
            }

        except Exception as e:
            logger.error(f"Whisper 음성 인식 실패: {e}")
            return {
                "success": False,
                "text": "",
                "error": str(e),
                "method": "whisper"
            }

    def speech_to_text_google(self, audio_file_path: str) -> Dict[str, Any]:
        """Google Speech Recognition을 사용한 음성 인식 (fallback)"""
        try:
            # 오디오 파일 로드
            with sr.AudioFile(audio_file_path) as source:
                audio = self.recognizer.record(source)

            # Google Speech Recognition으로 변환
            text = self.recognizer.recognize_google(audio, language='ko-KR')

            return {
                "success": True,
                "text": text,
                "language": "ko",
                "confidence": 0.8,  # 추정값
                "method": "google"
            }

        except sr.UnknownValueError:
            return {
                "success": False,
                "text": "",
                "error": "음성을 인식할 수 없습니다",
                "method": "google"
            }
        except sr.RequestError as e:
            return {
                "success": False,
                "text": "",
                "error": f"Google Speech Recognition 서비스 오류: {e}",
                "method": "google"
            }
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "error": str(e),
                "method": "google"
            }

    def speech_to_text(self, audio_file_path: str, prefer_whisper: bool = True) -> Dict[str, Any]:
        """음성을 텍스트로 변환 (Whisper 우선, Google fallback)"""

        if prefer_whisper and self.whisper_model:
            result = self.speech_to_text_whisper(audio_file_path)
            if result["success"]:
                return result

            # Whisper 실패 시 Google로 fallback
            logger.warning("Whisper 실패, Google Speech Recognition으로 fallback")

        # Google Speech Recognition 사용
        return self.speech_to_text_google(audio_file_path)

    def record_audio_from_microphone(self, duration: int = 5) -> Optional[str]:
        """마이크에서 오디오 녹음"""
        try:
            with self.microphone as source:
                logger.info(f"{duration}초 동안 오디오 녹음 시작...")
                audio = self.recognizer.listen(source, timeout=duration, phrase_time_limit=duration)

            # 임시 파일로 저장
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            with open(temp_file.name, "wb") as f:
                f.write(audio.get_wav_data())

            logger.info(f"오디오 녹음 완료: {temp_file.name}")
            return temp_file.name

        except Exception as e:
            logger.error(f"오디오 녹음 실패: {e}")
            return None

    def text_to_speech(self, text: str, language: str = 'ko', slow: bool = False) -> Dict[str, Any]:
        """텍스트를 음성으로 변환"""
        try:
            if not text.strip():
                return {
                    "success": False,
                    "audio_file": None,
                    "error": "텍스트가 비어있습니다"
                }

            # gTTS로 음성 합성
            tts = gTTS(text=text, lang=language, slow=slow)

            # 임시 파일로 저장
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_file.name)

            logger.info(f"음성 합성 완료: {temp_file.name}")

            return {
                "success": True,
                "audio_file": temp_file.name,
                "text": text,
                "language": language,
                "duration_estimate": len(text) * 0.1  # 대략적인 재생 시간 추정
            }

        except Exception as e:
            logger.error(f"음성 합성 실패: {e}")
            return {
                "success": False,
                "audio_file": None,
                "error": str(e)
            }

    def convert_audio_format(self, input_file: str, output_format: str = "wav") -> str:
        """오디오 파일 포맷 변환"""
        try:
            audio = AudioSegment.from_file(input_file)

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}")
            audio.export(temp_file.name, format=output_format)

            return temp_file.name

        except Exception as e:
            logger.error(f"오디오 포맷 변환 실패: {e}")
            return input_file

    def cleanup_temp_file(self, file_path: str):
        """임시 파일 정리"""
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"임시 파일 삭제: {file_path}")
        except Exception as e:
            logger.warning(f"임시 파일 삭제 실패: {e}")

    def get_supported_languages(self) -> Dict[str, str]:
        """지원되는 언어 목록"""
        return {
            'ko': '한국어',
            'en': 'English',
            'ja': '日本語',
            'zh': '中文',
            'es': 'Español',
            'fr': 'Français',
            'de': 'Deutsch',
            'it': 'Italiano',
            'pt': 'Português',
            'ru': 'Русский'
        }