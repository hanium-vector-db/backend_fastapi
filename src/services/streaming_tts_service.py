import asyncio
import json
import re
import tempfile
import time
import logging
from typing import AsyncGenerator, Dict, Any, List
from queue import Queue
import threading
from services.speech_service import SpeechService

logger = logging.getLogger(__name__)

class StreamingTTSService:
    """스트리밍 텍스트 생성과 실시간 음성 합성을 결합한 서비스"""

    def __init__(self, speech_service: SpeechService):
        self.speech_service = speech_service
        self.sentence_buffer = ""
        self.sentence_queue = Queue()

        # 문장 구분자들 (한국어, 영어 등)
        self.sentence_endings = ['.', '!', '?', '。', '！', '？', '\n']

    def extract_complete_sentences(self, text_chunk: str) -> List[str]:
        """텍스트 청크에서 완성된 문장들을 추출"""
        self.sentence_buffer += text_chunk

        sentences = []
        current_pos = 0

        while current_pos < len(self.sentence_buffer):
            # 문장 끝 찾기
            sentence_end = -1
            for ending in self.sentence_endings:
                pos = self.sentence_buffer.find(ending, current_pos)
                if pos != -1:
                    if sentence_end == -1 or pos < sentence_end:
                        sentence_end = pos

            if sentence_end != -1:
                # 완성된 문장 추출
                sentence = self.sentence_buffer[current_pos:sentence_end + 1].strip()
                if sentence:
                    sentences.append(sentence)
                current_pos = sentence_end + 1
            else:
                # 완성되지 않은 문장은 버퍼에 남김
                break

        # 처리된 부분을 버퍼에서 제거
        self.sentence_buffer = self.sentence_buffer[current_pos:]

        return sentences

    def should_speak_partial_sentence(self, partial_text: str) -> bool:
        """부분 문장도 읽을지 판단 (쉼표, 세미콜론 등으로)"""
        # 쉼표나 세미콜론 등이 있으면 중간에도 읽기
        partial_endings = [', ', '; ', ': ', ' - ']

        for ending in partial_endings:
            if partial_text.endswith(ending):
                return True

        # 충분히 긴 구문 (20자 이상)이면 읽기
        if len(partial_text.strip()) >= 20:
            return True

        return False

    def generate_with_realtime_speech(
        self,
        llm_handler,
        prompt: str,
        model_key: str = None,
        voice_language: str = "ko",
        voice_slow: bool = False,
        read_partial: bool = True
    ):
        """텍스트를 스트리밍 생성하면서 실시간으로 음성 읽기"""

        try:
            # 텍스트 생성 스트림 시작
            text_stream = llm_handler.generate_stream(prompt, model_key)

            full_text = ""
            sentence_counter = 0
            audio_files = []

            # 시작 메시지
            yield {
                "type": "status",
                "status": "starting",
                "message": "텍스트 생성 및 실시간 음성 합성을 시작합니다..."
            }

            for chunk_data in text_stream:
                if chunk_data.get('error'):
                    yield {
                        "type": "error",
                        "error": chunk_data['error']
                    }
                    return

                text_chunk = chunk_data.get('content', '')
                if not text_chunk:
                    continue

                full_text += text_chunk

                # 텍스트 청크 전송
                yield {
                    "type": "text_chunk",
                    "content": text_chunk,
                    "full_text": full_text
                }

                # 완성된 문장 추출
                complete_sentences = self.extract_complete_sentences(text_chunk)

                # 완성된 문장들을 음성으로 변환
                for sentence in complete_sentences:
                    if sentence.strip():
                        sentence_counter += 1

                        # 음성 합성
                        tts_result = self.speech_service.text_to_speech(
                            text=sentence,
                            language=voice_language,
                            slow=voice_slow
                        )

                        if tts_result["success"]:
                            audio_files.append(tts_result["audio_file"])

                            yield {
                                "type": "sentence_audio",
                                "sentence_number": sentence_counter,
                                "sentence_text": sentence,
                                "audio_file": tts_result["audio_file"],
                                "audio_ready": True
                            }
                        else:
                            yield {
                                "type": "sentence_audio",
                                "sentence_number": sentence_counter,
                                "sentence_text": sentence,
                                "audio_error": tts_result["error"],
                                "audio_ready": False
                            }

                # 부분 문장 처리 (선택적)
                if read_partial and self.sentence_buffer:
                    if self.should_speak_partial_sentence(self.sentence_buffer):
                        partial_text = self.sentence_buffer.strip()

                        tts_result = self.speech_service.text_to_speech(
                            text=partial_text,
                            language=voice_language,
                            slow=voice_slow
                        )

                        if tts_result["success"]:
                            yield {
                                "type": "partial_audio",
                                "partial_text": partial_text,
                                "audio_file": tts_result["audio_file"],
                                "audio_ready": True
                            }

                        # 읽은 부분은 버퍼에서 제거
                        self.sentence_buffer = ""

            # 남은 텍스트 처리
            if self.sentence_buffer.strip():
                final_text = self.sentence_buffer.strip()
                sentence_counter += 1

                tts_result = self.speech_service.text_to_speech(
                    text=final_text,
                    language=voice_language,
                    slow=voice_slow
                )

                if tts_result["success"]:
                    audio_files.append(tts_result["audio_file"])

                    yield {
                        "type": "sentence_audio",
                        "sentence_number": sentence_counter,
                        "sentence_text": final_text,
                        "audio_file": tts_result["audio_file"],
                        "audio_ready": True
                    }

            # 완료 메시지
            yield {
                "type": "completed",
                "status": "completed",
                "message": "텍스트 생성 및 음성 합성이 완료되었습니다.",
                "full_text": full_text,
                "total_sentences": sentence_counter,
                "audio_files": audio_files
            }

        except Exception as e:
            logger.error(f"실시간 스트리밍 TTS 오류: {e}")
            yield {
                "type": "error",
                "error": str(e)
            }

    def reset_buffer(self):
        """문장 버퍼 초기화"""
        self.sentence_buffer = ""

    def cleanup_audio_files(self, audio_files: List[str]):
        """생성된 임시 오디오 파일들을 정리"""
        for audio_file in audio_files:
            try:
                self.speech_service.cleanup_temp_file(audio_file)
            except Exception as e:
                logger.warning(f"오디오 파일 정리 실패: {e}")

class SentenceBasedTTSService:
    """문장 단위로 음성을 생성하고 관리하는 서비스"""

    def __init__(self, speech_service: SpeechService):
        self.speech_service = speech_service

    async def convert_sentences_to_speech(
        self,
        sentences: List[str],
        language: str = "ko",
        slow: bool = False
    ) -> List[Dict[str, Any]]:
        """여러 문장을 비동기적으로 음성으로 변환"""

        results = []

        for i, sentence in enumerate(sentences):
            if sentence.strip():
                tts_result = self.speech_service.text_to_speech(
                    text=sentence.strip(),
                    language=language,
                    slow=slow
                )

                results.append({
                    "sentence_index": i,
                    "sentence_text": sentence.strip(),
                    "audio_file": tts_result.get("audio_file") if tts_result["success"] else None,
                    "success": tts_result["success"],
                    "error": tts_result.get("error") if not tts_result["success"] else None
                })

        return results

    def split_text_into_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분할"""
        # 문장 구분자로 분할
        sentence_pattern = r'[.!?。！？]\s*'
        sentences = re.split(sentence_pattern, text)

        # 빈 문장 제거 및 정리
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # 문장 부호가 제거되었으므로 다시 추가
                if not sentence.endswith(('.', '!', '?', '。', '！', '？')):
                    sentence += '.'
                cleaned_sentences.append(sentence)

        return cleaned_sentences