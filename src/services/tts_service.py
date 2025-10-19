"""
TTS (Text-to-Speech) 서비스
Edge TTS (Microsoft)를 사용한 자연스러운 한국어 음성 합성
"""

import logging
import asyncio
import edge_tts
from io import BytesIO
import tempfile
import os

logger = logging.getLogger(__name__)


class TTSService:
    """Text-to-Speech 서비스 (Edge TTS)"""

    def __init__(self):
        # 한국어 음성: SunHiNeural (여성, 자연스러움)
        self.voice = "ko-KR-SunHiNeural"
        # 대안: ko-KR-InJoonNeural (남성)

    def text_to_speech(self, text: str, language: str = 'ko', slow: bool = False) -> BytesIO:
        """
        텍스트를 음성으로 변환 (Edge TTS)

        Args:
            text: 변환할 텍스트
            language: 언어 코드 (기본값: 'ko')
            slow: 느린 속도로 읽기 (기본값: False)

        Returns:
            BytesIO: MP3 오디오 데이터
        """
        try:
            # [PAGE:...] 형식의 링크 제거
            import re
            clean_text = re.sub(r'\[PAGE:[^\]]+\]', '', text)

            # 마크다운 제거
            clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_text)  # **bold**
            clean_text = re.sub(r'\*([^*]+)\*', r'\1', clean_text)  # *italic*
            clean_text = re.sub(r'`([^`]+)`', r'\1', clean_text)  # `code`
            clean_text = re.sub(r'#+\s', '', clean_text)  # # heading

            # 빈 텍스트 체크
            if not clean_text.strip():
                raise ValueError("변환할 텍스트가 비어있습니다.")

            logger.info(f"Edge TTS 변환 시작: {clean_text[:50]}...")

            # 속도 조정
            rate = "-20%" if slow else "+0%"

            # 임시 파일에 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_path = temp_file.name

            # Edge TTS로 음성 생성 (비동기 함수를 동기적으로 실행)
            asyncio.run(self._generate_audio(clean_text, temp_path, rate))

            # 파일 읽기
            with open(temp_path, 'rb') as f:
                audio_data = f.read()

            # 임시 파일 삭제
            os.unlink(temp_path)

            # BytesIO로 변환
            audio_buffer = BytesIO(audio_data)
            audio_buffer.seek(0)

            logger.info("Edge TTS 변환 완료")
            return audio_buffer

        except Exception as e:
            logger.error(f"Edge TTS 변환 오류: {e}")
            raise

    async def _generate_audio(self, text: str, output_path: str, rate: str):
        """비동기 음성 생성"""
        communicate = edge_tts.Communicate(text, self.voice, rate=rate)
        await communicate.save(output_path)
