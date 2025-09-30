#!/usr/bin/env python3
"""
음성 기능 테스트 스크립트
"""

import requests
import json
import sys
import os

# 서버 URL
BASE_URL = "http://localhost:8000/api/v1"

def test_speech_service_status():
    """음성 서비스 상태 테스트"""
    print("🔍 음성 서비스 상태 확인 중...")
    try:
        response = requests.get(f"{BASE_URL}/speech/status", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 음성 서비스 상태: {result.get('status')}")
            print(f"   - Whisper: {'✅' if result.get('whisper_available') else '❌'}")
            print(f"   - Google STT: {'✅' if result.get('google_stt_available') else '❌'}")
            print(f"   - gTTS: {'✅' if result.get('gtts_available') else '❌'}")
            print(f"   - 마이크: {'✅' if result.get('microphone_available') else '❌'}")
            print(f"   - 지원 언어 수: {result.get('supported_languages', 0)}개")
            return True
        else:
            print(f"❌ 상태 확인 실패: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 상태 확인 오류: {e}")
        return False

def test_text_to_speech():
    """텍스트를 음성으로 변환 테스트"""
    print("\n🔊 텍스트→음성 변환 테스트...")

    test_text = "안녕하세요! 이것은 음성 합성 테스트입니다."

    try:
        payload = {
            "text": test_text,
            "language": "ko",
            "slow": False
        }

        response = requests.post(f"{BASE_URL}/speech/text-to-speech", json=payload, timeout=30)

        if response.status_code == 200:
            # 음성 파일 저장
            output_file = "test_output.mp3"
            with open(output_file, 'wb') as f:
                f.write(response.content)

            print(f"✅ 음성 합성 성공! 파일 저장: {output_file}")
            print(f"   - 텍스트: {test_text}")
            print(f"   - 언어: ko")

            # 파일 크기 확인
            file_size = os.path.getsize(output_file)
            print(f"   - 파일 크기: {file_size} bytes")

            return True
        else:
            error_detail = response.json().get("detail", "알 수 없는 오류")
            print(f"❌ 음성 합성 실패: {error_detail}")
            return False

    except Exception as e:
        print(f"❌ 음성 합성 오류: {e}")
        return False

def test_voice_chat():
    """음성 채팅 테스트 (텍스트 입력)"""
    print("\n💬 음성 채팅 테스트...")

    test_message = "안녕하세요! 간단한 인사말을 해주세요."

    try:
        payload = {
            "text": test_message,
            "model_key": None,  # 기본 모델 사용
            "voice_language": "ko",
            "voice_slow": False
        }

        response = requests.post(f"{BASE_URL}/speech/voice-chat", json=payload, timeout=120)

        if response.status_code == 200:
            # 응답 텍스트 확인
            response_text = response.headers.get('X-Response-Text', '응답 텍스트 없음')

            # 음성 파일 저장
            output_file = "test_chat_response.mp3"
            with open(output_file, 'wb') as f:
                f.write(response.content)

            print(f"✅ 음성 채팅 성공!")
            print(f"   - 입력: {test_message}")
            print(f"   - AI 응답: {response_text}")
            print(f"   - 음성 파일: {output_file}")

            # 파일 크기 확인
            file_size = os.path.getsize(output_file)
            print(f"   - 파일 크기: {file_size} bytes")

            return True
        else:
            error_detail = response.json().get("detail", "알 수 없는 오류")
            print(f"❌ 음성 채팅 실패: {error_detail}")
            return False

    except Exception as e:
        print(f"❌ 음성 채팅 오류: {e}")
        return False

def test_supported_languages():
    """지원 언어 목록 테스트"""
    print("\n🌐 지원 언어 목록 테스트...")

    try:
        response = requests.get(f"{BASE_URL}/speech/languages", timeout=10)

        if response.status_code == 200:
            result = response.json()
            languages = result.get("supported_languages", {})

            print(f"✅ 지원 언어 조회 성공!")
            print(f"   - 총 {result.get('total_languages', 0)}개 언어 지원")
            print(f"   - 기본 언어: {result.get('default_language', 'unknown')}")
            print("   - 지원 언어 목록:")

            for code, name in languages.items():
                print(f"     * {code}: {name}")

            return True
        else:
            error_detail = response.json().get("detail", "알 수 없는 오류")
            print(f"❌ 언어 목록 조회 실패: {error_detail}")
            return False

    except Exception as e:
        print(f"❌ 언어 목록 조회 오류: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🎤 음성 기능 테스트 시작\n")
    print("=" * 50)

    # 서버 연결 확인
    try:
        response = requests.get("http://localhost:8000", timeout=5)
        if response.status_code != 200:
            print("❌ 서버가 실행되지 않았습니다. src/main.py를 먼저 실행해주세요.")
            return False
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}")
        print("서버가 실행되지 않았습니다. src/main.py를 먼저 실행해주세요.")
        return False

    print("✅ 서버 연결 확인 완료\n")

    # 테스트 실행
    tests = [
        ("음성 서비스 상태", test_speech_service_status),
        ("지원 언어 목록", test_supported_languages),
        ("텍스트→음성 변환", test_text_to_speech),
        ("음성 채팅", test_voice_chat),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n📋 {test_name} 테스트 실행 중...")
        print("-" * 30)

        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 예외 발생: {e}")
            results.append((test_name, False))

    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ 통과" if success else "❌ 실패"
        print(f"{test_name}: {status}")
        if success:
            passed += 1

    print(f"\n총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 모든 음성 기능 테스트가 성공적으로 완료되었습니다!")
        return True
    else:
        print("⚠️  일부 테스트가 실패했습니다. 로그를 확인해주세요.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)