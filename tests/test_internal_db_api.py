#!/usr/bin/env python3
"""
Internal DB RAG API 테스트 스크립트
시뮬레이션 모드로 MariaDB 없이 테스트 가능
"""

import requests
import json

# API 서버 URL (포트는 실제 서버 포트로 조정)
BASE_URL = "http://localhost:8001/api/v1"

def test_internal_db_tables():
    """테이블 목록 조회 테스트 (시뮬레이션 모드)"""
    print("🔍 Internal DB 테이블 조회 테스트...")

    try:
        # 시뮬레이션 모드로 테이블 조회
        response = requests.get(f"{BASE_URL}/internal-db/tables?simulate=true")

        if response.status_code == 200:
            result = response.json()
            print("✅ 테이블 조회 성공!")
            print(f"   - 백엔드: {result['backend']}")
            print(f"   - 시뮬레이션: {result['simulate']}")
            print(f"   - 테이블: {result['tables']}")
            return True
        else:
            print(f"❌ 테이블 조회 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ API 서버에 연결할 수 없습니다.")
        print("   서버가 실행 중인지 확인하세요: python src/main.py")
        return False
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        return False

def test_internal_db_ingest():
    """테이블 인제스트 테스트 (시뮬레이션 모드)"""
    print("\n📥 Internal DB 인제스트 테스트...")

    payload = {
        "table": "knowledge",
        "save_name": "test_knowledge",
        "simulate": True  # 시뮬레이션 모드
    }

    try:
        response = requests.post(
            f"{BASE_URL}/internal-db/ingest",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ 인제스트 성공!")
            print(f"   - 저장 경로: {result['save_dir']}")
            print(f"   - 처리된 행: {result['rows']}")
            print(f"   - 생성된 청크: {result['chunks']}")
            print(f"   - 스키마: {result['schema']}")
            return True
        else:
            print(f"❌ 인제스트 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return False

    except Exception as e:
        print(f"❌ 인제스트 오류: {e}")
        return False

def test_internal_db_query():
    """질의응답 테스트"""
    print("\n💬 Internal DB 질의응답 테스트...")

    payload = {
        "save_name": "test_knowledge",
        "question": "Self-Attention은 무엇인가? 역할과 함께 설명하라.",
        "top_k": 5,
        "margin": 0.12
    }

    try:
        response = requests.post(
            f"{BASE_URL}/internal-db/query",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ 질의응답 성공!")
            print(f"   - 질문: {result['question']}")
            print(f"   - 답변: {result['answer'][:200]}...")
            print(f"   - 참조 소스: {len(result['sources'])}개")
            for i, source in enumerate(result['sources'][:3]):
                print(f"     [{source['marker']}] {source['title']} (점수: {source['score']:.3f})")
            return True
        else:
            print(f"❌ 질의응답 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return False

    except Exception as e:
        print(f"❌ 질의응답 오류: {e}")
        return False

def test_internal_db_status():
    """상태 조회 테스트"""
    print("\n📊 Internal DB 상태 조회 테스트...")

    try:
        response = requests.get(f"{BASE_URL}/internal-db/status")

        if response.status_code == 200:
            result = response.json()
            print("✅ 상태 조회 성공!")
            print(f"   - FAISS 인덱스: {result['faiss_indices']}")
            print(f"   - 캐시된 키: {result['cache_keys']}")
            return True
        else:
            print(f"❌ 상태 조회 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return False

    except Exception as e:
        print(f"❌ 상태 조회 오류: {e}")
        return False

def main():
    """전체 테스트 실행"""
    print("🚀 Enhanced Internal DB RAG API 테스트 시작\n")
    print("=" * 60)

    results = []

    # 1. 테이블 조회 (시뮬레이션)
    results.append(test_internal_db_tables())

    # 2. 인제스트 (시뮬레이션)
    results.append(test_internal_db_ingest())

    # 3. 질의응답
    results.append(test_internal_db_query())

    # 4. 상태 조회
    results.append(test_internal_db_status())

    # 결과 요약
    print("\n" + "=" * 60)
    print("📈 테스트 결과 요약:")

    test_names = ["테이블 조회", "인제스트", "질의응답", "상태 조회"]

    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ 성공" if result else "❌ 실패"
        print(f"   {i+1}. {name}: {status}")

    success_count = sum(results)
    total_count = len(results)

    if success_count == total_count:
        print(f"\n🎉 모든 테스트 성공! ({success_count}/{total_count})")
        print("   Enhanced Internal DB RAG API가 정상 작동합니다.")
    else:
        print(f"\n⚠️  일부 테스트 실패 ({success_count}/{total_count})")
        print("   실패한 테스트를 확인하고 문제를 해결하세요.")

if __name__ == "__main__":
    main()