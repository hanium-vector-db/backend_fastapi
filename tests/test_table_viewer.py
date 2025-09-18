#!/usr/bin/env python3
"""
테이블 뷰어 기능 테스트 스크립트
새로 추가된 Internal DB RAG 테이블 관리 기능을 테스트합니다.
"""

import requests
import json

# API 서버 URL
BASE_URL = "http://localhost:8000/api/v1"

def test_table_listing():
    """테이블 목록 조회 테스트"""
    print("🔍 테이블 목록 조회 테스트...")

    try:
        response = requests.get(f"{BASE_URL}/internal-db/tables")

        if response.status_code == 200:
            result = response.json()
            print("✅ 테이블 목록 조회 성공!")
            print(f"   - 백엔드: {result.get('backend', 'N/A')}")
            print(f"   - 시뮬레이션: {result.get('simulate', 'N/A')}")
            print(f"   - 테이블: {result.get('tables', [])}")
            return result.get('tables', [])
        else:
            print(f"❌ 테이블 목록 조회 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return []

    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        return []

def test_table_viewer(table_name, limit=10):
    """테이블 내용 보기 테스트"""
    print(f"\n📊 테이블 '{table_name}' 내용 보기 테스트...")

    try:
        response = requests.get(
            f"{BASE_URL}/internal-db/view-table/{table_name}",
            params={"limit": limit}
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ 테이블 조회 성공!")
            print(f"   - 테이블명: {result.get('table_name', 'N/A')}")
            print(f"   - 모드: {'시뮬레이션' if result.get('simulate') else '실제 DB'}")
            print(f"   - 전체 행 수: {result.get('total_rows', 0)}")
            print(f"   - 표시 행 수: {result.get('displayed_rows', 0)}")
            print(f"   - 컬럼: {result.get('columns', [])}")
            print(f"   - 메시지: {result.get('message', 'N/A')}")

            # 데이터 샘플 출력
            data = result.get('data', [])
            if data:
                print(f"\n📋 데이터 샘플 (최대 3행):")
                for i, row in enumerate(data[:3]):
                    print(f"   행 {i+1}: {row}")
            else:
                print("   ⚠️ 데이터가 없습니다.")

            return True
        else:
            print(f"❌ 테이블 조회 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return False

    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        return False

def test_gradio_ui_access():
    """Gradio UI 접근 테스트"""
    print("\n🌐 Gradio UI 접근 테스트...")

    try:
        # Gradio는 보통 8000 포트를 사용
        response = requests.get("http://localhost:8000", timeout=5)

        if response.status_code == 200:
            print("✅ Gradio UI 접근 성공!")
            print("   URL: http://localhost:8000")
            return True
        else:
            print(f"❌ Gradio UI 접근 실패: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Gradio UI 서버에 연결할 수 없습니다.")
        print("   서버가 실행 중인지 확인하세요:")
        print("   python src/main.py  # FastAPI 서버")
        print("   python -m gradio src/gradio_app.py  # Gradio UI")
        return False

    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        return False

def main():
    """전체 테스트 실행"""
    print("🚀 Internal DB RAG 테이블 뷰어 기능 테스트 시작\n")
    print("=" * 60)

    results = []

    # 1. 테이블 목록 조회
    tables = test_table_listing()
    results.append(len(tables) > 0)

    # 2. 각 테이블 내용 보기 (시뮬레이션 데이터)
    test_tables = tables if tables else ["knowledge", "products"]

    for table in test_tables[:2]:  # 최대 2개 테이블만 테스트
        result = test_table_viewer(table, limit=5)
        results.append(result)

    # 3. Gradio UI 접근
    ui_result = test_gradio_ui_access()
    results.append(ui_result)

    # 결과 요약
    print("\n" + "=" * 60)
    print("📈 테스트 결과 요약:")

    test_names = ["테이블 목록 조회"] + [f"테이블 '{t}' 조회" for t in test_tables[:2]] + ["Gradio UI 접근"]

    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ 성공" if result else "❌ 실패"
        print(f"   {i+1}. {name}: {status}")

    success_count = sum(results)
    total_count = len(results)

    if success_count == total_count:
        print(f"\n🎉 모든 테스트 성공! ({success_count}/{total_count})")
        print("\n📋 사용 방법:")
        print("1. FastAPI 서버 실행: python src/main.py")
        print("2. Gradio UI 접속: http://localhost:8000")
        print("3. 'Internal-DBMS RAG' 탭 → '테이블 관리' 이동")
        print("4. '테이블 목록 조회' 클릭")
        print("5. 드롭다운에서 테이블 선택")
        print("6. '테이블 내용 보기' 클릭")
    else:
        print(f"\n⚠️  일부 테스트 실패 ({success_count}/{total_count})")
        print("   실패한 테스트를 확인하고 문제를 해결하세요.")

    return success_count == total_count

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)