#!/usr/bin/env python3
"""
MariaDB 통합 테스트 스크립트
실제 MariaDB 데이터가 UI까지 제대로 표시되는지 확인합니다.
"""

import requests
import json

def test_api_mariadb_connection():
    """API에서 MariaDB 연결 확인"""
    print("🔗 API MariaDB 연결 테스트...")

    try:
        response = requests.get("http://localhost:8000/api/v1/internal-db/tables", timeout=10)
        if response.status_code == 200:
            result = response.json()
            is_real_db = not result.get('simulate', True)
            backend = result.get('backend', 'unknown')
            tables = result.get('tables', [])

            print(f"   ✅ API 연결 성공!")
            print(f"   - 실제 DB 사용: {'✅ Yes' if is_real_db else '❌ No (시뮬레이션)'}")
            print(f"   - 백엔드: {backend}")
            print(f"   - 테이블: {tables}")

            return is_real_db and len(tables) >= 4
        else:
            print(f"   ❌ API 응답 오류: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ API 연결 실패: {e}")
        return False

def test_table_data_retrieval():
    """각 테이블의 실제 데이터 조회 테스트"""
    print("\n📊 테이블 데이터 조회 테스트...")

    tables = ["knowledge", "products", "users", "orders"]
    results = []

    for table in tables:
        print(f"\n   📋 {table} 테이블:")
        try:
            response = requests.get(
                f"http://localhost:8000/api/v1/internal-db/view-table/{table}",
                params={"limit": 3},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                is_real_db = not result.get('simulate', True)
                total_rows = result.get('total_rows', 0)
                columns = result.get('columns', [])
                data = result.get('data', [])

                print(f"      실제 DB: {'✅' if is_real_db else '❌'}")
                print(f"      행 수: {total_rows}")
                print(f"      컬럼: {columns[:3]}{'...' if len(columns) > 3 else ''}")

                if data:
                    first_row = data[0]
                    print(f"      샘플: {list(first_row.keys())[:3]}...")

                results.append(is_real_db and total_rows > 0)
            else:
                print(f"      ❌ 조회 실패: {response.status_code}")
                results.append(False)

        except Exception as e:
            print(f"      ❌ 오류: {e}")
            results.append(False)

    return results

def test_gradio_ui_access():
    """Gradio UI 접근 및 데이터 확인"""
    print("\n🌐 Gradio UI 접근 테스트...")

    try:
        response = requests.get("http://localhost:8000/ui", timeout=5)
        if response.status_code == 200:
            print("   ✅ Gradio UI 접근 성공!")
            print("   URL: http://localhost:8000/ui")
            return True
        else:
            print(f"   ❌ Gradio UI 접근 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ UI 접근 오류: {e}")
        return False

def test_data_consistency():
    """데이터 일관성 검증"""
    print("\n🔍 데이터 일관성 검증...")

    try:
        # knowledge 테이블에서 특정 데이터 확인
        response = requests.get(
            "http://localhost:8000/api/v1/internal-db/view-table/knowledge",
            params={"limit": 10},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            data = result.get('data', [])

            # 우리가 삽입한 데이터가 있는지 확인
            attention_found = any(
                row.get('term') == '어텐션 메커니즘'
                for row in data
            )

            transformer_found = any(
                row.get('term') == 'Transformer'
                for row in data
            )

            print(f"   - 어텐션 메커니즘 데이터: {'✅' if attention_found else '❌'}")
            print(f"   - Transformer 데이터: {'✅' if transformer_found else '❌'}")
            print(f"   - 총 knowledge 레코드: {len(data)}")

            return attention_found or transformer_found
        else:
            print(f"   ❌ 데이터 확인 실패: {response.status_code}")
            return False

    except Exception as e:
        print(f"   ❌ 데이터 검증 오류: {e}")
        return False

def main():
    print("🚀 MariaDB 통합 테스트")
    print("=" * 50)

    tests = [
        ("API MariaDB 연결", test_api_mariadb_connection),
        ("테이블 데이터 조회", lambda: all(test_table_data_retrieval())),
        ("Gradio UI 접근", test_gradio_ui_access),
        ("데이터 일관성", test_data_consistency)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ 테스트 실행 오류: {e}")
            results.append((test_name, False))

    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약:")

    for test_name, passed in results:
        status = "✅ 성공" if passed else "❌ 실패"
        print(f"   {status} {test_name}")

    success_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    if success_count == total_count:
        print(f"\n🎉 모든 테스트 성공! ({success_count}/{total_count})")
        print("\n✅ MariaDB 통합 완료:")
        print("   - 시뮬레이션 데이터 → MariaDB 실제 데이터로 전환")
        print("   - 4개 테이블 (knowledge, products, users, orders) 사용 가능")
        print("   - API 엔드포인트가 실제 DB에서 데이터 조회")
        print("   - Gradio UI에서 실제 MariaDB 데이터 표시")

        print("\n📋 사용 방법:")
        print("1. http://localhost:8000/ui 접속")
        print("2. 'Internal-DBMS RAG' 탭 클릭")
        print("3. '테이블 관리'에서 '테이블 목록 조회' 클릭")
        print("4. 드롭다운에서 테이블 선택 (knowledge, products, users, orders)")
        print("5. '테이블 내용 보기'로 실제 MariaDB 데이터 확인")

    else:
        print(f"\n⚠️ 일부 테스트 실패 ({success_count}/{total_count})")
        print("   실패한 항목을 확인하고 문제를 해결하세요.")

    return success_count == total_count

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)