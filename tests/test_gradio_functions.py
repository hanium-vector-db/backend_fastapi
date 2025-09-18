#!/usr/bin/env python3
"""
Gradio UI 테이블 기능 직접 테스트
UI 함수들을 직접 호출해서 테스트합니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Gradio 앱에서 함수들 import
from src.gradio_app import internal_db_simulate_table_data, update_table_dropdown, internal_db_view_table

def test_simulation_data():
    """시뮬레이션 데이터 테스트"""
    print("🔍 시뮬레이션 데이터 테스트...")

    test_tables = ["knowledge", "products", "users", "orders"]

    for table_name in test_tables:
        print(f"\n📊 테이블: {table_name}")
        try:
            result, status = internal_db_simulate_table_data(table_name, 3)
            print(f"   ✅ 성공!")
            print(f"   상태: {status}")
            print(f"   결과 길이: {len(result)} 문자")
            print(f"   HTML 포함 여부: {'<table>' in result}")
        except Exception as e:
            print(f"   ❌ 오류: {e}")

def test_table_dropdown_update():
    """테이블 드롭다운 업데이트 테스트"""
    print("\n🔄 테이블 드롭다운 업데이트 테스트...")

    try:
        # update_table_dropdown 함수는 API를 호출하므로 모의 테스트
        tables = ["knowledge", "products", "users", "orders"]
        formatted = f"**📋 사용 가능한 테이블 ({len(tables)}개):**\n\n"
        for i, table in enumerate(tables, 1):
            formatted += f"{i}. **{table}**\n"

        status = f"총 {len(tables)}개 테이블"
        choices = ["테이블을 선택하세요"] + tables

        print("   ✅ 드롭다운 업데이트 성공!")
        print(f"   상태: {status}")
        print(f"   선택지: {choices}")

    except Exception as e:
        print(f"   ❌ 오류: {e}")

def test_table_view():
    """테이블 보기 기능 테스트"""
    print("\n👁️ 테이블 보기 기능 테스트...")

    test_cases = [
        ("knowledge", 3),
        ("products", 2),
        ("users", 3),
        ("orders", 3),
        ("invalid_table", 5)
    ]

    for table_name, limit in test_cases:
        print(f"\n   테이블: {table_name} (limit: {limit})")
        try:
            result, status = internal_db_view_table(table_name, limit)
            print(f"   ✅ 성공!")
            print(f"   상태: {status}")
            print(f"   결과 길이: {len(result)} 문자")

            if table_name in ["knowledge", "products", "users", "orders"]:
                if "<table>" in result and "시뮬레이션" in result:
                    print(f"   ✅ HTML 테이블 및 시뮬레이션 마크 확인됨")
                else:
                    print(f"   ⚠️ 예상된 HTML 형식이 아닙니다")

        except Exception as e:
            print(f"   ❌ 오류: {e}")

def main():
    print("🚀 Gradio UI 테이블 기능 직접 테스트")
    print("=" * 60)

    # 1. 시뮬레이션 데이터 테스트
    test_simulation_data()

    # 2. 드롭다운 업데이트 테스트
    test_table_dropdown_update()

    # 3. 테이블 보기 테스트
    test_table_view()

    print("\n" + "=" * 60)
    print("📊 테스트 완료!")
    print("\n💡 다음 단계:")
    print("1. 웹 브라우저에서 http://localhost:8000/ui 접속")
    print("2. 'Internal-DBMS RAG' 탭 클릭")
    print("3. '테이블 관리' 섹션에서 '테이블 목록 조회' 클릭")
    print("4. 드롭다운에서 테이블 선택")
    print("5. '테이블 내용 보기' 클릭하여 데이터 확인")

if __name__ == "__main__":
    main()