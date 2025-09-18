#!/usr/bin/env python3
"""
테이블 API 디버그 스크립트
Internal DB RAG 테이블 관련 API 엔드포인트를 직접 테스트합니다.
"""

import requests
import json

def test_api_endpoint(url, method="GET", data=None):
    """API 엔드포인트 테스트"""
    print(f"\n🔍 Testing: {method} {url}")

    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=10)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=10)

        print(f"   Status: {response.status_code}")
        print(f"   Headers: {dict(response.headers)}")

        if response.status_code == 200:
            try:
                result = response.json()
                print(f"   ✅ Success!")
                print(f"   Response keys: {list(result.keys())}")
                return result
            except json.JSONDecodeError:
                print(f"   ⚠️ Non-JSON response: {response.text[:200]}...")
                return None
        else:
            print(f"   ❌ Failed: {response.text}")
            return None

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def main():
    # 가능한 API 서버 주소들
    base_urls = [
        "http://localhost:8001/api/v1",  # FastAPI 기본
        "http://localhost:8000/api/v1",  # 대체 포트
        "http://127.0.0.1:8001/api/v1",  # 로컬호스트 대체
    ]

    print("🚀 Internal DB RAG API 디버그 테스트")
    print("=" * 50)

    working_base_url = None

    # 1. API 서버 연결 확인
    for base_url in base_urls:
        print(f"\n🔗 Trying base URL: {base_url}")
        result = test_api_endpoint(f"{base_url}/models")
        if result:
            working_base_url = base_url
            print(f"   ✅ Found working API server!")
            break

    if not working_base_url:
        print("\n❌ No working API server found!")
        print("Please start the FastAPI server:")
        print("   python src/main.py")
        return

    print(f"\n✅ Using API base URL: {working_base_url}")

    # 2. Internal DB 관련 엔드포인트 테스트
    endpoints = [
        ("GET", "/internal-db/tables", None),
        ("GET", "/internal-db/tables?simulate=true", None),
        ("GET", "/internal-db/status", None),
        ("GET", "/internal-db/view-table/knowledge", None),
        ("GET", "/internal-db/view-table/knowledge?simulate=true&limit=5", None),
        ("GET", "/internal-db/view-table/products?simulate=true&limit=3", None),
    ]

    results = []

    for method, endpoint, data in endpoints:
        full_url = working_base_url + endpoint
        result = test_api_endpoint(full_url, method, data)
        results.append((endpoint, result is not None))

    # 3. 결과 요약
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    for endpoint, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {endpoint}")

    success_count = sum(1 for _, success in results if success)
    total_count = len(results)

    if success_count == total_count:
        print(f"\n🎉 All tests passed! ({success_count}/{total_count})")
        print("\nThe API endpoints are working correctly.")
        print("If the UI still doesn't work, the issue is in the Gradio interface.")
    else:
        print(f"\n⚠️ Some tests failed ({success_count}/{total_count})")
        print("Check the FastAPI server logs for errors.")

if __name__ == "__main__":
    main()