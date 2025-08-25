import requests
import json

def test_streaming_page():
    """스트리밍 페이지 테스트"""
    
    # 1. 스트리밍 페이지 접근 테스트
    print("1. 스트리밍 페이지 접근 테스트...")
    try:
        response = requests.get("http://localhost:8001/stream", timeout=5)
        if response.status_code == 200:
            print("SUCCESS: 스트리밍 페이지 접근 가능")
            if "실시간 스트리밍 LLM" in response.text:
                print("SUCCESS: 페이지 내용 확인됨")
            else:
                print("WARNING: 페이지 내용이 예상과 다름")
        else:
            print(f"ERROR: HTTP {response.status_code}")
    except Exception as e:
        print(f"ERROR: {e}")

    # 2. 메인 페이지에서 스트리밍 링크 확인
    print("\n2. 메인 페이지 엔드포인트 확인...")
    try:
        response = requests.get("http://localhost:8001/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            streaming_ui = data.get("endpoints", {}).get("UI 인터페이스", {}).get("streaming_ui")
            if streaming_ui == "/stream":
                print("SUCCESS: 메인 페이지에 스트리밍 링크 확인됨")
            else:
                print("WARNING: 스트리밍 링크가 메인 페이지에 없음")
        else:
            print(f"ERROR: HTTP {response.status_code}")
    except Exception as e:
        print(f"ERROR: {e}")

    # 3. API 스트리밍 기능 재확인
    print("\n3. API 스트리밍 기능 재확인...")
    try:
        payload = {
            "prompt": "간단한 테스트입니다.",
            "stream": True,
            "max_length": 100
        }
        
        response = requests.post(
            "http://localhost:8001/api/v1/generate",
            json=payload,
            stream=True,
            headers={'Accept': 'text/event-stream'},
            timeout=30
        )
        
        if response.status_code == 200:
            print("SUCCESS: 스트리밍 API 접근 가능")
            token_count = 0
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        if 'content' in data and data['content']:
                            token_count += 1
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
            print(f"SUCCESS: {token_count}개 토큰 스트리밍 받음")
        else:
            print(f"ERROR: HTTP {response.status_code}")
    except Exception as e:
        print(f"ERROR: {e}")

    print("\n" + "="*50)
    print("스트리밍 페이지 테스트 완료!")
    print("브라우저에서 http://localhost:8001/stream 를 방문해보세요!")
    print("="*50)

if __name__ == "__main__":
    test_streaming_page()