import requests
import json

def test_streaming_api():
    """스트리밍 API 테스트"""
    base_url = "http://localhost:8001/api/v1"
    
    test_cases = [
        {
            "name": "일반 모드",
            "payload": {
                "prompt": "안녕하세요! 자기소개를 해주세요.",
                "max_length": 200,
                "stream": False
            }
        },
        {
            "name": "스트리밍 모드",
            "payload": {
                "prompt": "Python의 장점에 대해 설명해주세요.",
                "max_length": 300,
                "stream": True
            }
        }
    ]
    
    print("스트리밍 API 테스트 시작...")
    print("="*50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n테스트 {i}: {test_case['name']}")
        print(f"프롬프트: {test_case['payload']['prompt']}")
        print("-" * 40)
        
        try:
            if test_case["payload"]["stream"]:
                # 스트리밍 테스트
                print("스트리밍 응답:")
                response = requests.post(
                    f"{base_url}/generate",
                    json=test_case["payload"],
                    stream=True,
                    headers={'Accept': 'text/event-stream'},
                    timeout=60
                )
                
                if response.status_code == 200:
                    full_text = ""
                    for line in response.iter_lines(decode_unicode=True):
                        if line and line.startswith('data: '):
                            try:
                                data_str = line[6:]  # 'data: ' 제거
                                data = json.loads(data_str)
                                if 'error' in data:
                                    print(f"ERROR: {data['error']}")
                                    break
                                if 'content' in data and data['content']:
                                    print(data['content'], end='', flush=True)
                                    full_text += data['content']
                                if data.get('done', False):
                                    print("\nSTREAMING COMPLETE!")
                                    break
                            except json.JSONDecodeError:
                                continue
                    print(f"\n전체 텍스트 길이: {len(full_text)} 문자")
                else:
                    print(f"❌ 오류 (Status: {response.status_code})")
                    print(f"응답: {response.text}")
            else:
                # 일반 모드 테스트
                response = requests.post(
                    f"{base_url}/generate",
                    json=test_case["payload"],
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print("SUCCESS!")
                    print(f"응답: {result.get('response', 'N/A')}")
                else:
                    print(f"ERROR (Status: {response.status_code})")
                    print(f"응답: {response.text}")
                    
        except requests.exceptions.Timeout:
            print("TIMEOUT")
        except Exception as e:
            print(f"EXCEPTION: {e}")
    
    print("\n" + "="*50)
    print("스트리밍 API 테스트 완료!")

if __name__ == "__main__":
    test_streaming_api()