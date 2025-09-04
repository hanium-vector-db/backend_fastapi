import requests
import json

def test_api():
    """API 테스트"""
    base_url = "http://localhost:8001/api/v1"
    
    # 테스트 프롬프트들
    test_cases = [
        {
            "name": "한국어 인사",
            "prompt": "안녕하세요! 자기소개를 해주세요.",
            "max_length": 200
        },
        {
            "name": "Python 질문",
            "prompt": "Python에서 리스트와 튜플의 차이점을 설명해주세요.",
            "max_length": 300
        },
        {
            "name": "간단한 대화",
            "prompt": "오늘 날씨가 좋네요.",
            "max_length": 150
        }
    ]
    
    print("API 테스트 시작...")
    print("="*50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n테스트 {i}: {test_case['name']}")
        print(f"프롬프트: {test_case['prompt']}")
        print("-" * 30)
        
        try:
            # Generate API 테스트
            response = requests.post(
                f"{base_url}/generate",
                json={
                    "prompt": test_case["prompt"],
                    "max_length": test_case["max_length"]
                },
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print("SUCCESS!")
                print(f"응답: {result.get('response', 'N/A')}")
                print(f"모델 정보: {result.get('model_info', {}).get('model_id', 'N/A')}")
            else:
                print(f"ERROR (Status: {response.status_code})")
                print(f"오류 내용: {response.text}")
                
        except requests.exceptions.Timeout:
            print("TIMEOUT (60초 초과)")
        except Exception as e:
            print(f"EXCEPTION: {e}")
    
    print("\n" + "="*50)
    print("API 테스트 완료!")

if __name__ == "__main__":
    test_api()