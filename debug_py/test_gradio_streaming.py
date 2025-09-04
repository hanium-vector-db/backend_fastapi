import requests
import json
import time

def test_gradio_streaming():
    """Gradio UI를 통한 스트리밍 테스트"""
    
    # Gradio API 엔드포인트
    url = "http://localhost:8001/ui/gradio_api/call/generate_text"
    
    # 테스트 데이터
    data = {
        "data": [
            "안녕하세요! Python의 장점에 대해 설명해주세요.",  # prompt
            "qwen2.5-7b",  # model_key  
            True  # streaming_mode
        ]
    }
    
    print("Gradio 스트리밍 테스트 시작...")
    print("="*50)
    
    try:
        # Gradio API 호출
        response = requests.post(url, json=data, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print(f"큐 상태: {result}")
            
            # 결과 가져오기
            event_id = result.get("event_id")
            if event_id:
                result_url = f"http://localhost:8001/ui/gradio_api/call/generate_text/{event_id}"
                
                print("스트리밍 결과 대기 중...")
                time.sleep(2)  # 약간의 대기
                
                result_response = requests.get(result_url, timeout=60)
                if result_response.status_code == 200:
                    result_data = result_response.json()
                    print(f"결과: {result_data}")
                else:
                    print(f"결과 가져오기 실패: {result_response.status_code}")
            else:
                print("이벤트 ID를 찾을 수 없습니다.")
        else:
            print(f"요청 실패: {response.status_code}")
            print(f"응답: {response.text}")
            
    except Exception as e:
        print(f"오류 발생: {e}")

def test_direct_api_streaming():
    """직접 API를 통한 스트리밍 테스트"""
    print("\n" + "="*50)
    print("직접 API 스트리밍 테스트...")
    
    payload = {
        "prompt": "안녕하세요! Python의 특징을 간단히 설명해주세요.",
        "stream": True,
        "max_length": 200
    }
    
    try:
        response = requests.post(
            "http://localhost:8001/api/v1/generate",
            json=payload,
            stream=True,
            headers={'Accept': 'text/event-stream'},
            timeout=60
        )
        
        if response.status_code == 200:
            print("스트리밍 응답:")
            full_text = ""
            
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith('data: '):
                    try:
                        data_str = line[6:]  # 'data: ' 제거
                        data = json.loads(data_str)
                        if 'content' in data and data['content']:
                            print(data['content'], end='', flush=True)
                            full_text += data['content']
                        if data.get('done', False):
                            print("\n\n✅ 스트리밍 완료!")
                            break
                    except json.JSONDecodeError:
                        continue
            
            print(f"전체 길이: {len(full_text)} 문자")
        else:
            print(f"API 요청 실패: {response.status_code}")
            
    except Exception as e:
        print(f"API 오류: {e}")

if __name__ == "__main__":
    test_gradio_streaming()
    test_direct_api_streaming()