"""
Llama 3.1 접근 권한 확인 스크립트
"""
import os
from transformers import AutoTokenizer
from dotenv import load_dotenv

def check_llama_access():
    print("Llama 3.1 접근 권한 확인 중...")
    print("=" * 50)
    
    # .env 파일 로드
    load_dotenv()
    
    # 토큰 확인
    token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    
    if not token:
        print("ERROR: Hugging Face 토큰이 설정되지 않았습니다!")
        print("\n해결 방법:")
        print("1. https://huggingface.co/settings/tokens 에서 토큰 생성")
        print("2. .env 파일에 HUGGINGFACE_TOKEN='your_token' 추가")
        print("3. 환경변수 설정:")
        print("   Windows CMD: set HUGGINGFACE_TOKEN=your_token")
        print("   PowerShell: $env:HUGGINGFACE_TOKEN=\"your_token\"")
        return False
    
    print(f"SUCCESS: 토큰 확인됨: {token[:10]}...")
    
    try:
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        print(f"모델 접근 테스트: {model_id}")
        
        # 토크나이저 로드 시도
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            cache_dir="C:\\huggingface_models"
        )
        
        print("SUCCESS! Llama 3.1에 접근할 수 있습니다!")
        print("이제 서버에서 llama3.1-8b 모델을 사용할 수 있습니다.")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR 접근 실패: {error_msg}")
        
        if "gated repo" in error_msg or "restricted" in error_msg:
            print("\nERROR: 접근 권한이 없습니다!")
            print("\n해결 단계:")
            print("1. https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
            print("2. 'Request access to this model' 클릭")
            print("3. Meta AI 라이선스 동의")
            print("4. 승인 대기 (몇 분~몇 시간)")
            print("5. 승인 이메일 확인 후 다시 시도")
        elif "401" in error_msg:
            print("\nERROR: 토큰 인증 문제!")
            print("토큰이 유효한지 확인하세요.")
        else:
            print(f"\nERROR: 기타 오류: {error_msg}")
        
        return False

if __name__ == "__main__":
    if check_llama_access():
        print("\nSUCCESS: 준비 완료! 서버를 시작하세요:")
        print("   python src/main.py")
    else:
        print("\nWARNING: 권한 설정 후 다시 시도하세요.")