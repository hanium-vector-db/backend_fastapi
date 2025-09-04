"""
Hugging Face 토큰 설정 및 Llama 접근 권한 확인 스크립트
"""
import os
import sys
from huggingface_hub import login, whoami

def setup_huggingface_token():
    """Hugging Face 토큰 설정"""
    print("🤖 Hugging Face 토큰 설정")
    print("="*50)
    
    # 기존 토큰 확인
    token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    
    if token:
        print(f"✅ 환경변수에서 토큰 발견: {token[:10]}...")
        try:
            user_info = whoami(token=token)
            print(f"✅ 인증된 사용자: {user_info['name']}")
            return token
        except Exception as e:
            print(f"❌ 토큰 인증 실패: {e}")
    else:
        print("⚠️ 환경변수에 토큰이 없습니다.")
    
    # 토큰 입력 받기
    print("\n📝 Hugging Face 토큰을 입력하세요:")
    print("1. https://huggingface.co/settings/tokens 에서 토큰 생성")
    print("2. 'Read' 권한으로 충분합니다")
    print("3. 아래에 토큰을 입력하세요:")
    
    new_token = input("\nHugging Face Token: ").strip()
    
    if not new_token:
        print("❌ 토큰이 입력되지 않았습니다.")
        return None
    
    try:
        login(token=new_token)
        user_info = whoami(token=new_token)
        print(f"✅ 로그인 성공: {user_info['name']}")
        
        # 환경변수로 설정
        os.environ['HUGGINGFACE_TOKEN'] = new_token
        
        return new_token
    except Exception as e:
        print(f"❌ 토큰 인증 실패: {e}")
        return None

def check_llama_access(token):
    """Llama 모델 접근 권한 확인"""
    print("\n🦙 Llama 3.1 접근 권한 확인")
    print("="*50)
    
    try:
        from transformers import AutoTokenizer
        
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        print(f"🔍 모델 접근 테스트: {model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            cache_dir="/home/ubuntu_euphoria/.huggingface_models"
        )
        
        print("✅ Llama 3.1 접근 권한 확인됨!")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Llama 3.1 접근 실패: {error_msg}")
        
        if "gated repo" in error_msg or "restricted" in error_msg:
            print("\n📋 해결 방법:")
            print("1. https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct 방문")
            print("2. 'Request access to this model' 클릭")
            print("3. Meta AI의 라이선스 동의")
            print("4. 승인까지 몇 분~몇 시간 소요")
            
        return False

def suggest_alternatives():
    """대체 모델 제안"""
    print("\n🔄 대체 모델 옵션")
    print("="*50)
    
    alternatives = [
        {
            "name": "Microsoft Phi-3.5 Mini",
            "model_id": "microsoft/Phi-3.5-mini-instruct",
            "description": "고성능 소형 모델, 접근 제한 없음"
        },
        {
            "name": "Mistral 7B v0.3",
            "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "description": "Mistral AI의 고성능 모델"
        },
        {
            "name": "CodeLlama 7B",
            "model_id": "codellama/CodeLlama-7b-Instruct-hf",
            "description": "코딩 특화 Llama 기반 모델"
        }
    ]
    
    for i, alt in enumerate(alternatives, 1):
        print(f"{i}. {alt['name']}")
        print(f"   모델 ID: {alt['model_id']}")
        print(f"   설명: {alt['description']}")
        print()
    
    choice = input("대체 모델을 사용하시겠습니까? (1-3, 또는 Enter로 건너뛰기): ").strip()
    
    if choice in ['1', '2', '3']:
        selected = alternatives[int(choice) - 1]
        print(f"\n✅ 선택된 대체 모델: {selected['name']}")
        print(f"📝 llm_handler.py 에서 모델 ID를 다음으로 변경하세요:")
        print(f"   {selected['model_id']}")
        return selected
    
    return None

def main():
    print("🤖 LLM 서버 Hugging Face 설정")
    print("="*60)
    
    # 1. 토큰 설정
    token = setup_huggingface_token()
    if not token:
        print("❌ 토큰 설정이 완료되지 않았습니다.")
        sys.exit(1)
    
    # 2. Llama 접근 권한 확인
    has_llama_access = check_llama_access(token)
    
    # 3. 접근 권한이 없으면 대체 모델 제안
    if not has_llama_access:
        suggest_alternatives()
    
    print("\n" + "="*60)
    print("🎉 설정 완료! 이제 서버를 시작할 수 있습니다:")
    print("   python src/main.py")
    print("="*60)

if __name__ == "__main__":
    main()