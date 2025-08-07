import gradio as gr
import requests
import json

# FastAPI 서버 주소
API_URL = "http://127.0.0.1:8000/api/v1"

def handle_api_error(response):
    """API 응답 에러를 처리하고 사용자에게 보여줄 메시지를 반환합니다."""
    if response.status_code == 500:
        try:
            detail = response.json().get("detail", "알 수 없는 서버 오류")
            return f"오류: 백엔드 서버에서 오류가 발생했습니다. \n{detail}"
        except json.JSONDecodeError:
            return f"오류: 서버에서 유효하지 않은 응답을 보냈습니다. \n내용: {response.text}"
    response.raise_for_status() # 500 외 다른 HTTP 에러 처리
    return None

def generate_text(prompt, model_key):
    """'/generate' 엔드포인트를 호출하여 텍스트를 생성합니다."""
    if not prompt:
        return "오류: 프롬프트를 입력해주세요.", ""
    try:
        response = requests.post(
            f"{API_URL}/generate",
            json={"prompt": prompt, "model_key": model_key or None}
        )
        error = handle_api_error(response)
        if error:
            return error, ""
        
        data = response.json()
        return data.get("response", "응답이 없습니다."), data.get("model_info", "")
    except requests.exceptions.RequestException as e:
        return f"오류: 백엔드 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.\n{e}", ""

def chat_with_bot(message, history, model_key):
    """'/chat' 엔드포인트를 호출하여 채팅 응답을 받습니다."""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"message": message, "model_key": model_key or None}
        )
        error = handle_api_error(response)
        if error:
            return error
        
        return response.json().get("response", "응답이 없습니다.")
    except requests.exceptions.RequestException as e:
        return f"오류: 백엔드 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.\n{e}"

def rag_query(question, model_key):
    """'/rag' 엔드포인트를 호출하여 RAG 응답을 받습니다."""
    if not question:
        return "오류: 질문을 입력해주세요.", "", ""
    try:
        response = requests.post(
            f"{API_URL}/rag",
            json={"question": question, "model_key": model_key or None}
        )
        error = handle_api_error(response)
        if error:
            return error, "", ""
            
        data = response.json()
        answer = data.get("response", "응답이 없습니다.")
        docs = data.get("relevant_documents", [])
        
        # 관련 문서를 보기 좋게 포맷팅
        doc_str = ""
        if docs:
            for doc in docs:
                doc_str += f"### 📄 {doc.get('title', '출처 없음')}\n"
                doc_str += f"{doc.get('content', '')}\n\n---\n\n"
        else:
            doc_str = "관련 문서를 찾지 못했습니다."
            
        model_info = data.get("model_info", "")
        return answer, doc_str, model_info
    except requests.exceptions.RequestException as e:
        return f"오류: 백엔드 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.\n{e}", "", ""

def get_available_models():
    """서버에서 사용 가능한 모델 목록을 가져옵니다."""
    try:
        response = requests.get(f"{API_URL}/models")
        response.raise_for_status()
        return list(response.json().get("supported_models", {}).keys())
    except requests.exceptions.RequestException:
        return []

# UI 구성
with gr.Blocks(theme=gr.themes.Soft()) as gradio_ui:
    gr.Markdown("# 🤖 LLM FastAPI 서버 UI")
    gr.Markdown("Gradio 인터페이스를 통해 서버의 LLM 기능을 쉽게 사용해보세요.")
    
    model_choices = get_available_models()
    
    with gr.Tabs():
        # 1. 텍스트 생성 탭
        with gr.TabItem("📝 텍스트 생성"):
            with gr.Row():
                with gr.Column(scale=2):
                    gen_prompt = gr.Textbox(lines=5, label="프롬프트", placeholder="인공지능의 미래에 대해 짧은 글을 써줘.")
                    gen_model_select = gr.Dropdown(choices=["기본 모델"] + model_choices, value="기본 모델", label="모델 선택")
                    gen_button = gr.Button("생성하기", variant="primary")
                with gr.Column(scale=3):
                    gen_output = gr.Textbox(lines=10, label="생성된 텍스트", interactive=False)
                    gen_model_info = gr.JSON(label="사용된 모델 정보")

        # 2. 채팅 탭
        with gr.TabItem("💬 채팅"):
            with gr.Row():
                chat_model_select = gr.Dropdown(choices=["기본 모델"] + model_choices, value="기본 모델", label="채팅 모델 선택")
            
            gr.ChatInterface(
                fn=lambda message, history: chat_with_bot(message, history, chat_model_select.value),
                chatbot=gr.Chatbot(height=400, label="채팅창", bubble_full_width=False),
                textbox=gr.Textbox(placeholder="메시지를 입력하세요...", label="입력"),
                submit_btn="보내기"
            )

        # 3. RAG 질의응답 탭
        with gr.TabItem("📚 RAG 질의응답"):
            with gr.Row():
                with gr.Column(scale=2):
                    rag_question = gr.Textbox(lines=2, label="질문", placeholder="트랜스포머 모델의 주요 특징은 무엇이야?")
                    rag_model_select = gr.Dropdown(choices=["기본 모델"] + model_choices, value="기본 모델", label="모델 선택")
                    rag_button = gr.Button("질문하기", variant="primary")
                with gr.Column(scale=3):
                    rag_answer = gr.Textbox(lines=5, label="답변", interactive=False)
                    rag_model_info_output = gr.JSON(label="사용된 모델 정보")
            
            with gr.Row():
                rag_docs = gr.Markdown(label="참고 문서")

    # 버튼과 함수 연결
    gen_button.click(
        fn=generate_text,
        inputs=[gen_prompt, gen_model_select],
        outputs=[gen_output, gen_model_info]
    )
    rag_button.click(
        fn=rag_query,
        inputs=[rag_question, rag_model_select],
        outputs=[rag_answer, rag_docs, rag_model_info_output]
    )
