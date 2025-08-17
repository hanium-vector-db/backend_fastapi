import gradio as gr
import requests
import json
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 서버 주소
API_URL = "http://127.0.0.1:8000/api/v1"

def handle_api_error(response):
    """
    API 응답 에러를 처리하고 사용자에게 보여줄 메시지를 반환합니다.
    성공 시 None을, 실패 시 에러 메시지 문자열을 반환합니다.
    """
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", "알 수 없는 서버 오류")
            return f"**오류가 발생했습니다 (HTTP {response.status_code})**\n\n**오류 내용:**\n{detail}\n\n**서버 응답 원문:**\n```\n{response.text}\n```"
        except json.JSONDecodeError:
            return f"**서버에서 심각한 오류가 발생했습니다 (HTTP {response.status_code})**\n\n**서버 응답 원문:**\n```\n{response.text}\n```"
    return None

def make_api_call(endpoint, payload, method="post"):
    """중복되는 API 호출 로직을 처리하는 헬퍼 함수"""
    try:
        if method.lower() == "post":
            response = requests.post(f"{API_URL}/{endpoint}", json=payload, timeout=300) # 타임아웃 증가
        else:
            response = requests.get(f"{API_URL}/{endpoint}", timeout=30)
            
        error_message = handle_api_error(response)
        if error_message:
            return {"error": error_message}
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"백엔드 서버 연결 실패: {e}"}

def generate_text(prompt, model_key):
    """'/generate' 엔드포인트를 호출합니다."""
    if not prompt:
        return "오류: 프롬프트를 입력해주세요.", {"error": "Prompt is empty"}
    
    payload = {"prompt": prompt}
    if model_key and model_key != "기본 모델":
        payload["model_key"] = model_key

    result = make_api_call("generate", payload)
    if "error" in result:
        return result["error"], result
    
    return result.get("response", "응답 없음"), result.get("model_info", {})

def chat_with_bot(message, history, model_key):
    """'/chat' 엔드포인트를 호출합니다."""
    payload = {"message": message}
    if model_key and model_key != "기본 모델":
        payload["model_key"] = model_key

    result = make_api_call("chat", payload)
    if "error" in result:
        return result["error"]
        
    return result.get("response", "응답 없음")

def rag_query(question, model_key):
    """'/rag' 엔드포인트를 호출합니다."""
    if not question:
        return "오류: 질문을 입력해주세요.", "", {"error": "Question is empty"}

    payload = {"question": question}
    if model_key and model_key != "기본 모델":
        payload["model_key"] = model_key

    result = make_api_call("rag", payload)
    if "error" in result:
        return result["error"], "", result

    docs = result.get("relevant_documents", [])
    doc_str = ""
    if docs:
        for doc in docs:
            doc_str += f"### 📄 [{doc.get('title', '출처 없음')}]({doc.get('source', '#')})\n> {doc.get('content', '')}\n\n---\n"
    else:
        doc_str = "관련 문서를 찾지 못했습니다."
        
    return result.get("response", "응답 없음"), doc_str, result.get("model_info", {})

def update_rag_news(query, max_results):
    """'/rag/update-news' 엔드포인트를 호출하여 DB를 최신 뉴스로 업데이트합니다."""
    if not query:
        return "오류: 뉴스 검색어를 입력해주세요."
    
    gr.Info(f"'{query}' 관련 최신 뉴스 {max_results}개를 검색하여 DB 업데이트를 시작합니다. 잠시만 기다려주세요...")
    
    payload = {"query": query, "max_results": int(max_results)}
    result = make_api_call("rag/update-news", payload)
    
    if "error" in result:
        gr.Error("업데이트 실패!")
        return result["error"]
    
    gr.Info("업데이트 성공!")
    return result.get("message", "알 수 없는 응답")

def update_model_list():
    """UI가 로드될 때 서버에서 모델 목록을 동적으로 가져옵니다."""
    logger.info("UI: 모델 목록을 서버에서 가져오는 중...")
    try:
        response = requests.get(f"{API_URL}/models", timeout=10)
        response.raise_for_status()
        models = list(response.json().get("supported_models", {}).keys())
        choices = ["기본 모델"] + models
        logger.info(f"UI: 모델 목록 업데이트 완료. 찾은 모델: {models}")
        return gr.Dropdown(choices=choices, value="기본 모델"), gr.Dropdown(choices=choices, value="기본 모델"), gr.Dropdown(choices=choices, value="기본 모델")
    except requests.exceptions.RequestException as e:
        logger.error(f"UI: 서버에서 모델 목록을 가져오는 데 실패했습니다: {e}")
        gr.Warning("서버에서 모델 목록을 가져오지 못했습니다. 백엔드 서버가 실행 중인지 확인하세요.")
        return gr.Dropdown(choices=["기본 모델"], value="기본 모델"), gr.Dropdown(choices=["기본 모델"], value="기본 모델"), gr.Dropdown(choices=["기본 모델"], value="기본 모델")

# --- Gradio UI 구성 ---
with gr.Blocks(theme=gr.themes.Soft(), title="LLM 서버 UI") as gradio_ui:
    gr.Markdown("# 🤖 LLM FastAPI 서버 UI")
    gr.Markdown("Gradio 인터페이스를 통해 서버의 LLM 기능을 쉽게 사용해보세요.")
    
    with gr.Tabs():
        # 1. 텍스트 생성 탭
        with gr.TabItem("📝 텍스트 생성"):
            with gr.Row():
                with gr.Column(scale=2):
                    gen_prompt = gr.Textbox(lines=5, label="프롬프트", placeholder="인공지능의 미래에 대해 짧은 글을 써줘.")
                    gen_model_select = gr.Dropdown(label="모델 선택 (UI 로딩 시 자동 업데이트)")
                    gen_button = gr.Button("생성하기", variant="primary")
                with gr.Column(scale=3):
                    gen_output = gr.Textbox(lines=10, label="생성된 텍스트", interactive=False)
                    gen_model_info = gr.JSON(label="사용된 모델 정보 / 오류 상세")

        # 2. 채팅 탭
        with gr.TabItem("💬 채팅"):
            chat_model_select = gr.Dropdown(label="채팅 모델 선택 (UI 로딩 시 자동 업데이트)")
            gr.ChatInterface(
                fn=chat_with_bot,
                additional_inputs=[chat_model_select],
                chatbot=gr.Chatbot(height=400, label="채팅창"),
                textbox=gr.Textbox(placeholder="메시지를 입력하세요...", label="입력"),
                submit_btn="보내기",
            )

        # 3. RAG 질의응답 탭
        with gr.TabItem("📚 RAG 질의응답"):
            with gr.Accordion("최신 뉴스로 DB 업데이트", open=False):
                with gr.Row():
                    news_query = gr.Textbox(label="뉴스 검색어", placeholder="예: 삼성전자 AI")
                    news_max_results = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="최대 검색 기사 수")
                update_button = gr.Button("DB 업데이트 실행", variant="primary")
                update_status = gr.Textbox(label="업데이트 결과", interactive=False)
            
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column(scale=2):
                    rag_question = gr.Textbox(lines=2, label="질문", placeholder="삼성전자의 최신 AI 기술에 대해 알려줘.")
                    rag_model_select = gr.Dropdown(label="모델 선택 (UI 로딩 시 자동 업데이트)")
                    rag_button = gr.Button("질문하기", variant="primary")
                with gr.Column(scale=3):
                    rag_answer = gr.Textbox(lines=5, label="답변", interactive=False)
                    rag_model_info_output = gr.JSON(label="사용된 모델 정보 / 오류 상세")
            rag_docs = gr.Markdown(label="참고 문서")

    # --- 이벤트 핸들러 ---
    gen_button.click(fn=generate_text, inputs=[gen_prompt, gen_model_select], outputs=[gen_output, gen_model_info])
    rag_button.click(fn=rag_query, inputs=[rag_question, rag_model_select], outputs=[rag_answer, rag_docs, rag_model_info_output])
    update_button.click(fn=update_rag_news, inputs=[news_query, news_max_results], outputs=update_status)
    
    # UI가 브라우저에 로드될 때, update_model_list 함수를 실행하여 드롭다운 메뉴를 채웁니다.
    gradio_ui.load(fn=update_model_list, inputs=None, outputs=[gen_model_select, chat_model_select, rag_model_select])

if __name__ == "__main__":
    gradio_ui.launch()
