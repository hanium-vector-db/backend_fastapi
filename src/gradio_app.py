import gradio as gr
import requests
import json
import logging
import time
import re
from utils.config_loader import config

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 서버 주소 - 설정에서 읽어옴
backend_api_config = config.ui_backend_api_config
API_URL = backend_api_config['base_url']

def process_streaming_response(response):
    """
    스트리밍 응답을 처리하고 실시간 업데이트를 제공합니다.
    """
    content_buffer = ""
    status_message = ""
    final_result = None
    
    try:
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                try:
                    data_str = line[6:]  # "data: " 부분 제거
                    data = json.loads(data_str)
                    
                    if data.get('status') == 'streaming' and 'chunk' in data:
                        # 실시간 텍스트 청크
                        content_buffer += data['chunk']
                        yield content_buffer, status_message, None
                    
                    elif data.get('status') in ['starting', 'searching', 'processing', 'generating', 'categories', 'category_analyzing', 'overall_analyzing']:
                        # 진행 상태 메시지
                        status_message = data.get('message', '')
                        yield content_buffer, status_message, None
                    
                    elif data.get('status') == 'category_completed':
                        # 카테고리 완료 상태
                        category = data.get('category', '')
                        summary = data.get('summary', '')
                        status_message = f"✓ {category} 분석 완료"
                        yield content_buffer, status_message, None
                    
                    elif data.get('status') == 'completed':
                        # 최종 완료 상태
                        final_result = data
                        if 'summary' in data:
                            content_buffer = data['summary']
                        elif 'overall_trend' in data:
                            content_buffer = data['overall_trend']
                        status_message = "✓ 완료"
                        yield content_buffer, status_message, final_result
                        return
                    
                    elif data.get('status') == 'error':
                        # 에러 상태
                        error_msg = data.get('message', '알 수 없는 오류가 발생했습니다.')
                        yield f"❌ 오류: {error_msg}", "오류 발생", None
                        return
                        
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        yield f"❌ 스트리밍 처리 오류: {str(e)}", "오류", None

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

def generate_text(prompt, model_key, streaming_mode):
    """'/generate' 엔드포인트를 호출합니다."""
    if not prompt:
        return "오류: 프롬프트를 입력해주세요.", {"error": "Prompt is empty"}
    
    payload = {"prompt": prompt, "stream": streaming_mode}
    if model_key and model_key != "기본 모델":
        payload["model_key"] = model_key

    if streaming_mode:
        # 스트리밍 모드
        try:
            response = requests.post(
                f"{API_URL}/generate", 
                json=payload, 
                stream=True,
                headers={'Accept': 'text/event-stream'},
                timeout=300
            )
            
            if response.status_code != 200:
                return f"오류: HTTP {response.status_code}", {"error": "Stream failed"}
            
            full_text = ""
            token_count = 0
            
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith('data: '):
                    try:
                        data_str = line[6:]  # 'data: ' 제거
                        data = json.loads(data_str)
                        if 'error' in data:
                            return data['error'], {"error": data['error']}
                        if 'content' in data and data['content']:
                            full_text += data['content']
                            token_count += 1
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
            
            return full_text, {"model_info": {"streaming": True, "complete": True, "tokens": token_count}}
            
        except Exception as e:
            return f"스트리밍 오류: {str(e)}", {"error": str(e)}
    else:
        # 일반 모드
        result = make_api_call("generate", payload)
        
        if "error" in result:
            return result["error"], result
        else:
            return result.get("response", "응답 없음"), result.get("model_info", {})

def stream_generate_text_generator(payload):
    """스트리밍 텍스트 생성 (generator)"""
    try:
        response = requests.post(
            f"{API_URL}/generate", 
            json=payload, 
            stream=True,
            headers={'Accept': 'text/event-stream'},
            timeout=300
        )
        
        if response.status_code != 200:
            yield f"오류: HTTP {response.status_code}", {"error": "Stream failed"}
            return
        
        full_text = ""
        
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith('data: '):
                try:
                    data_str = line[6:]  # 'data: ' 제거
                    data = json.loads(data_str)
                    if 'error' in data:
                        yield data['error'], {"error": data['error']}
                        return
                    if 'content' in data and data['content']:
                        full_text += data['content']
                        # 실시간으로 누적된 텍스트 출력
                        yield full_text, {"model_info": {"streaming": True, "tokens_so_far": len(full_text.split())}}
                    if data.get('done', False):
                        # 최종 완성된 텍스트
                        yield full_text, {"model_info": {"streaming": True, "complete": True}}
                        return
                except json.JSONDecodeError:
                    continue
        
    except Exception as e:
        yield f"스트리밍 오류: {str(e)}", {"error": str(e)}

def stream_generate_text(payload):
    """기존 호환성을 위한 래퍼 함수"""
    full_text = ""
    model_info = {}
    
    for text, info in stream_generate_text_generator(payload):
        full_text = text
        model_info = info
    
    return full_text, model_info

def stream_generate_text_with_progress(payload):
    """진행상황을 표시하면서 스트리밍 텍스트 생성"""
    import time
    try:
        response = requests.post(
            f"{API_URL}/generate", 
            json=payload, 
            stream=True,
            headers={'Accept': 'text/event-stream'},
            timeout=300
        )
        
        if response.status_code != 200:
            return f"오류: HTTP {response.status_code}", {"error": "Stream failed"}
        
        full_text = ""
        last_update_time = time.time()
        
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith('data: '):
                try:
                    data_str = line[6:]  # 'data: ' 제거
                    data = json.loads(data_str)
                    if 'error' in data:
                        return data['error'], {"error": data['error']}
                    if 'content' in data and data['content']:
                        full_text += data['content']
                        # 짧은 지연을 통해 실시간 효과 시뮬레이션
                        current_time = time.time()
                        if current_time - last_update_time > 0.05:  # 50ms마다 업데이트
                            time.sleep(0.01)
                            last_update_time = current_time
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
        
        return full_text, {"model_info": {"streaming": True, "complete": True, "length": len(full_text)}}
        
    except Exception as e:
        return f"스트리밍 오류: {str(e)}", {"error": str(e)}

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

# === 뉴스 기능 함수들 ===
def get_latest_news(categories, max_results, time_range):
    """최신 뉴스 조회"""
    try:
        # 카테고리 문자열 처리
        if categories and categories.strip():
            category_param = ",".join([cat.strip() for cat in categories.split(",")])
        else:
            category_param = ""
        
        params = {
            "max_results": max_results,
            "time_range": time_range
        }
        if category_param:
            params["categories"] = category_param
        
        response = requests.get(f"{API_URL}/news/latest", params=params, timeout=30)
        
        error_message = handle_api_error(response)
        if error_message:
            return error_message, "오류 발생"
        
        result = response.json()
        news_list = result.get("news", [])
        
        if not news_list:
            return "검색 결과가 없습니다.", "결과 없음"
        
        # 뉴스 목록 포맷팅
        formatted_news = f"**📰 총 {len(news_list)}개의 최신 뉴스**\n\n"
        for i, news in enumerate(news_list, 1):
            formatted_news += f"### {i}. {news.get('title', '제목 없음')}\n"
            formatted_news += f"**카테고리:** {news.get('category', 'N/A')} | "
            formatted_news += f"**점수:** {news.get('score', 0):.2f}\n"
            formatted_news += f"**URL:** [{news.get('url', '#')[:50]}...]({news.get('url', '#')})\n"
            content = news.get('content', '')
            if content:
                formatted_news += f"**내용 미리보기:** {content[:200]}...\n"
            formatted_news += f"**발행일:** {news.get('published_date', 'N/A')}\n\n---\n\n"
        
        summary = f"카테고리: {result.get('categories', [])} | 시간범위: {result.get('time_range', 'N/A')}"
        return formatted_news, summary
        
    except Exception as e:
        error_msg = f"최신 뉴스 조회 중 오류 발생: {str(e)}"
        return error_msg, "오류"

def search_news(query, category, max_results, time_range):
    """뉴스 검색"""
    if not query.strip():
        return "검색어를 입력해주세요.", "오류"
    
    try:
        params = {
            "query": query,
            "max_results": max_results,
            "time_range": time_range
        }
        if category and category != "전체":
            params["category"] = category
        
        response = requests.get(f"{API_URL}/news/search", params=params, timeout=30)
        
        error_message = handle_api_error(response)
        if error_message:
            return error_message, "오류 발생"
        
        result = response.json()
        news_list = result.get("news", [])
        
        if not news_list:
            return f"'{query}' 검색 결과가 없습니다.", "결과 없음"
        
        # 뉴스 검색 결과 포맷팅
        formatted_news = f"**🔍 '{query}' 검색 결과 ({len(news_list)}개)**\n\n"
        for i, news in enumerate(news_list, 1):
            formatted_news += f"### {i}. {news.get('title', '제목 없음')}\n"
            formatted_news += f"**카테고리:** {news.get('category', 'N/A')} | "
            formatted_news += f"**점수:** {news.get('score', 0):.2f}\n"
            formatted_news += f"**URL:** [{news.get('url', '#')[:50]}...]({news.get('url', '#')})\n"
            content = news.get('content', '')
            if content:
                formatted_news += f"**내용 미리보기:** {content[:200]}...\n"
            formatted_news += "---\n\n"
        
        summary = f"검색어: {query} | 카테고리: {category or '전체'} | 시간범위: {time_range}"
        return formatted_news, summary
        
    except Exception as e:
        error_msg = f"뉴스 검색 중 오류 발생: {str(e)}"
        return error_msg, "오류"

def summarize_news(query, summary_type, max_results, model_key):
    """AI 뉴스 요약 (스트리밍)"""
    if not query.strip():
        yield "요약할 뉴스 주제를 입력해주세요.", "오류", {}
        return
    
    try:
        payload = {
            "query": query,
            "summary_type": summary_type,
            "max_results": max_results
        }
        if model_key and model_key != "기본 모델":
            payload["model_key"] = model_key
        
        response = requests.post(f"{API_URL}/news/summary", json=payload, timeout=300, stream=True)
        
        if response.status_code >= 400:
            error_message = f"API 오류 ({response.status_code}): {response.text}"
            yield error_message, "오류 발생", {}
            return
        
        # 스트리밍 응답 처리
        final_result = None
        for content, status, result in process_streaming_response(response):
            if result:  # 최종 결과
                final_result = result
                articles = result.get("articles", [])
                
                # 참고 기사 목록 생성
                articles_info = f"**📊 분석된 기사 ({len(articles)}개):**\n\n"
                for i, article in enumerate(articles[:5], 1):  # 상위 5개만 표시
                    articles_info += f"{i}. **{article.get('title', '제목 없음')}**\n"
                    if article.get('url'):
                        articles_info += f"   🔗 [기사 링크]({article['url']})\n"
                    articles_info += "\n"
                
                model_info = {
                    "query": result.get("query"),
                    "summary_type": result.get("summary_type"),
                    "total_articles": result.get("total_articles"),
                    "model_info": result.get("model_info", {})
                }
                
                yield content, articles_info, model_info
                return
            else:
                # 중간 진행 상태
                yield content, f"🔄 {status}", {}
        
        # 만약 결과가 없다면
        if not final_result:
            yield "요약 생성에 실패했습니다.", "오류", {}
        
    except Exception as e:
        error_msg = f"뉴스 요약 중 오류 발생: {str(e)}"
        yield error_msg, "오류", {}

def analyze_news_trends(categories, max_results, time_range, model_key):
    """뉴스 트렌드 분석 (스트리밍)"""
    try:
        payload = {
            "max_results": max_results,
            "time_range": time_range
        }
        
        # 카테고리 처리
        if categories and categories.strip():
            category_list = [cat.strip() for cat in categories.split(",")]
            payload["categories"] = category_list
        
        if model_key and model_key != "기본 모델":
            payload["model_key"] = model_key
        
        response = requests.post(f"{API_URL}/news/analysis", json=payload, timeout=300, stream=True)
        
        if response.status_code >= 400:
            error_message = f"API 오류 ({response.status_code}): {response.text}"
            yield error_message, "오류 발생", {}
            return
        
        # 스트리밍 응답 처리
        final_result = None
        category_info_buffer = ""
        
        for content, status, result in process_streaming_response(response):
            if result:  # 최종 결과
                final_result = result
                trend_analysis = result.get("overall_trend", "트렌드 분석 실패")
                category_trends = result.get("category_trends", {})
                
                # 카테고리별 트렌드 포맷팅
                category_info = "**📊 카테고리별 트렌드:**\n\n"
                for category, trend in category_trends.items():
                    category_map = {
                        "politics": "정치", "economy": "경제", 
                        "technology": "기술", "society": "사회"
                    }
                    category_name = category_map.get(category, category)
                    category_info += f"**{category_name}:** {trend}\n\n"
                
                analysis_info = {
                    "total_articles": result.get("total_articles_analyzed"),
                    "categories": result.get("categories"),
                    "time_range": result.get("time_range"),
                    "model_info": result.get("model_info", {})
                }
                
                yield trend_analysis, category_info, analysis_info
                return
            else:
                # 중간 진행 상태
                yield content, f"🔄 {status}", {}
        
        # 만약 결과가 없다면
        if not final_result:
            yield "트렌드 분석에 실패했습니다.", "오류", {}
        
    except Exception as e:
        error_msg = f"트렌드 분석 중 오류 발생: {str(e)}"
        yield error_msg, "오류", {}

def get_news_categories():
    """뉴스 카테고리 조회"""
    try:
        response = requests.get(f"{API_URL}/news/categories", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            categories = result.get("categories", {})
            return list(categories.keys())
        else:
            # 기본 카테고리 반환
            return ["politics", "economy", "technology", "sports", "health", "culture", "society", "international"]
    except:
        return ["politics", "economy", "technology", "sports", "health", "culture", "society", "international"]

# === External-Web RAG 기능 함수들 ===
def external_web_upload_topic(topic, max_results):
    """External-Web RAG: 주제 업로드"""
    if not topic.strip():
        return "주제를 입력해주세요.", "오류"
    
    try:
        payload = {"topic": topic, "max_results": max_results}
        result = make_api_call("external-web/upload-topic", payload)
        
        if "error" in result:
            return result["error"], "오류"
        
        success_msg = f"✅ **업로드 성공!**\n\n"
        success_msg += f"**주제:** {result.get('topic', 'N/A')}\n"
        success_msg += f"**추가된 청크 수:** {result.get('added_chunks', 0)}개\n"
        success_msg += f"**최대 검색 결과:** {result.get('max_results', 0)}개\n\n"
        success_msg += f"**메시지:** {result.get('message', '')}"
        
        status = f"주제: {topic} | 청크: {result.get('added_chunks', 0)}개"
        return success_msg, status
        
    except Exception as e:
        error_msg = f"External-Web 업로드 중 오류 발생: {str(e)}"
        return error_msg, "오류"

def external_web_rag_query(prompt, top_k, model_key):
    """External-Web RAG: 질의응답"""
    if not prompt.strip():
        return "질문을 입력해주세요.", "", "오류"

    try:
        payload = {"prompt": prompt, "top_k": top_k}
        if model_key and model_key != "기본 모델":
            payload["model_key"] = model_key

        result = make_api_call("external-web/rag-query", payload)

        if "error" in result:
            return result["error"], "", "오류"

        # 답변 포맷팅
        response = result.get("response", "응답 없음")

        # 관련 문서 포맷팅
        docs = result.get("relevant_documents", [])
        doc_str = ""
        if docs:
            doc_str = f"**📄 참고 문서 ({len(docs)}개):**\n\n"
            for i, doc in enumerate(docs, 1):
                doc_str += f"### {i}. {doc.get('title', '제목 없음')}\n"
                doc_str += f"**출처:** [{doc.get('source', 'N/A')}]({doc.get('source', '#')})\n"
                content = doc.get('content', '')
                if content:
                    doc_str += f"**내용:** {content[:300]}...\n"
                doc_str += "---\n\n"
        else:
            doc_str = "관련 문서를 찾지 못했습니다."

        # 상태 정보
        model_info = result.get("model_info", {})
        status = f"모델: {model_info.get('model_key', 'N/A')} | 문서: {len(docs)}개 | 소스: External-Web"

        return response, doc_str, status

    except Exception as e:
        error_msg = f"External-Web RAG 질의 중 오류 발생: {str(e)}"
        return error_msg, "", "오류"

def create_progress_html(progress, message, status):
    """진행률을 시각적으로 표시하는 HTML 생성"""
    color = "green" if progress == 100 else "blue" if progress > 50 else "orange"
    return f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="font-weight: bold;">{status}</span>
            <span>{progress}%</span>
        </div>
        <div style="background-color: #f0f0f0; border-radius: 10px; padding: 2px;">
            <div style="background-color: {color}; height: 20px; border-radius: 8px; width: {progress}%; transition: width 0.5s ease;"></div>
        </div>
        <div style="margin-top: 8px; color: #666; font-size: 14px;">{message}</div>
    </div>
    """

def external_web_auto_rag(query, max_results, model_key):
    """External-Web RAG: 자동 질의응답 (스트리밍으로 실시간 진행 상황 표시)"""
    if not query.strip():
        yield "", "질문을 입력해주세요.", "", "오류"
        return

    try:
        payload = {"query": query, "max_results": max_results}
        if model_key and model_key != "기본 모델":
            payload["model_key"] = model_key

        # 스트리밍 요청
        response = requests.post(
            f"{API_URL}/external-web/auto-rag",
            json=payload,
            timeout=300,
            stream=True,
            headers={'Accept': 'text/event-stream'}
        )

        if response.status_code >= 400:
            error_message = f"API 오류 ({response.status_code}): {response.text}"
            yield "", error_message, "", "오류 발생"
            return

        # 스트리밍 응답 처리
        final_result = None
        current_answer = ""
        current_docs = ""
        current_status = "시작 중..."

        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                try:
                    data_str = line[6:]  # "data: " 부분 제거
                    data = json.loads(data_str)

                    status = data.get('status', '')
                    message = data.get('message', '')
                    progress = data.get('progress', 0)

                    if status == 'starting':
                        progress_html = create_progress_html(5, message, "🚀 시작")
                        current_answer = "🔄 자동 RAG 처리를 시작합니다..."
                        current_status = f"🚀 {message}"
                        yield progress_html, current_answer, current_docs, current_status

                    elif status == 'searching':
                        progress_html = create_progress_html(progress, message, "🔍 뉴스 검색")
                        current_answer = "🔍 웹에서 관련 뉴스를 검색하는 중입니다...\n\n잠시만 기다려주세요."
                        current_status = f"🔍 {message}"
                        yield progress_html, current_answer, current_docs, current_status

                    elif status == 'vectorizing':
                        progress_html = create_progress_html(progress, message, "📚 벡터 DB 저장")
                        current_answer = f"✅ 뉴스 검색 완료!\n\n📚 {message}\n\n다음 단계로 진행 중..."
                        current_status = f"📚 {message}"
                        yield progress_html, current_answer, current_docs, current_status

                    elif status == 'generating':
                        progress_html = create_progress_html(progress, message, "🤖 답변 생성")
                        current_answer = "🤖 AI가 수집된 정보를 바탕으로 종합적인 답변을 생성하고 있습니다...\n\n조금만 더 기다려주세요."
                        current_status = f"🤖 {message}"
                        yield progress_html, current_answer, current_docs, current_status

                    elif status == 'finalizing':
                        progress_html = create_progress_html(progress, message, "📝 마무리")
                        current_answer = "📝 답변 생성 완료! 관련 문서 정보를 정리하는 중..."
                        current_status = f"📝 {message}"
                        yield progress_html, current_answer, current_docs, current_status

                    elif status == 'completed':
                        final_result = data
                        break

                    elif status == 'no_results':
                        progress_html = create_progress_html(0, message, "⚠️ 검색 결과 없음")
                        current_answer = f"⚠️ {message}"
                        current_status = "검색 결과 없음"
                        yield progress_html, current_answer, current_docs, current_status
                        return

                    elif status == 'error':
                        error_msg = data.get('message', '알 수 없는 오류')
                        progress_html = create_progress_html(0, error_msg, "❌ 오류 발생")
                        yield progress_html, f"❌ 오류: {error_msg}", "", "오류 발생"
                        return

                except json.JSONDecodeError:
                    continue

        # 최종 결과 처리
        if final_result:
            response_text = final_result.get("response", "응답 없음")
            docs = final_result.get("relevant_documents", [])

            # 관련 문서 포맷팅
            doc_str = ""
            if docs:
                doc_str = f"**🔍 자동 검색된 관련 문서 ({len(docs)}개):**\n\n"
                for i, doc in enumerate(docs, 1):
                    doc_str += f"### {i}. {doc.get('title', '제목 없음')}\n"
                    doc_str += f"**출처:** [{doc.get('source', 'N/A')}]({doc.get('source', '#')})\n"
                    content = doc.get('content', '')
                    if content:
                        doc_str += f"**내용:** {content[:300]}...\n"
                    doc_str += "---\n\n"
            else:
                doc_str = "관련 문서를 찾지 못했습니다."

            # 최종 완료 상태
            model_info = final_result.get("model_info", {})
            added_chunks = final_result.get("added_chunks", 0)
            final_status = f"✅ 완료! | 모델: {model_info.get('model_key', 'N/A')} | 추가 청크: {added_chunks}개 | 문서: {len(docs)}개"

            progress_html = create_progress_html(100, "처리 완료!", "✅ 완료")
            yield progress_html, response_text, doc_str, final_status
        else:
            progress_html = create_progress_html(0, "처리가 완료되지 않았습니다.", "❌ 오류")
            yield progress_html, "처리가 완료되지 않았습니다.", "", "오류"

    except Exception as e:
        error_msg = f"External-Web 자동 RAG 중 오류 발생: {str(e)}"
        progress_html = create_progress_html(0, error_msg, "❌ 오류")
        yield progress_html, error_msg, "", "오류"

# === Internal-DBMS RAG 기능 함수들 ===
def internal_db_get_tables():
    """Internal-DB: 테이블 목록 조회"""
    try:
        result = make_api_call("internal-db/tables", {}, method="get")

        if "error" in result:
            return result["error"], "오류", []

        tables = result.get("tables", [])
        if not tables:
            return "사용 가능한 테이블이 없습니다.", "결과 없음", []

        formatted_tables = f"**📋 사용 가능한 테이블 ({len(tables)}개):**\n\n"
        for i, table in enumerate(tables, 1):
            formatted_tables += f"{i}. **{table}**\n"

        status = f"총 {len(tables)}개 테이블"

        # 테이블 드롭다운 선택지 업데이트용
        table_choices = ["테이블을 선택하세요"] + tables

        return formatted_tables, status, table_choices

    except Exception as e:
        error_msg = f"테이블 조회 중 오류 발생: {str(e)}"
        return error_msg, "오류", ["테이블을 선택하세요"]

def internal_db_simulate_table_data(table_name, limit):
    """시뮬레이션 테이블 데이터 생성"""
    if table_name == "knowledge":
        data = [
            {
                "id": 1,
                "term": "어텐션 메커니즘",
                "description": "어텐션은 입력의 중요한 부분에 가중치를 부여해 정보를 통합하는 기법이다.",
                "role": "입력 토큰 간 상호연관성을 계산하며 정보 흐름을 개선한다.",
                "details": "Transformer의 핵심 구성요소로 번역·요약 등에서 성능을 끌어올린다.",
                "updated_at": "2024-01-01 00:00:00"
            },
            {
                "id": 2,
                "term": "Self-Attention",
                "description": "Self-Attention은 동일 시퀀스 내 토큰들이 서로를 참조하여 가중합을 구한다.",
                "role": "장기 의존성 문제를 완화하고 각 토큰의 전역 문맥 파악을 돕는다.",
                "details": "멀티헤드로 다양한 표현 공간에서 주의를 분산해 학습을 안정화한다.",
                "updated_at": "2024-01-01 00:00:00"
            },
            {
                "id": 3,
                "term": "FAISS",
                "description": "FAISS는 대규모 벡터에 대한 빠른 유사도 검색을 제공한다.",
                "role": "대규모 임베딩 인덱싱과 빠른 검색을 제공한다.",
                "details": "Facebook AI Research에서 개발되었고 CPU/GPU 백엔드를 제공한다.",
                "updated_at": "2024-01-01 00:00:00"
            }
        ]
        columns = ["id", "term", "description", "role", "details", "updated_at"]
    elif table_name == "products":
        data = [
            {
                "id": 1,
                "name": "QA 시스템 Pro",
                "category": "AI Software",
                "description": "RAG 기반 질의응답 시스템으로 대규모 문서에서 정확한 답변을 제공합니다.",
                "price": 299.99,
                "features": "자동 인덱싱, 실시간 검색, 다국어 지원, API 제공"
            },
            {
                "id": 2,
                "name": "벡터 검색 엔진",
                "category": "Database",
                "description": "고성능 벡터 유사도 검색을 지원하는 전문 데이터베이스입니다.",
                "price": 499.99,
                "features": "FAISS 통합, 분산 처리, REST API, 실시간 업데이트"
            },
            {
                "id": 3,
                "name": "문서 임베딩 도구",
                "category": "AI Tools",
                "description": "다양한 형식의 문서를 고품질 벡터로 변환하는 도구입니다.",
                "price": 199.99,
                "features": "다중 형식 지원, 배치 처리, 클라우드 연동, 자동 청킹"
            }
        ]
        columns = ["id", "name", "category", "description", "price", "features"]
    elif table_name == "users":
        data = [
            {
                "id": 1,
                "username": "admin",
                "email": "admin@company.com",
                "role": "관리자",
                "created_at": "2024-01-01 09:00:00",
                "last_login": "2024-01-15 14:30:00"
            },
            {
                "id": 2,
                "username": "analyst",
                "email": "analyst@company.com",
                "role": "분석가",
                "created_at": "2024-01-02 10:15:00",
                "last_login": "2024-01-15 13:45:00"
            },
            {
                "id": 3,
                "username": "viewer",
                "email": "viewer@company.com",
                "role": "조회자",
                "created_at": "2024-01-03 11:30:00",
                "last_login": "2024-01-14 16:20:00"
            }
        ]
        columns = ["id", "username", "email", "role", "created_at", "last_login"]
    elif table_name == "orders":
        data = [
            {
                "id": 1,
                "user_id": 2,
                "product_id": 1,
                "quantity": 1,
                "total_price": 299.99,
                "status": "완료",
                "created_at": "2024-01-10 14:30:00"
            },
            {
                "id": 2,
                "user_id": 3,
                "product_id": 2,
                "quantity": 2,
                "total_price": 999.98,
                "status": "진행중",
                "created_at": "2024-01-12 09:15:00"
            },
            {
                "id": 3,
                "user_id": 1,
                "product_id": 3,
                "quantity": 1,
                "total_price": 199.99,
                "status": "대기",
                "created_at": "2024-01-14 16:45:00"
            }
        ]
        columns = ["id", "user_id", "product_id", "quantity", "total_price", "status", "created_at"]
    else:
        return f"테이블 '{table_name}'을 찾을 수 없습니다.", "테이블 없음"

    # limit 적용
    displayed_data = data[:limit]

    # HTML 테이블 생성
    html_output = f"""
    <div style="font-family: Arial, sans-serif; margin: 10px 0;">
        <h3>📊 테이블: {table_name}</h3>
        <p><strong>모드:</strong> 시뮬레이션 (데모 데이터)</p>
        <p><strong>전체 행 수:</strong> {len(data)}개 | <strong>표시 행 수:</strong> {len(displayed_data)}개</p>
    </div>
    """

    if displayed_data:
        # 테이블 헤더
        html_output += """
        <div style="overflow-x: auto; margin: 10px 0;">
            <table style="border-collapse: collapse; width: 100%; border: 1px solid #ddd;">
                <thead style="background-color: #f5f5f5;">
                    <tr>
        """

        for col in columns:
            html_output += f'<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">{col}</th>'

        html_output += "</tr></thead><tbody>"

        # 테이블 데이터
        for i, row in enumerate(displayed_data):
            bg_color = "#f9f9f9" if i % 2 == 0 else "white"
            html_output += f'<tr style="background-color: {bg_color};">'

            for col in columns:
                value = str(row.get(col, "")) if row.get(col) is not None else ""
                # 긴 텍스트는 자르기
                if len(value) > 100:
                    value = value[:100] + "..."
                html_output += f'<td style="border: 1px solid #ddd; padding: 8px; word-wrap: break-word;">{value}</td>'

            html_output += "</tr>"

        html_output += "</tbody></table></div>"
    else:
        html_output += "<p>⚠️ 테이블에 데이터가 없습니다.</p>"

    status = f"테이블: {table_name} | 모드: 시뮬레이션 | 행: {len(displayed_data)}/{len(data)}"
    return html_output, status

def internal_db_view_table(table_name, limit):
    """Internal-DB: 선택한 테이블 내용 보기"""
    if not table_name or table_name == "테이블을 선택하세요":
        return "테이블을 선택해주세요.", "테이블 미선택"

    try:
        # 시뮬레이션 데이터 사용 (API 엔드포인트 이슈 해결까지)
        if table_name in ["knowledge", "products", "users", "orders"]:
            return internal_db_simulate_table_data(table_name, limit)

        # 지원되지 않는 테이블
        return f"테이블 '{table_name}'은(는) 아직 지원되지 않습니다.", "미지원 테이블"

    except Exception as e:
        return f"테이블 조회 중 오류가 발생했습니다: {str(e)}", "오류"

def internal_db_ingest(table_name, save_name, simulate, id_col, title_col, text_cols):
    """Internal-DB: 테이블 인제스트"""
    if not table_name.strip():
        return "테이블 이름을 입력해주세요.", "오류"
    
    try:
        payload = {
            "table": table_name,
            "save_name": save_name or table_name,
            "simulate": simulate
        }
        
        # 선택적 컬럼 정보 추가
        if id_col and id_col.strip():
            payload["id_col"] = id_col
        if title_col and title_col.strip():
            payload["title_col"] = title_col
        if text_cols and text_cols.strip():
            payload["text_cols"] = [col.strip() for col in text_cols.split(",")]
        
        result = make_api_call("internal-db/ingest", payload)
        
        if "error" in result:
            return result["error"], "오류"
        
        success_msg = f"✅ **인제스트 성공!**\n\n"
        success_msg += f"**테이블:** {result.get('table', 'N/A')}\n"
        success_msg += f"**저장 경로:** {result.get('save_dir', 'N/A')}\n"
        success_msg += f"**처리된 행 수:** {result.get('rows', 0)}개\n"
        success_msg += f"**생성된 청크 수:** {result.get('chunks', 0)}개\n"
        success_msg += f"**시뮬레이션 모드:** {'예' if result.get('simulate') else '아니오'}\n\n"
        
        schema = result.get("schema", {})
        if schema:
            success_msg += f"**스키마 정보:**\n"
            success_msg += f"- ID 컬럼: {schema.get('id_col', 'N/A')}\n"
            success_msg += f"- 제목 컬럼: {schema.get('title_col', 'N/A')}\n"
            success_msg += f"- 텍스트 컬럼: {', '.join(schema.get('text_cols', []))}\n"
        
        status = f"테이블: {table_name} | 행: {result.get('rows', 0)} | 청크: {result.get('chunks', 0)}"
        return success_msg, status
        
    except Exception as e:
        error_msg = f"인제스트 중 오류 발생: {str(e)}"
        return error_msg, "오류"

def internal_db_query(save_name, question, top_k, margin):
    """Internal-DB: 질의응답"""
    if not question.strip():
        return "질문을 입력해주세요.", "", "오류"
    
    try:
        payload = {
            "save_name": save_name,
            "question": question,
            "top_k": top_k,
            "margin": margin
        }
        
        result = make_api_call("internal-db/query", payload)
        
        if "error" in result:
            return result["error"], "", "오류"
        
        # 답변 포맷팅
        answer = result.get("answer", "응답 없음")
        
        # 출처 정보 포맷팅
        sources = result.get("sources", [])
        source_str = ""
        if sources:
            source_str = f"**🔍 참고 출처 ({len(sources)}개):**\n\n"
            for source in sources:
                marker = source.get("marker", "S?")
                title = source.get("title", "제목 없음")
                content = source.get("content", "")
                score = source.get("score", 0)
                
                source_str += f"### {marker}. {title}\n"
                source_str += f"**유사도 점수:** {score:.4f}\n"
                if content:
                    source_str += f"**내용:** {content}\n"
                source_str += "---\n\n"
        else:
            source_str = "참고할 출처를 찾지 못했습니다."
        
        # 상태 정보
        status = f"인덱스: {save_name} | 출처: {len(sources)}개 | top_k: {top_k} | margin: {margin}"
        
        return answer, source_str, status
        
    except Exception as e:
        error_msg = f"Internal-DB 질의 중 오류 발생: {str(e)}"
        return error_msg, "", "오류"

def internal_db_get_status():
    """Internal-DB: 상태 조회"""
    try:
        result = make_api_call("internal-db/status", {}, method="get")
        
        if "error" in result:
            return result["error"], "오류"
        
        faiss_indices = result.get("faiss_indices", [])
        cache_keys = result.get("cache_keys", [])
        
        status_msg = f"**📊 Internal-DB 상태 정보**\n\n"
        status_msg += f"**디스크 저장 인덱스:** {len(faiss_indices)}개\n"
        for i, index in enumerate(faiss_indices, 1):
            status_msg += f"  {i}. {index}\n"
        
        status_msg += f"\n**메모리 캐시 인덱스:** {len(cache_keys)}개\n"
        for i, key in enumerate(cache_keys, 1):
            status_msg += f"  {i}. {key}\n"
        
        if not faiss_indices and not cache_keys:
            status_msg += "\n⚠️ 사용 가능한 인덱스가 없습니다. 먼저 인제스트를 수행해주세요."
        
        summary = f"디스크: {len(faiss_indices)}개 | 캐시: {len(cache_keys)}개"
        return status_msg, summary
        
    except Exception as e:
        error_msg = f"상태 조회 중 오류 발생: {str(e)}"
        return error_msg, "오류"

# === 음성 기능 함수들 ===
def text_to_speech_gradio(text, language, slow):
    """텍스트를 음성으로 변환 (Gradio용)"""
    if not text.strip():
        return None, "텍스트를 입력해주세요."

    try:
        payload = {
            "text": text,
            "language": language,
            "slow": slow
        }

        response = requests.post(f"{API_URL}/speech/text-to-speech", json=payload, timeout=30)

        if response.status_code == 200:
            # 임시 파일로 저장
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_file.write(response.content)
            temp_file.close()

            return temp_file.name, f"✅ 음성 합성 완료! (언어: {language})"
        else:
            error_detail = response.json().get("detail", "알 수 없는 오류")
            return None, f"❌ 음성 합성 실패: {error_detail}"

    except Exception as e:
        return None, f"❌ 오류 발생: {str(e)}"

def speech_to_text_gradio(audio_file, prefer_whisper):
    """음성을 텍스트로 변환 (Gradio용)"""
    if audio_file is None:
        return "", "음성 파일을 업로드해주세요."

    try:
        with open(audio_file, 'rb') as f:
            files = {'audio_file': f}
            data = {'prefer_whisper': prefer_whisper}

            response = requests.post(f"{API_URL}/speech/speech-to-text", files=files, data=data, timeout=60)

        if response.status_code == 200:
            result = response.json()
            if result["success"]:
                text = result["text"]
                method = result.get("method", "unknown")
                confidence = result.get("confidence", 0.0)
                language = result.get("language", "unknown")

                status = f"✅ 음성 인식 완료! (방법: {method}, 신뢰도: {confidence:.2f}, 언어: {language})"
                return text, status
            else:
                return "", f"❌ 음성 인식 실패: {result.get('error', '알 수 없는 오류')}"
        else:
            error_detail = response.json().get("detail", "알 수 없는 오류")
            return "", f"❌ 음성 인식 실패: {error_detail}"

    except Exception as e:
        return "", f"❌ 오류 발생: {str(e)}"

def voice_chat_gradio(text, model_key, voice_language, voice_slow):
    """음성 채팅 (Gradio용)"""
    if not text.strip():
        return None, "텍스트를 입력해주세요.", ""

    try:
        payload = {
            "text": text,
            "model_key": model_key if model_key != "기본 모델" else None,
            "voice_language": voice_language,
            "voice_slow": voice_slow
        }

        response = requests.post(f"{API_URL}/speech/voice-chat", json=payload, timeout=120)

        if response.status_code == 200:
            response_text = response.headers.get('X-Response-Text', '응답 텍스트를 가져올 수 없습니다.')

            # 임시 파일로 저장
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_file.write(response.content)
            temp_file.close()

            status = f"✅ 음성 채팅 완료! (모델: {model_key}, 언어: {voice_language})"
            return temp_file.name, response_text, status
        else:
            error_detail = response.json().get("detail", "알 수 없는 오류")
            return None, "", f"❌ 음성 채팅 실패: {error_detail}"

    except Exception as e:
        return None, "", f"❌ 오류 발생: {str(e)}"

def get_speech_service_status():
    """음성 서비스 상태 조회"""
    try:
        response = requests.get(f"{API_URL}/speech/status", timeout=10)

        if response.status_code == 200:
            result = response.json()

            status_text = f"""**🎤 음성 서비스 상태**

**Whisper (음성인식):** {'✅ 사용 가능' if result.get('whisper_available') else '❌ 사용 불가'}
**Google STT:** {'✅ 사용 가능' if result.get('google_stt_available') else '❌ 사용 불가'}
**gTTS (음성합성):** {'✅ 사용 가능' if result.get('gtts_available') else '❌ 사용 불가'}
**마이크:** {'✅ 감지됨' if result.get('microphone_available') else '❌ 감지 안됨'}
**지원 언어 수:** {result.get('supported_languages', 0)}개
**전체 상태:** {result.get('status', 'unknown')}
"""
            return status_text, "상태 조회 완료"
        else:
            return "음성 서비스 상태를 가져올 수 없습니다.", "오류"

    except Exception as e:
        return f"오류 발생: {str(e)}", "오류"

def get_streaming_tts_status():
    """스트리밍 TTS 서비스 상태 조회"""
    try:
        response = requests.get(f"{API_URL}/speech/streaming-tts/status", timeout=10)

        if response.status_code == 200:
            result = response.json()

            status_text = f"""**🎯 실시간 스트리밍 TTS 상태**

**스트리밍 TTS:** {'✅ 사용 가능' if result.get('streaming_tts_available') else '❌ 사용 불가'}
**문장 기반 TTS:** {'✅ 사용 가능' if result.get('sentence_tts_available') else '❌ 사용 불가'}
**음성 서비스:** {'✅ 사용 가능' if result.get('speech_service_available') else '❌ 사용 불가'}
**Whisper:** {'✅ 사용 가능' if result.get('whisper_available') else '❌ 사용 불가'}
**gTTS:** {'✅ 사용 가능' if result.get('gtts_available') else '❌ 사용 불가'}

**지원 기능:**
"""
            for feature in result.get('supported_features', []):
                status_text += f"• {feature}\n"

            status_text += f"\n**지원 언어:** {len(result.get('supported_languages', {}))}개"
            status_text += f"\n**전체 상태:** {result.get('status', 'unknown')}"

            return status_text, "스트리밍 TTS 상태 조회 완료"
        else:
            return "스트리밍 TTS 상태를 가져올 수 없습니다.", "오류"

    except Exception as e:
        return f"오류 발생: {str(e)}", "오류"

def streaming_generate_with_voice_gradio(prompt, model_key, voice_language, voice_slow, read_partial):
    """실시간 스트리밍 텍스트 생성 및 음성 읽기 (Gradio용)"""
    if not prompt.strip():
        return "프롬프트를 입력해주세요.", "", "오류"

    try:
        payload = {
            "prompt": prompt,
            "model_key": model_key if model_key != "기본 모델" else None,
            "voice_language": voice_language,
            "voice_slow": voice_slow,
            "read_partial": read_partial
        }

        # 참고: Gradio에서는 스트리밍 응답을 직접 처리하기 어려우므로
        # 사용자에게 전용 페이지로 안내
        message = f"""🎯 **실시간 스트리밍 음성 기능은 전용 페이지에서 이용하세요!**

**프롬프트:** {prompt}
**모델:** {model_key}
**언어:** {voice_language}
**설정:** 느린음성={voice_slow}, 부분읽기={read_partial}

**전용 페이지 링크:** [실시간 스트리밍 음성 페이지](/streaming-voice)

전용 페이지에서는 다음 기능을 제공합니다:
• ⚡ AI가 텍스트 생성하는 동시에 완성된 문장을 실시간 음성으로 읽기
• 🎵 음성 대기열 및 자동 재생
• 📊 실시간 진행 상황 표시
• 🔊 개별 문장 음성 재생 제어

더 나은 경험을 위해 전용 페이지를 이용해주세요!"""

        return message, "전용 페이지 이용 권장", f"설정: {model_key}, {voice_language}"

    except Exception as e:
        return f"❌ 오류 발생: {str(e)}", "", "오류"

def update_model_list():
    """UI가 로드될 때 서버에서 모델 목록을 동적으로 가져옵니다."""
    logger.info("UI: 모델 목록을 서버에서 가져오는 중...")
    
    # 정적 모델 목록 (서버와 동기화)
    default_models = ["qwen2.5-7b", "llama3.1-8b", "gemma-3-4b"]
    choices = ["기본 모델"] + default_models
    
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        if response.status_code == 200:
            server_models = list(response.json().get("supported_models", {}).keys())
            if server_models:
                choices = ["기본 모델"] + server_models
                logger.info(f"UI: 서버에서 모델 목록 가져오기 성공: {server_models}")
            else:
                logger.warning("UI: 서버에서 빈 모델 목록을 받았습니다. 기본 목록 사용")
        else:
            logger.warning(f"UI: 서버 응답 실패 (HTTP {response.status_code}). 기본 목록 사용")
    except requests.exceptions.RequestException as e:
        logger.warning(f"UI: 서버 연결 실패, 기본 모델 목록 사용: {e}")
    
    logger.info(f"UI: 최종 모델 선택지: {choices}")
    return (
        gr.Dropdown(choices=choices, value="기본 모델"),
        gr.Dropdown(choices=choices, value="기본 모델"),
        gr.Dropdown(choices=choices, value="기본 모델"),
        gr.Dropdown(choices=choices, value="기본 모델"),  # 뉴스 요약용
        gr.Dropdown(choices=choices, value="기본 모델"),  # 뉴스 분석용
        gr.Dropdown(choices=choices, value="기본 모델")   # Auto RAG용
    )

# --- Gradio UI 구성 ---
with gr.Blocks(theme=gr.themes.Soft(), title="LLM 서버 UI") as gradio_ui:
    gr.Markdown("# 🤖 LLM FastAPI 서버 UI")
    gr.Markdown("Gradio 인터페이스를 통해 서버의 LLM 기능을 쉽게 사용해보세요.")
    
    with gr.Tabs():
        # 1. 텍스트 생성 탭
        with gr.TabItem("📝 텍스트 생성"):
            gr.Markdown("### 💡 실시간 스트리밍을 원하시면 [전용 스트리밍 페이지](/stream)를 이용해주세요!")
            with gr.Row():
                with gr.Column(scale=2):
                    gen_prompt = gr.Textbox(lines=5, label="프롬프트", placeholder="인공지능의 미래에 대해 짧은 글을 써줘.")
                    gen_model_select = gr.Dropdown(
                        label="모델 선택", 
                        choices=["기본 모델", "qwen2.5-7b", "llama3.1-8b", "gemma-3-4b"],
                        value="기본 모델"
                    )
                    gen_streaming = gr.Checkbox(label="스트리밍 모드 (완성 후 일괄 표시)", value=False)
                    gen_button = gr.Button("생성하기", variant="primary")
                with gr.Column(scale=3):
                    gen_output = gr.Textbox(lines=10, label="생성된 텍스트", interactive=False)
                    gen_model_info = gr.JSON(label="사용된 모델 정보 / 오류 상세")

        # 2. 채팅 탭
        with gr.TabItem("💬 채팅"):
            chat_model_select = gr.Dropdown(
                label="채팅 모델 선택", 
                choices=["기본 모델", "qwen2.5-7b", "llama3.1-8b", "gemma-3-4b"],
                value="기본 모델"
            )
            gr.ChatInterface(
                fn=chat_with_bot,
                additional_inputs=[chat_model_select],
                chatbot=gr.Chatbot(height=400, label="채팅창", type="messages"),
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
                    rag_model_select = gr.Dropdown(
                        label="모델 선택", 
                        choices=["기본 모델", "qwen2.5-7b", "llama3.1-8b", "gemma-3-4b"],
                        value="기본 모델"
                    )
                    rag_button = gr.Button("질문하기", variant="primary")
                with gr.Column(scale=3):
                    rag_answer = gr.Textbox(lines=5, label="답변", interactive=False)
                    rag_model_info_output = gr.JSON(label="사용된 모델 정보 / 오류 상세")
            rag_docs = gr.Markdown(label="참고 문서")

        # 4. 뉴스 기능 탭 (NEW!)
        with gr.TabItem("📰 뉴스 기능 (NEW!)"):
            gr.Markdown("### 🆕 Tavily 기반 실시간 뉴스 검색 및 AI 요약 기능")
            gr.Markdown("**주의**: 뉴스 기능을 사용하려면 `.env` 파일에 `TAVILY_API_KEY`를 설정해주세요.")
            
            with gr.Tabs():
                # 4-1. 최신 뉴스 조회
                with gr.TabItem("🔥 최신 뉴스"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            latest_categories = gr.Textbox(
                                label="카테고리 (쉼표로 구분)", 
                                placeholder="technology,economy,politics",
                                value="technology,economy"
                            )
                            latest_max_results = gr.Slider(
                                minimum=1, maximum=20, value=8, step=1, 
                                label="최대 뉴스 수"
                            )
                            latest_time_range = gr.Radio(
                                choices=["d", "w", "m"], value="d",
                                label="시간 범위 (d=1일, w=1주, m=1달)"
                            )
                            latest_button = gr.Button("최신 뉴스 조회", variant="primary")
                        
                        with gr.Column(scale=2):
                            latest_output = gr.Markdown(label="최신 뉴스 목록")
                            latest_summary = gr.Textbox(label="조회 정보", interactive=False)

                # 4-2. 뉴스 검색
                with gr.TabItem("🔍 뉴스 검색"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            search_query = gr.Textbox(
                                label="검색어", 
                                placeholder="ChatGPT, 삼성전자, 인공지능"
                            )
                            search_category = gr.Dropdown(
                                choices=["전체", "politics", "economy", "technology", "sports", "health", "culture", "society", "international"],
                                value="전체",
                                label="카테고리"
                            )
                            search_max_results = gr.Slider(
                                minimum=1, maximum=15, value=5, step=1,
                                label="최대 검색 결과"
                            )
                            search_time_range = gr.Radio(
                                choices=["d", "w", "m"], value="d",
                                label="시간 범위"
                            )
                            search_button = gr.Button("뉴스 검색", variant="primary")
                        
                        with gr.Column(scale=2):
                            search_output = gr.Markdown(label="검색 결과")
                            search_summary = gr.Textbox(label="검색 정보", interactive=False)

                # 4-3. AI 뉴스 요약
                with gr.TabItem("🤖 AI 뉴스 요약"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            summary_query = gr.Textbox(
                                label="요약할 뉴스 주제", 
                                placeholder="ChatGPT, 부동산 정책, 전기차"
                            )
                            summary_type = gr.Radio(
                                choices=["brief", "comprehensive", "analysis"],
                                value="comprehensive",
                                label="요약 타입 (간단/포괄적/심층분석)"
                            )
                            summary_max_results = gr.Slider(
                                minimum=1, maximum=10, value=5, step=1,
                                label="분석할 뉴스 수"
                            )
                            summary_model = gr.Dropdown(
                                label="요약 모델", 
                                choices=["기본 모델", "qwen2.5-7b", "llama3.1-8b", "gemma-3-4b"],
                                value="기본 모델"
                            )
                            summary_button = gr.Button("AI 뉴스 요약", variant="primary")
                        
                        with gr.Column(scale=2):
                            summary_output = gr.Markdown(label="AI 뉴스 요약")
                            summary_articles = gr.Markdown(label="참고 기사")
                            summary_info = gr.JSON(label="요약 정보")

                # 4-4. 뉴스 트렌드 분석
                with gr.TabItem("📊 트렌드 분석"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            trend_categories = gr.Textbox(
                                label="분석할 카테고리 (쉼표로 구분)", 
                                placeholder="politics,economy,technology,society",
                                value="politics,economy,technology"
                            )
                            trend_max_results = gr.Slider(
                                minimum=10, maximum=30, value=20, step=5,
                                label="분석할 총 뉴스 수"
                            )
                            trend_time_range = gr.Radio(
                                choices=["d", "w"], value="d",
                                label="분석 기간"
                            )
                            trend_model = gr.Dropdown(
                                label="분석 모델", 
                                choices=["기본 모델", "qwen2.5-7b", "llama3.1-8b", "gemma-3-4b"],
                                value="기본 모델"
                            )
                            trend_button = gr.Button("트렌드 분석", variant="primary")
                        
                        with gr.Column(scale=2):
                            trend_output = gr.Markdown(label="전체 트렌드 분석")
                            trend_categories_output = gr.Markdown(label="카테고리별 트렌드")
                            trend_info = gr.JSON(label="분석 정보")

        # 5. External-Web RAG 탭 (NEW!)
        with gr.TabItem("🌐 External-Web RAG (NEW!)"):
            gr.Markdown("### 🆕 외부 웹 검색 기반 RAG 시스템")
            gr.Markdown("웹에서 정보를 수집하여 질의응답하는 시스템입니다.")

            with gr.Tabs():
                # 5-1. 자동 RAG ⭐ NEW!
                with gr.TabItem("🚀 Auto RAG (추천!)"):
                    gr.Markdown("### ⚡ 완전 자동화된 RAG")
                    gr.Markdown("**질문만 하면 자동으로 관련 뉴스를 검색하고 벡터 DB화하여 답변을 제공합니다.**")

                    with gr.Row():
                        with gr.Column(scale=1):
                            auto_query = gr.Textbox(
                                lines=3,
                                label="질문",
                                placeholder="예: 삼성전자 AI 반도체 최신 동향은?\n인공지능 투자 현황은 어떻습니까?\nChatGPT 관련 최신 소식을 알려주세요."
                            )
                            auto_max_results = gr.Slider(
                                minimum=5, maximum=25, value=15, step=5,
                                label="검색할 뉴스 수"
                            )
                            auto_model = gr.Dropdown(
                                label="사용할 모델",
                                choices=["기본 모델", "qwen2.5-7b", "llama3.1-8b", "gemma-3-4b"],
                                value="기본 모델"
                            )
                            auto_button = gr.Button("🚀 자동 RAG 실행", variant="primary", size="lg")

                        with gr.Column(scale=2):
                            auto_progress = gr.HTML(label="📊 진행 상황", visible=True)
                            auto_answer = gr.Markdown(label="🤖 AI 답변")
                            auto_docs = gr.Markdown(label="📰 자동 검색된 관련 뉴스")
                            auto_status = gr.Textbox(label="🔄 처리 상태", interactive=False)

                # 5-2. 주제 업로드
                with gr.TabItem("📤 주제 업로드"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            ext_upload_topic = gr.Textbox(
                                label="업로드할 주제",
                                placeholder="예: 인공지능 ChatGPT, 삼성전자 반도체"
                            )
                            ext_upload_max_results = gr.Slider(
                                minimum=5, maximum=30, value=20, step=5,
                                label="최대 검색 결과 수"
                            )
                            ext_upload_button = gr.Button("주제 업로드", variant="primary")

                        with gr.Column(scale=2):
                            ext_upload_output = gr.Markdown(label="업로드 결과")
                            ext_upload_status = gr.Textbox(label="상태 정보", interactive=False)

                # 5-3. 질의응답
                with gr.TabItem("❓ 질의응답"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            ext_query_prompt = gr.Textbox(
                                lines=3,
                                label="질문",
                                placeholder="업로드한 주제에 대해 질문하세요"
                            )
                            ext_query_top_k = gr.Slider(
                                minimum=1, maximum=10, value=5, step=1,
                                label="검색할 문서 수 (top_k)"
                            )
                            ext_query_model = gr.Dropdown(
                                label="사용할 모델",
                                choices=["기본 모델", "qwen2.5-7b", "llama3.1-8b", "gemma-3-4b"],
                                value="기본 모델"
                            )
                            ext_query_button = gr.Button("질문하기", variant="primary")

                        with gr.Column(scale=2):
                            ext_query_answer = gr.Markdown(label="답변")
                            ext_query_docs = gr.Markdown(label="참고 문서")
                            ext_query_status = gr.Textbox(label="상태 정보", interactive=False)

        # 6. 음성 기능 탭 (NEW!)
        with gr.TabItem("🎤 음성 기능 (NEW!)"):
            gr.Markdown("### 🆕 AI 음성 대화 시스템")
            gr.Markdown("**음성으로 AI와 대화하고 음성으로 답변을 받아보세요!**")
            gr.Markdown("**🎯 추천**: 더 나은 경험을 위해 [전용 음성 채팅 페이지](/voice)를 이용해주세요!")

            with gr.Tabs():
                # 6-1. 음성 서비스 상태
                with gr.TabItem("📊 서비스 상태"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            speech_status_button = gr.Button("🔍 음성 서비스 상태 확인", variant="primary")

                        with gr.Column(scale=2):
                            speech_status_output = gr.Markdown(label="음성 서비스 상태")
                            speech_status_info = gr.Textbox(label="상태 정보", interactive=False)

                # 6-2. 텍스트 → 음성 (TTS)
                with gr.TabItem("📢 텍스트→음성"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            tts_text = gr.Textbox(
                                lines=4,
                                label="음성으로 변환할 텍스트",
                                placeholder="안녕하세요! 저는 AI 어시스턴트입니다."
                            )
                            tts_language = gr.Dropdown(
                                choices=["ko", "en", "ja", "zh", "es", "fr", "de"],
                                value="ko",
                                label="음성 언어"
                            )
                            tts_slow = gr.Checkbox(label="느린 음성", value=False)
                            tts_button = gr.Button("🔊 음성으로 변환", variant="primary")

                        with gr.Column(scale=2):
                            tts_audio_output = gr.Audio(label="생성된 음성", type="filepath")
                            tts_status = gr.Textbox(label="변환 상태", interactive=False)

                # 6-3. 음성 → 텍스트 (STT)
                with gr.TabItem("🎙️ 음성→텍스트"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            stt_audio_input = gr.Audio(
                                label="음성 파일 업로드",
                                type="filepath"
                            )
                            stt_prefer_whisper = gr.Checkbox(
                                label="Whisper 우선 사용 (더 정확함)",
                                value=True
                            )
                            stt_button = gr.Button("📝 텍스트로 변환", variant="primary")

                        with gr.Column(scale=2):
                            stt_text_output = gr.Textbox(
                                lines=4,
                                label="인식된 텍스트",
                                interactive=False
                            )
                            stt_status = gr.Textbox(label="인식 상태", interactive=False)

                # 6-4. 음성 채팅
                with gr.TabItem("💬 음성 채팅"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            voice_chat_text = gr.Textbox(
                                lines=3,
                                label="채팅 메시지",
                                placeholder="AI에게 질문해보세요!"
                            )
                            voice_chat_model = gr.Dropdown(
                                label="사용할 모델",
                                choices=["기본 모델", "qwen2.5-7b", "llama3.1-8b", "gemma-3-4b"],
                                value="기본 모델"
                            )
                            voice_chat_language = gr.Dropdown(
                                choices=["ko", "en", "ja", "zh"],
                                value="ko",
                                label="응답 음성 언어"
                            )
                            voice_chat_slow = gr.Checkbox(label="느린 음성", value=False)
                            voice_chat_button = gr.Button("🎤 음성 채팅", variant="primary")

                        with gr.Column(scale=2):
                            voice_chat_audio = gr.Audio(label="AI 음성 응답", type="filepath")
                            voice_chat_response = gr.Textbox(
                                lines=4,
                                label="AI 응답 텍스트",
                                interactive=False
                            )
                            voice_chat_status = gr.Textbox(label="채팅 상태", interactive=False)

                # 6-5. 실시간 스트리밍 음성 (NEW!)
                with gr.TabItem("🎯 스트리밍 음성 (NEW!)"):
                    gr.Markdown("### ⚡ 실시간 스트리밍 TTS")
                    gr.Markdown("**AI가 텍스트를 생성하는 동시에 완성된 문장을 실시간으로 읽어줍니다!**")
                    gr.Markdown("**🎯 최고의 경험을 위해 [전용 스트리밍 페이지](/streaming-voice)를 이용하세요!**")

                    with gr.Tabs():
                        # 6-5-1. 스트리밍 TTS 상태
                        with gr.TabItem("📊 스트리밍 상태"):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    streaming_status_button = gr.Button("🔍 스트리밍 TTS 상태 확인", variant="primary")

                                with gr.Column(scale=2):
                                    streaming_status_output = gr.Markdown(label="스트리밍 TTS 상태")
                                    streaming_status_info = gr.Textbox(label="상태 정보", interactive=False)

                        # 6-5-2. 스트리밍 텍스트 생성 (안내용)
                        with gr.TabItem("🚀 스트리밍 생성"):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    streaming_prompt = gr.Textbox(
                                        lines=4,
                                        label="프롬프트",
                                        placeholder="AI에게 질문하거나 요청하세요...\n예: 인공지능의 미래에 대해 설명해주세요."
                                    )
                                    streaming_model = gr.Dropdown(
                                        label="사용할 모델",
                                        choices=["기본 모델", "qwen2.5-7b", "llama3.1-8b", "gemma-3-4b"],
                                        value="기본 모델"
                                    )
                                    streaming_voice_lang = gr.Dropdown(
                                        choices=["ko", "en", "ja", "zh"],
                                        value="ko",
                                        label="음성 언어"
                                    )
                                    streaming_voice_slow = gr.Checkbox(label="느린 음성", value=False)
                                    streaming_read_partial = gr.Checkbox(label="부분 문장도 읽기", value=True)
                                    streaming_generate_button = gr.Button("🎯 전용 페이지로 이동", variant="primary")

                                with gr.Column(scale=2):
                                    streaming_output = gr.Markdown(label="안내 메시지")
                                    streaming_info = gr.Textbox(label="상태", interactive=False)

        # 7. Internal-DBMS RAG 탭 (NEW!)
        with gr.TabItem("🗄️ Internal-DBMS RAG (NEW!)"):
            gr.Markdown("### 🆕 내부 데이터베이스 기반 RAG 시스템")
            gr.Markdown("내부 DB 테이블을 벡터화하여 질의응답하는 시스템입니다.")
            
            with gr.Tabs():
                # 6-1. 테이블 관리
                with gr.TabItem("📋 테이블 관리"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            int_tables_button = gr.Button("테이블 목록 조회", variant="secondary")
                            int_status_button = gr.Button("FAISS 인덱스 상태", variant="secondary")

                            # 새로운 테이블 선택 및 보기 기능
                            gr.Markdown("---")
                            gr.Markdown("### 🔍 테이블 데이터 보기")
                            int_table_selector = gr.Dropdown(
                                choices=["테이블을 선택하세요"],
                                value="테이블을 선택하세요",
                                label="테이블 선택"
                            )
                            int_view_limit = gr.Slider(
                                minimum=10, maximum=500, value=50, step=10,
                                label="표시할 최대 행 수"
                            )
                            int_view_button = gr.Button("테이블 내용 보기", variant="primary")

                        with gr.Column(scale=2):
                            int_tables_output = gr.Markdown(label="테이블 목록")
                            int_status_output = gr.Markdown(label="상태 정보")
                            int_tables_status = gr.Textbox(label="조회 상태", interactive=False)

                            # 새로운 테이블 데이터 표시 영역
                            gr.Markdown("---")
                            int_table_data = gr.HTML(label="선택한 테이블 데이터", visible=True)
                            int_view_status = gr.Textbox(label="테이블 조회 상태", interactive=False)

                # 6-2. 테이블 인제스트
                with gr.TabItem("⚡ 테이블 인제스트"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            int_ingest_table = gr.Textbox(
                                label="테이블 이름", 
                                placeholder="knowledge",
                                value="knowledge"
                            )
                            int_ingest_save_name = gr.Textbox(
                                label="저장 이름", 
                                placeholder="knowledge (비워두면 테이블명 사용)"
                            )
                            int_ingest_simulate = gr.Checkbox(
                                label="시뮬레이션 모드 (SQLite 샘플 데이터 사용)", 
                                value=True
                            )
                            
                            with gr.Accordion("고급 설정 (선택사항)", open=False):
                                int_ingest_id_col = gr.Textbox(
                                    label="ID 컬럼명", 
                                    placeholder="자동 추론 (예: id)"
                                )
                                int_ingest_title_col = gr.Textbox(
                                    label="제목 컬럼명", 
                                    placeholder="자동 추론 (예: term, title)"
                                )
                                int_ingest_text_cols = gr.Textbox(
                                    label="텍스트 컬럼명 (쉼표 구분)", 
                                    placeholder="자동 추론 (예: description,role,details)"
                                )
                            
                            int_ingest_button = gr.Button("인제스트 실행", variant="primary")
                        
                        with gr.Column(scale=2):
                            int_ingest_output = gr.Markdown(label="인제스트 결과")
                            int_ingest_status = gr.Textbox(label="상태 정보", interactive=False)

                # 6-3. 질의응답
                with gr.TabItem("❓ 질의응답"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            int_query_save_name = gr.Textbox(
                                label="인덱스 이름", 
                                placeholder="knowledge",
                                value="knowledge"
                            )
                            int_query_question = gr.Textbox(
                                lines=3,
                                label="질문", 
                                placeholder="예: Self-Attention은 무엇인가? 역할과 함께 설명하라."
                            )
                            int_query_top_k = gr.Slider(
                                minimum=1, maximum=10, value=5, step=1,
                                label="검색할 문서 수 (top_k)"
                            )
                            int_query_margin = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.12, step=0.01,
                                label="마진 필터 (유사도 차이 허용 범위)"
                            )
                            int_query_button = gr.Button("질문하기", variant="primary")
                        
                        with gr.Column(scale=2):
                            int_query_answer = gr.Markdown(label="답변")
                            int_query_sources = gr.Markdown(label="참고 출처")
                            int_query_status = gr.Textbox(label="상태 정보", interactive=False)

    # --- 이벤트 핸들러 ---
    gen_button.click(fn=generate_text, inputs=[gen_prompt, gen_model_select, gen_streaming], outputs=[gen_output, gen_model_info])
    rag_button.click(fn=rag_query, inputs=[rag_question, rag_model_select], outputs=[rag_answer, rag_docs, rag_model_info_output])
    update_button.click(fn=update_rag_news, inputs=[news_query, news_max_results], outputs=update_status)
    
    # 뉴스 기능 이벤트 핸들러들
    latest_button.click(
        fn=get_latest_news, 
        inputs=[latest_categories, latest_max_results, latest_time_range], 
        outputs=[latest_output, latest_summary]
    )
    search_button.click(
        fn=search_news, 
        inputs=[search_query, search_category, search_max_results, search_time_range], 
        outputs=[search_output, search_summary]
    )
    summary_button.click(
        fn=summarize_news, 
        inputs=[summary_query, summary_type, summary_max_results, summary_model], 
        outputs=[summary_output, summary_articles, summary_info]
    )
    trend_button.click(
        fn=analyze_news_trends, 
        inputs=[trend_categories, trend_max_results, trend_time_range, trend_model], 
        outputs=[trend_output, trend_categories_output, trend_info]
    )
    
    # External-Web RAG 이벤트 핸들러들
    auto_button.click(
        fn=external_web_auto_rag,
        inputs=[auto_query, auto_max_results, auto_model],
        outputs=[auto_progress, auto_answer, auto_docs, auto_status]
    )
    ext_upload_button.click(
        fn=external_web_upload_topic,
        inputs=[ext_upload_topic, ext_upload_max_results],
        outputs=[ext_upload_output, ext_upload_status]
    )
    ext_query_button.click(
        fn=external_web_rag_query,
        inputs=[ext_query_prompt, ext_query_top_k, ext_query_model],
        outputs=[ext_query_answer, ext_query_docs, ext_query_status]
    )
    
    # Internal-DBMS RAG 이벤트 핸들러들
    def update_table_dropdown():
        """테이블 목록을 조회하고 드롭다운을 업데이트합니다"""
        formatted_tables, status, table_choices = internal_db_get_tables()
        return [
            formatted_tables,
            status,
            gr.Dropdown(choices=table_choices, value="테이블을 선택하세요", label="테이블 선택")
        ]

    int_tables_button.click(
        fn=update_table_dropdown,
        inputs=[],
        outputs=[int_tables_output, int_tables_status, int_table_selector]
    )
    int_status_button.click(
        fn=internal_db_get_status,
        inputs=[],
        outputs=[int_status_output, int_tables_status]
    )
    int_view_button.click(
        fn=internal_db_view_table,
        inputs=[int_table_selector, int_view_limit],
        outputs=[int_table_data, int_view_status]
    )
    int_ingest_button.click(
        fn=internal_db_ingest,
        inputs=[int_ingest_table, int_ingest_save_name, int_ingest_simulate, 
                int_ingest_id_col, int_ingest_title_col, int_ingest_text_cols],
        outputs=[int_ingest_output, int_ingest_status]
    )
    int_query_button.click(
        fn=internal_db_query,
        inputs=[int_query_save_name, int_query_question, int_query_top_k, int_query_margin],
        outputs=[int_query_answer, int_query_sources, int_query_status]
    )

    # 음성 기능 이벤트 핸들러들
    speech_status_button.click(
        fn=get_speech_service_status,
        inputs=[],
        outputs=[speech_status_output, speech_status_info]
    )
    tts_button.click(
        fn=text_to_speech_gradio,
        inputs=[tts_text, tts_language, tts_slow],
        outputs=[tts_audio_output, tts_status]
    )
    stt_button.click(
        fn=speech_to_text_gradio,
        inputs=[stt_audio_input, stt_prefer_whisper],
        outputs=[stt_text_output, stt_status]
    )
    voice_chat_button.click(
        fn=voice_chat_gradio,
        inputs=[voice_chat_text, voice_chat_model, voice_chat_language, voice_chat_slow],
        outputs=[voice_chat_audio, voice_chat_response, voice_chat_status]
    )

    # 스트리밍 음성 기능 이벤트 핸들러들
    streaming_status_button.click(
        fn=get_streaming_tts_status,
        inputs=[],
        outputs=[streaming_status_output, streaming_status_info]
    )
    streaming_generate_button.click(
        fn=streaming_generate_with_voice_gradio,
        inputs=[streaming_prompt, streaming_model, streaming_voice_lang, streaming_voice_slow, streaming_read_partial],
        outputs=[streaming_output, streaming_info]
    )

if __name__ == "__main__":
    gradio_ui.launch()
