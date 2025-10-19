"""
LLM이 사용할 수 있는 Tool 정의
"""

TOOL_DEFINITIONS = [
    {
        "name": "get_news",
        "description": "최신 뉴스 기사를 조회합니다. 키워드로 필터링할 수 있습니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "검색할 뉴스 키워드 (예: '의료', '재테크', 'AI')"
                },
                "limit": {
                    "type": "integer",
                    "description": "조회할 뉴스 개수 (기본값: 5)",
                    "default": 5
                }
            }
        }
    },
    {
        "name": "get_weather",
        "description": "현재 날씨 정보를 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "위치 (예: '서울', '부산')",
                    "default": "서울"
                }
            }
        }
    },
    {
        "name": "get_health_status",
        "description": "사용자의 전반적인 건강 상태를 조회합니다 (기저질환, 복용약, 건강 점수 등).",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_diseases",
        "description": "사용자의 기저질환 목록을 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_medications",
        "description": "사용자의 복용약 목록을 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "add_disease",
        "description": "새로운 기저질환을 추가합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "질병명 (예: '고혈압', '당뇨')"
                },
                "diagnosed_date": {
                    "type": "string",
                    "description": "진단일 (YYYY-MM-DD 형식, 선택사항)"
                },
                "status": {
                    "type": "string",
                    "description": "상태 (active/controlled/cured, 기본값: active)",
                    "default": "active"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "add_medication",
        "description": "새로운 복용약을 추가합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "약 이름"
                },
                "dosage": {
                    "type": "string",
                    "description": "복용량 (예: '100mg', '1정')"
                },
                "intake_time": {
                    "type": "string",
                    "description": "복용 시간 (HH:MM 형식, 예: '08:00')"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "get_finance_updates",
        "description": "금융 및 주식 정보를 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "카테고리 (stocks/crypto/forex, 선택사항)"
                }
            }
        }
    },
    {
        "name": "get_calendar_events",
        "description": "사용자의 일정을 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "날짜 (YYYY-MM-DD 형식, 생략 시 오늘 일정)"
                },
                "limit": {
                    "type": "integer",
                    "description": "조회할 일정 개수",
                    "default": 10
                }
            }
        }
    },
    {
        "name": "add_calendar_event",
        "description": "새로운 일정을 추가합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "일정 제목 (예: '병원 예약', '회의', '저녁 약속')"
                },
                "event_date": {
                    "type": "string",
                    "description": "일정 날짜 (YYYY-MM-DD 형식)"
                },
                "event_time": {
                    "type": "string",
                    "description": "일정 시간 (HH:MM 형식, 예: '14:30')"
                },
                "event_type": {
                    "type": "string",
                    "description": "일정 유형 (병원/회의/약속/기타 중 선택, 기본값: 약속)",
                    "default": "약속"
                },
                "location": {
                    "type": "string",
                    "description": "장소 (선택사항)"
                },
                "description": {
                    "type": "string",
                    "description": "일정 설명 (선택사항)"
                }
            },
            "required": ["title", "event_date", "event_time"]
        }
    },
    {
        "name": "delete_calendar_event",
        "description": "기존 일정을 삭제합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "integer",
                    "description": "삭제할 일정의 ID"
                }
            },
            "required": ["event_id"]
        }
    },
    {
        "name": "get_diet_plan",
        "description": "사용자의 건강 상태(기저질환, 복용약)를 확인하여 개인화된 식단 및 음식을 추천합니다. 건강에 좋은 음식, 피해야 할 음식 등을 알려줄 수 있습니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "날짜 (YYYY-MM-DD 형식, 생략 시 오늘 식단)"
                }
            }
        }
    },
    {
        "name": "get_notifications",
        "description": "미읽은 알림을 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "조회할 알림 개수",
                    "default": 5
                }
            }
        }
    },
    {
        "name": "get_grocery_prices",
        "description": "식료품의 최저가 구매처를 조회합니다. 여러 마트의 가격을 비교하여 알려줍니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "product": {
                    "type": "string",
                    "description": "조회할 식료품 이름 (예: '바나나', '사과', '우유', '계란')"
                }
            },
            "required": ["product"]
        }
    }
]


def get_tool_by_name(tool_name: str):
    """도구 이름으로 정의 찾기"""
    for tool in TOOL_DEFINITIONS:
        if tool["name"] == tool_name:
            return tool
    return None


def get_all_tool_names():
    """모든 도구 이름 반환"""
    return [tool["name"] for tool in TOOL_DEFINITIONS]
