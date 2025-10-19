from typing import List, Optional, Dict
from datetime import datetime
import sys
import os

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.finance import FinanceItem

class FinanceService:
    """재정 관리 서비스"""

    def __init__(self):
        # 임시 데이터베이스 (실제로는 DB 사용)
        self.finance_items_db: List[Dict] = []
        self.next_id: int = 1

    async def list_items(self, category: Optional[str] = None) -> List[Dict]:
        """재정 항목 목록 조회"""
        if category:
            return [item for item in self.finance_items_db if item["category"] == category]
        return self.finance_items_db

    async def create_item(self, item: FinanceItem) -> Dict:
        """재정 항목 추가"""
        new_item = item.dict()
        new_item["id"] = self.next_id
        new_item["createdAt"] = datetime.now().isoformat()
        new_item["updatedAt"] = datetime.now().isoformat()

        self.finance_items_db.append(new_item)
        self.next_id += 1

        print(f"자산 추가: {new_item['category']} - {new_item['name']} - {new_item['amount']}원")

        return new_item

    async def get_item(self, item_id: int) -> Optional[Dict]:
        """특정 재정 항목 조회"""
        for item in self.finance_items_db:
            if item["id"] == item_id:
                return item
        return None

    async def update_item(self, item_id: int, item: FinanceItem) -> Optional[Dict]:
        """재정 항목 수정"""
        for i, existing_item in enumerate(self.finance_items_db):
            if existing_item["id"] == item_id:
                updated_item = item.dict()
                updated_item["id"] = item_id
                updated_item["createdAt"] = existing_item["createdAt"]
                updated_item["updatedAt"] = datetime.now().isoformat()

                self.finance_items_db[i] = updated_item
                return updated_item

        return None

    async def delete_item(self, item_id: int) -> bool:
        """재정 항목 삭제"""
        for i, item in enumerate(self.finance_items_db):
            if item["id"] == item_id:
                deleted_item = self.finance_items_db.pop(i)
                print(f"자산 삭제: {deleted_item['category']} - {deleted_item['name']}")
                return True

        return False
