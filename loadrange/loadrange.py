from dataclasses import dataclass
from typing import List, Dict, Optional
from queue import PriorityQueue
import time
import threading
from loadpredict import PrioritizedRequest

@dataclass
class QueueInfo:
    """队列信息"""
    queue: PriorityQueue
    current_priority: int  # 当前优先级
    promotion_count: int   # 优先级提升次数
    total_tokens: int      # 队列中请求的总token数

class LoadRange:
    def __init__(self, max_batch_size: int = 1024*1024, max_batch_time: float = 1.0):
        """
        初始化LoadRange

        Args:
            max_batch_size: 最大批处理权值B
            max_batch_time: 最大批处理等待时间（秒）
        """
        self.max_batch_size = max_batch_size
        self.max_batch_time = max_batch_time

        # 工作区
        self.working_area: List[PrioritizedRequest] = []

        # 等待区：9个优先级队列
        self.waiting_area: Dict[str, QueueInfo] = {
            category: QueueInfo(
                queue=PriorityQueue(),
                current_priority=i+1,
                promotion_count=0,
                total_tokens=0
            )
            for i, category in enumerate(['SS', 'MS', 'LS', 'SM', 'MM', 'LM', 'SL', 'ML', 'LL'])
        }

        # 启动定时检查线程
        self.last_check_time = time.time()
        self.check_thread = threading.Thread(target=self._periodic_check, daemon=True)
        self.check_thread.start()

        # 线程同步锁
        self.lock = threading.Lock()

    def add_request(self, request: PrioritizedRequest) -> bool:
        """
        添加请求到等待区对应的优先级队列

        Args:
            request: 优先级请求对象
        Returns:
            bool: 是否成功添加
        """
        with self.lock:
            queue_info = self.waiting_area[request.category]
            queue_info.queue.put(request)
            # 更新队列的总token数（假设request对象有input_tokens属性）
            queue_info.total_tokens += request.request.input_tokens
            return True

    def _get_highest_priority_full_queue(self) -> Optional[str]:
        """
        获取达到最大批处理权值且优先级最高的队列

        Returns:
            str: 队列类别名称，如果没有满足条件的队列则返回None
        """
        highest_priority = float('inf')
        selected_category = None
        min_promotion_count = float('inf')

        for category, info in self.waiting_area.items():
            if info.total_tokens >= self.max_batch_size:
                if info.current_priority < highest_priority:
                    highest_priority = info.current_priority
                    selected_category = category
                    min_promotion_count = info.promotion_count
                elif info.current_priority == highest_priority and info.promotion_count > min_promotion_count:
                    selected_category = category
                    min_promotion_count = info.promotion_count

        return selected_category

    def _get_highest_priority_queue(self) -> Optional[str]:
        """
        获取优先级最高且提升次数最少的非空队列

        Returns:
            str: 队列类别名称，如果没有满足条件的队列则返回None
        """
        highest_priority = float('inf')
        selected_category = None
        min_promotion_count = float('inf')

        for category, info in self.waiting_area.items():
            if not info.queue.empty():
                if info.current_priority < highest_priority:
                    highest_priority = info.current_priority
                    selected_category = category
                    min_promotion_count = info.promotion_count
                elif info.current_priority == highest_priority and info.promotion_count < min_promotion_count:
                    selected_category = category
                    min_promotion_count = info.promotion_count

        return selected_category

    def _promote_queues(self, selected_category: Optional[str]):
        """
        提升未被选中的满队列的优先级

        Args:
            selected_category: 被选中的队列类别
        """
        for category, info in self.waiting_area.items():
            if (category != selected_category and
                info.total_tokens >= self.max_batch_size and
                info.current_priority > 1):
                info.current_priority = max(1, info.current_priority - 1)
                info.promotion_count += 1

    def _load_to_working_area(self, category: str):
        """
        将指定队列中的请求加载到工作区

        Args:
            category: 队列类别
        """
        queue_info = self.waiting_area[category]
        while not queue_info.queue.empty():
            request = queue_info.queue.get()
            self.working_area.append(request)
        queue_info.total_tokens = 0

    def _periodic_check(self):
        """定期检查批处理工作区"""
        while True:
            time.sleep(0.1)  # 避免过于频繁的检查
            current_time = time.time()

            with self.lock:
                if (len(self.working_area) == 0 and
                    current_time - self.last_check_time >= self.max_batch_time):
                    # 检查是否有满足条件的队列
                    selected_category = self._get_highest_priority_full_queue()

                    # 如果没有满队列，但已经超时，选择优先级最高的非空队列
                    if selected_category is None:
                        selected_category = self._get_highest_priority_queue()

                    if selected_category:
                        self._load_to_working_area(selected_category)
                        self._promote_queues(selected_category)
                        self.last_check_time = current_time

    def get_working_requests(self) -> List[PrioritizedRequest]:
        """
        获取工作区中的请求

        Returns:
            List[PrioritizedRequest]: 工作区中的请求列表
        """
        with self.lock:
            requests = self.working_area.copy()
            self.working_area.clear()
            return requests

    def is_working_area_empty(self) -> bool:
        """
        检查工作区是否为空

        Returns:
            bool: 工作区是否为空
        """
        with self.lock:
            return len(self.working_area) == 0
