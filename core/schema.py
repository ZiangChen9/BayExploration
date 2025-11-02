"""定义优化结果的输出"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel, Field, model_validator


class OptimizationRecord(BaseModel):
    """单次贝叶斯优化结果记录"""

    iteration_id: int = Field(ge=0, description="迭代次数编号")
    replication_id: int = Field(ge=0, description="重复次数编号")
    current_value: float = Field(description="当前迭代的函数值")
    current_position: torch.Tensor = Field(description="当前迭代的采样位置")
    best_value: float = Field(description="迄今为止的最优函数值")
    best_position: torch.Tensor = Field(description="迄今为止的最优位置")
    created_at: datetime = Field(
        default_factory=datetime.now, description="记录创建时间戳"
    )
    end_at: Optional[datetime] = Field(description="记录结束时间戳")
    duration: float = Field(description="优化过程持续时间")

    @model_validator(mode="after")
    def calculate_duration(self) -> "OptimizationRecord":
        """Auto count the duration of the optimization process"""
        if self.created_at and self.end_at:
            self.duration = (self.end_at - self.created_at).total_seconds()
        return self

    def to_dict(self, tensor_to_list: bool = True) -> dict:
        # Turn Pydantic model into a dictionary
        data = self.model_dump()
        if tensor_to_list:
            # Convert torch.Tensor to list
            if isinstance(data.get("current_position"), torch.Tensor):
                data["current_position"] = data["current_position"].tolist()
            if isinstance(data.get("best_position"), torch.Tensor):
                data["best_position"] = data["best_position"].tolist()
        # Convert datetime to string
        if isinstance(data.get("created_at"), datetime):
            data["created_at"] = data["created_at"].isoformat()
        if isinstance(data.get("end_at"), datetime):
            data["end_at"] = data["end_at"].isoformat()
        return data


class OptimizationHistory(BaseModel):
    """某个优化实验的所有历史记录"""

    bo_alg_name: str = Field(description="Bayesian Optimization Algorithm Name")
    target_func_name: str = Field(description="优化问题名称")
    records: List[OptimizationRecord] = (
        Field(default_factory=list, description="优化记录列表"),
    )
    start_time: datetime = Field(
        default_factory=datetime.now, description="实验开始时间"
    )
    end_time: datetime = Field(default_factory=datetime.now, description="实验结束时间")
    total_iterations: int = Field(default_factory=int, description="单轮迭代次数")
    total_replications: int = Field(default_factory=int, description="总重复轮数")
    global_best_value: float = Field(default_factory=float, description="全局最优值")
    global_best_position: torch.Tensor = Field(
        default_factory=torch.Tensor, description="全局最优点"
    )
    parameters: Dict[str, Any] = Field(default_factory=dict, description="算法参数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="附加元数据")

    def add_record(self, record: OptimizationRecord):
        """添加单条记录"""
        self.records.append(record)
        # 更新全局最优
        if (
            self.global_best_value is None
            or record.best_value_by_now < self.global_best_value
        ):
            self.global_best_value = record.best_value_by_now
            self.global_best_position = record.best_position_by_now.clone()

    def get_best_record(self) -> Optional[OptimizationRecord]:
        """获取最优记录"""
        if not self.records:
            return None
        return min(self.records, key=lambda x: x.best_value_by_now)

    def get_records_by_replication(
        self, replication_id: int
    ) -> List[OptimizationRecord]:
        """按重复实验ID获取记录"""
        return self.records[replication_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_name": self.algorithm_name,
            "problem_name": self.problem_name,
            "total_iterations": self.total_iterations,
            "total_replications": self.total_replications,
            "global_best_value": self.global_best_value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "record_count": len(self.records),
            "parameters": self.parameters,
        }
