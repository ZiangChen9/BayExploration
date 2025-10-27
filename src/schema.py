"""定义优化结果的输出"""

from dataclasses import field
from datetime import datetime
from typing import Annotated, List, Optional, Dict, Any

import torch
from pydantic import Field, BaseModel, model_validator


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

    def __str__(self) -> str:
        """返回人类可读的字符串表示"""

        # 简化 Tensor 显示，只显示形状和前几个值
        def format_tensor(tensor: torch.Tensor, max_elements: int = 3) -> str:
            if tensor.numel() <= max_elements:
                values = tensor.tolist()
                return f"Tensor{list(tensor.shape)}[{', '.join(f'{v:.4f}' for v in values)}]"
            else:
                preview = tensor.flatten()[:max_elements].tolist()
                return f"Tensor{list(tensor.shape)}[{', '.join(f'{v:.4f}' for v in preview)}, ...]"

        return (
            f"OptimizationRecord(\n"
            f"  迭代: {self.iteration_id}, 重复: {self.replication_id}\n"
            f"  当前值: {self.current_value:.6f}\n"
            f"  当前位置: {format_tensor(self.current_position)}\n"
            f"  最佳值: {self.best_value:.6f}\n"
            f"  最佳位置: {format_tensor(self.best_position)}\n"
            f"  时间: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f")"
        )

    def __repr__(self) -> str:
        """返回简洁的字符串表示"""
        return (
            f"OptimizationRecord("
            f"iter={self.iteration_id}, "
            f"rep={self.replication_id}, "
            f"current={self.current_value:.4f}, "
            f"best={self.best_value:.4f}"
            f")"
        )


class OptimizationHistory(BaseModel):
    """某个优化实验的所有历史记录"""

    records: List[OptimizationRecord] = (
        Field(default_factory=list, description="优化记录列表"),
    )
    bo_alg_name: str = Field(description="Bayesian Optimization Algorithm Name")
    target_func_name: str = Field(description="优化问题名称")
    start_time: datetime = Field(
        default_factory=datetime.now, description="实验开始时间"
    )
    end_time: Optional[datetime] = Field(None, description="实验结束时间")
    total_iterations: int = Field(0, description="单轮迭代次数")
    total_replications: int = Field(0, description="总重复轮数")
    global_best_value: Optional[float] = Field(None, description="全局最优值")
    global_best_position: Optional[torch.Tensor] = Field(None, description="全局最优点")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="算法参数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="附加元数据")

    def add_record(self, record: OptimizationRecord):
        """添加单条记录"""
        self.records.append(record)
        self.total_iterations = max(self.total_iterations, record.iteration_id + 1)
        self.total_replications = max(
            self.total_replications, record.replication_id + 1
        )

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

    def get_records_by_replication(self, replication_id: int) -> List[OptimizationRecord]:
        """按重复实验ID获取记录"""
        return [r for r in self.records if r.replication_id == replication_id]

    def get_convergence_history(self) -> List[float]:
        """获取收敛历史（最优值变化）"""
        return [
            r.best_value_by_now
            for r in sorted(self.records, key=lambda x: x.iteration_id)
        ]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（排除Tensor）"""
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

   

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> OptimizationRecord:
        return self.records[index]
