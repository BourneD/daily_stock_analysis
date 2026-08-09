import json
from pathlib import Path
from typing import Optional
from .realtime_types import ChipDistribution

def load_fallback_data(stock_code: str) -> Optional[ChipDistribution]:
    file_path = Path("/root/daily_stock_analysis/data_fallback/chips.json")

    # 判断文件是否存在
    if not file_path.exists():
        return None

    # 读取并解析json
    with open(file_path, "r", encoding="utf-8") as f:
        data_list: list[ChipDistribution] = json.load(f)

    # 以code为key构建map
    chip_map: dict[str, ChipDistribution] = {item["code"]: ChipDistribution(
          code=item["code"],
          date=item["date"],
          profit_ratio=item['profit_ratio'],
          avg_cost=item['avg_cost'],
          cost_90_low=item['cost_90_low'],
          cost_90_high=item['cost_90_high'],
          concentration_90=item['concentration_90'],
          cost_70_low=item['cost_70_low'],
          cost_70_high=item['cost_70_high'],
          concentration_70=item['concentration_70']
        ) for item in data_list}

    # 根据stock_code取值
    return chip_map.get(stock_code)
