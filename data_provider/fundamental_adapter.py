# -*- coding: utf-8 -*-
"""
AkShare fundamental adapter (fail-open).

This adapter intentionally uses capability probing against multiple AkShare
endpoint candidates. It should never raise to caller; partial data is allowed.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_DIVIDEND_KEYWORD_MAP: Dict[str, List[str]] = {
    "per_share": [
        "每股派息",
        "每股现金红利",
        "每股分红",
        "每股派现",
        "派现(元/股)",
        "派息(元/股)",
        "税前派息(元/股)",
        "现金分红(税前)",
    ],
    "plan_text": [
        "分配方案",
        "分红方案",
        "实施方案",
        "派息方案",
        "方案",
        "预案",
        "方案说明",
    ],
    "ex_dividend_date": ["除权除息日", "除息日", "除权日", "除权除息", "除息日期"],
    "record_date": ["股权登记日", "登记日"],
    "announce_date": ["公告日期", "公告日", "实施公告日", "预案公告日"],
    "report_date": ["报告期", "报告日期", "截止日期", "统计截止日期"],
}


def _safe_float(value: Any) -> Optional[float]:
    """Best-effort float conversion."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    s = str(value).strip().replace(",", "").replace("%", "")
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    try:
        return parsed.to_pydatetime()
    except Exception:
        return None


def _normalize_code(raw: Any) -> str:
    s = _safe_str(raw).upper()
    if "." in s:
        s = s.split(".", 1)[0]
    s = re.sub(r"^(SH|SZ|BJ)", "", s)
    return s


def _pick_by_keywords(row: pd.Series, keywords: List[str]) -> Optional[Any]:
    """
    Return first non-empty row value whose column name contains any keyword.
    """
    for col in row.index:
        col_s = str(col)
        if any(k in col_s for k in keywords):
            val = row.get(col)
            if val is not None and str(val).strip() not in ("", "-", "nan", "None"):
                return val
    return None


def _parse_dividend_plan_to_per_share(plan_text: str) -> Optional[float]:
    """Parse per-share cash dividend from Chinese plan text."""
    text = _safe_str(plan_text)
    if not text:
        return None

    for pattern in (
        r"(?:每)?\s*10\s*股?\s*派(?:发)?\s*([0-9]+(?:\.[0-9]+)?)\s*元",
        r"10\s*派\s*([0-9]+(?:\.[0-9]+)?)\s*元",
    ):
        match = re.search(pattern, text)
        if match:
            parsed = _safe_float(match.group(1))
            if parsed is not None and parsed > 0:
                return parsed / 10.0

    match_per_share = re.search(r"每\s*股\s*派(?:发)?\s*([0-9]+(?:\.[0-9]+)?)\s*元", text)
    if match_per_share:
        parsed = _safe_float(match_per_share.group(1))
        if parsed is not None and parsed > 0:
            return parsed
    return None


def _extract_cash_dividend_per_share(row: pd.Series) -> Optional[float]:
    """Extract pre-tax cash dividend per share from a row."""
    plan_text = _safe_str(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["plan_text"]))
    # Keep pre-tax semantics; skip explicit after-tax plans unless pre-tax marker exists.
    if "税后" in plan_text and "税前" not in plan_text and "含税" not in plan_text:
        return None

    direct = _safe_float(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["per_share"]))
    if direct is not None and direct > 0:
        return direct
    return _parse_dividend_plan_to_per_share(plan_text)


def _filter_rows_by_code(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码", "symbol", "ts_code"))]
    if not code_cols:
        return df

    target = _normalize_code(stock_code)
    for col in code_cols:
        try:
            series = df[col].astype(str).map(_normalize_code)
            filtered = df[series == target]
            if not filtered.empty:
                return filtered
        except Exception:
            continue
    return pd.DataFrame()


def _normalize_report_date(value: Any) -> Optional[str]:
    parsed = _safe_datetime(value)
    return parsed.date().isoformat() if parsed else None


def _build_dividend_payload(
    dividend_df: pd.DataFrame,
    stock_code: str,
    max_events: int = 5,
) -> Dict[str, Any]:
    work_df = _filter_rows_by_code(dividend_df, stock_code)
    if work_df.empty:
        return {}

    now_date = datetime.now().date()
    ttm_start_date = now_date - timedelta(days=365)
    dedupe_keys = set()
    events: List[Dict[str, Any]] = []

    for _, row in work_df.iterrows():
        if not isinstance(row, pd.Series):
            continue
        ex_dt = _safe_datetime(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["ex_dividend_date"]))
        record_dt = _safe_datetime(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["record_date"]))
        announce_dt = _safe_datetime(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["announce_date"]))
        event_dt = ex_dt or record_dt or announce_dt
        if event_dt is None:
            continue
        event_date = event_dt.date()
        if event_date > now_date:
            continue

        per_share = _extract_cash_dividend_per_share(row)
        if per_share is None or per_share <= 0:
            continue

        dedupe_key = (event_date.isoformat(), round(per_share, 6))
        if dedupe_key in dedupe_keys:
            continue
        dedupe_keys.add(dedupe_key)

        events.append(
            {
                "event_date": event_date.isoformat(),
                "ex_dividend_date": ex_dt.date().isoformat() if ex_dt else None,
                "record_date": record_dt.date().isoformat() if record_dt else None,
                "announcement_date": announce_dt.date().isoformat() if announce_dt else None,
                "cash_dividend_per_share": round(per_share, 6),
                "is_pre_tax": True,
            }
        )

    if not events:
        return {}

    events.sort(key=lambda item: item.get("event_date") or "", reverse=True)
    ttm_events: List[Dict[str, Any]] = []
    for item in events:
        event_dt = _safe_datetime(item.get("event_date"))
        if event_dt is None:
            continue
        event_date = event_dt.date()
        if ttm_start_date <= event_date <= now_date:
            ttm_events.append(item)

    return {
        "events": events[:max(1, max_events)],
        "ttm_event_count": len(ttm_events),
        "ttm_cash_dividend_per_share": (
            round(sum(float(item.get("cash_dividend_per_share") or 0.0) for item in ttm_events), 6)
            if ttm_events else None
        ),
        "coverage": "cash_dividend_pre_tax",
        "as_of": now_date.isoformat(),
    }


def _latest_row_by_date(df: Optional[pd.DataFrame], date_col: str = "end_date") -> Optional[pd.Series]:
    """返回日期列最新的行（YYYYMMDD 字符串按字典序 == 时间序）。"""
    if df is None or df.empty or date_col not in df.columns:
        return None
    work = df.copy()
    work[date_col] = work[date_col].astype(str)
    work = work[work[date_col].str.fullmatch(r"\d{8}")].reset_index(drop=True)
    if work.empty:
        return None
    return work.iloc[work[date_col].idxmax()]


def _tushare_float(row: Optional[pd.Series], column: str) -> Optional[float]:
    """从 tushare 行取数值，NaN / 空串按 None 处理，避免 float(nan) 污染下游。"""
    if row is None or column not in row.index:
        return None
    value = row.get(column)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return _safe_float(value)


def _filter_consolidated(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """优先取合并报表口径（comp_type==1），无该列或不存在合并行时按原样返回。"""
    if df is None or df.empty or "comp_type" not in df.columns:
        return df
    work = df[df["comp_type"].astype(str).str.strip() == "1"]
    return work if not work.empty else df


def build_financial_bundle_from_tushare(
    frames: Dict[str, pd.DataFrame],
    stock_code: str,
) -> Dict[str, Any]:
    """
    将 Tushare 财报接口原始数据映射为与 akshare 口径一致的基本面 payload。

    frames 键为 tushare 接口名（fina_indicator / income / cashflow / forecast /
    express / dividend）。金额单位与 akshare 契约对齐（元），同比 / ROE / 毛利率
    为百分比数值。growth / financial_report 均无有效值时返回空 payload，由调用
    方回退 akshare 候选链。
    """
    result: Dict[str, Any] = {
        "status": "not_supported",
        "growth": {},
        "earnings": {},
        "source_chain": [],
        "errors": [],
    }

    fina_row = _latest_row_by_date(frames.get("fina_indicator"))
    inc_row = _latest_row_by_date(_filter_consolidated(frames.get("income")))
    cash_row = _latest_row_by_date(_filter_consolidated(frames.get("cashflow")))

    # Growth（财务指标）
    growth_payload: Dict[str, Any] = {}
    if fina_row is not None:
        revenue_yoy = _tushare_float(fina_row, "or_yoy")  # 营业收入同比（%）
        if revenue_yoy is None:
            revenue_yoy = _tushare_float(fina_row, "tr_yoy")  # 营业总收入同比（%）
        growth_payload = {
            "revenue_yoy": revenue_yoy,
            "net_profit_yoy": _tushare_float(fina_row, "netprofit_yoy"),
            "roe": _tushare_float(fina_row, "roe"),
            "gross_margin": _tushare_float(fina_row, "grossprofit_margin"),
        }
    if any(v is not None for v in growth_payload.values()):
        result["growth"] = growth_payload
        result["source_chain"].append("growth:tushare_fina_indicator")

    # Financial report（利润表 + 现金流量表 + 财务指标）
    financial_report_payload: Dict[str, Any] = {}
    report_date = None
    if fina_row is not None:
        report_date = _normalize_report_date(fina_row.get("end_date"))
    if inc_row is not None:
        if report_date is None:
            report_date = _normalize_report_date(inc_row.get("end_date"))
        financial_report_payload["revenue"] = _tushare_float(inc_row, "total_revenue")
        financial_report_payload["net_profit_parent"] = _tushare_float(inc_row, "n_income_attr_p")
    if cash_row is not None:
        financial_report_payload["operating_cash_flow"] = _tushare_float(cash_row, "n_cashflow_act")
    if fina_row is not None:
        financial_report_payload["roe"] = _tushare_float(fina_row, "roe")
    financial_report_payload["report_date"] = report_date
    if any(v is not None for v in financial_report_payload.values()):
        result["earnings"]["financial_report"] = financial_report_payload
        result["source_chain"].append("earnings_financial:tushare_financials")

    # Earnings forecast（业绩预告）
    forecast_row = _latest_row_by_date(frames.get("forecast"))
    if forecast_row is not None:
        forecast_type = _safe_str(forecast_row.get("type"))
        forecast_summary = _safe_str(forecast_row.get("summary")) or _safe_str(forecast_row.get("change_reason"))
        text = f"{forecast_type}：{forecast_summary}" if forecast_type and forecast_summary else (forecast_type or forecast_summary)
        if text:
            result["earnings"]["forecast_summary"] = text[:200]
            result["source_chain"].append("earnings_forecast:tushare_forecast")

    # Earnings quick report（业绩快报）
    express_row = _latest_row_by_date(frames.get("express"))
    if express_row is not None:
        express_revenue = _tushare_float(express_row, "revenue")
        if express_revenue is not None:
            parts = [f"营业收入{express_revenue / 1e8:.2f}亿"]
            net_income = _tushare_float(express_row, "n_income")
            if net_income is not None:
                parts.append(f"净利润{net_income / 1e8:.2f}亿")
            yoy_net_profit = _tushare_float(express_row, "yoy_net_profit")
            if yoy_net_profit is not None:
                parts.append(f"净利润同比{yoy_net_profit:.2f}%")
            perf_summary = _safe_str(express_row.get("perf_summary"))
            text = f"业绩快报：{'，'.join(parts)}"
            if perf_summary:
                text += f"（{perf_summary}）"
            result["earnings"]["quick_report_summary"] = text[:200]
            result["source_chain"].append("earnings_quick:tushare_express")

    # Dividend（分红送配，复用 akshare 分红 payload 构建逻辑）
    dividend_df = frames.get("dividend")
    if dividend_df is not None and not dividend_df.empty:
        rows: List[Dict[str, Any]] = []
        for _, row in dividend_df.iterrows():
            ex_date = _safe_str(row.get("ex_date"))
            cash_div_tax = _tushare_float(row, "cash_div_tax")  # 每股派息税前（元/股）
            if not ex_date or cash_div_tax is None or cash_div_tax <= 0:
                continue
            rows.append({
                "股票代码": stock_code,
                "除息日": ex_date,
                "分配方案": f"每股派{cash_div_tax}元(含税)",
            })
        if rows:
            dividend_payload = _build_dividend_payload(pd.DataFrame(rows), stock_code, max_events=5)
            if dividend_payload:
                result["earnings"]["dividend"] = dividend_payload
                result["source_chain"].append("dividend:tushare_dividend")

    has_content = bool(result["growth"] or result["earnings"])
    result["status"] = "partial" if has_content else "not_supported"
    return result


def _extract_latest_row(df: pd.DataFrame, stock_code: str) -> Optional[pd.Series]:
    """
    Select the most relevant row for the given stock.
    """
    if df is None or df.empty:
        return None

    code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码", "ts_code", "symbol"))]
    target = _normalize_code(stock_code)
    if code_cols:
        for col in code_cols:
            try:
                series = df[col].astype(str).map(_normalize_code)
                matched = df[series == target]
                if not matched.empty:
                    return matched.iloc[0]
            except Exception:
                continue
        return None

    # Fallback: use latest row
    return df.iloc[0]


class AkshareFundamentalAdapter:
    """AkShare adapter for fundamentals, capital flow and dragon-tiger signals."""

    def _call_df_candidates(
        self,
        candidates: List[Tuple[str, Dict[str, Any]]],
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], List[str]]:
        errors: List[str] = []
        try:
            import akshare as ak
        except Exception as exc:
            return None, None, [f"import_akshare:{type(exc).__name__}"]

        for func_name, kwargs in candidates:
            fn = getattr(ak, func_name, None)
            if fn is None:
                continue
            try:
                df = fn(**kwargs)
                if isinstance(df, pd.Series):
                    df = df.to_frame().T
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df, func_name, errors
            except Exception as exc:
                errors.append(f"{func_name}:{type(exc).__name__}")
                continue
        return None, None, errors

    def get_fundamental_bundle(
        self,
        stock_code: str,
        financial_bundle: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Return normalized fundamental blocks with partial tolerance.

        financial_bundle: 预取的财报 payload（当前来自 Tushare fina_indicator/income/
        cashflow/forecast/express/dividend），字段与 akshare 口径对齐。growth 或
        financial_report 非空时跳过 akshare 财报候选链；机构/十大股东仍走 akshare
        （fail-open）。为 None 或核心财报为空时按原有 akshare 候选链获取。
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "growth": {},
            "earnings": {},
            "institution": {},
            "source_chain": [],
            "errors": [],
        }

        use_tushare = bool(
            isinstance(financial_bundle, dict)
            and (
                financial_bundle.get("growth")
                or (financial_bundle.get("earnings") or {}).get("financial_report")
            )
        )
        if use_tushare:
            tushare_earnings = financial_bundle.get("earnings")
            result["growth"] = dict(financial_bundle.get("growth") or {})
            result["earnings"] = dict(tushare_earnings) if isinstance(tushare_earnings, dict) else {}
            result["source_chain"].extend(list(financial_bundle.get("source_chain") or []))
            result["errors"].extend(list(financial_bundle.get("errors") or []))
        else:
            # Financial indicators
            fin_df, fin_source, fin_errors = self._call_df_candidates([
                ("stock_financial_abstract", {"symbol": stock_code}),
                ("stock_financial_analysis_indicator", {"symbol": stock_code}),
                ("stock_financial_analysis_indicator", {}),
            ])
            result["errors"].extend(fin_errors)
            if fin_df is not None:
                row = _extract_latest_row(fin_df, stock_code)
                if row is not None:
                    revenue_yoy = _safe_float(_pick_by_keywords(row, ["营业收入同比", "营收同比", "收入同比", "同比增长"]))
                    profit_yoy = _safe_float(_pick_by_keywords(row, ["净利润同比", "净利同比", "归母净利润同比"]))
                    roe = _safe_float(_pick_by_keywords(row, ["净资产收益率", "ROE", "净资产收益"]))
                    gross_margin = _safe_float(_pick_by_keywords(row, ["毛利率"]))
                    report_date = _normalize_report_date(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["report_date"]))
                    revenue = _safe_float(_pick_by_keywords(row, ["营业总收入", "营业收入", "营收"]))
                    net_profit_parent = _safe_float(_pick_by_keywords(row, ["归母净利润", "母公司股东净利润", "净利润"]))
                    operating_cash_flow = _safe_float(
                        _pick_by_keywords(row, ["经营活动产生的现金流量净额", "经营现金流", "经营活动现金流"])
                    )
                    result["growth"] = {
                        "revenue_yoy": revenue_yoy,
                        "net_profit_yoy": profit_yoy,
                        "roe": roe,
                        "gross_margin": gross_margin,
                    }
                    financial_report_payload = {
                        "report_date": report_date,
                        "revenue": revenue,
                        "net_profit_parent": net_profit_parent,
                        "operating_cash_flow": operating_cash_flow,
                        "roe": roe,
                    }
                    if any(v is not None for v in financial_report_payload.values()):
                        result["earnings"]["financial_report"] = financial_report_payload
                    result["source_chain"].append(f"growth:{fin_source}")

            # Earnings forecast
            forecast_df, forecast_source, forecast_errors = self._call_df_candidates([
                ("stock_yjyg_em", {"symbol": stock_code}),
                ("stock_yjyg_em", {}),
                ("stock_yjbb_em", {"symbol": stock_code}),
                ("stock_yjbb_em", {}),
            ])
            result["errors"].extend(forecast_errors)
            if forecast_df is not None:
                row = _extract_latest_row(forecast_df, stock_code)
                if row is not None:
                    result["earnings"]["forecast_summary"] = _safe_str(
                        _pick_by_keywords(row, ["预告", "业绩变动", "内容", "摘要", "公告"])
                    )[:200]
                    result["source_chain"].append(f"earnings_forecast:{forecast_source}")

            # Earnings quick report
            quick_df, quick_source, quick_errors = self._call_df_candidates([
                ("stock_yjkb_em", {"symbol": stock_code}),
                ("stock_yjkb_em", {}),
            ])
            result["errors"].extend(quick_errors)
            if quick_df is not None:
                row = _extract_latest_row(quick_df, stock_code)
                if row is not None:
                    result["earnings"]["quick_report_summary"] = _safe_str(
                        _pick_by_keywords(row, ["快报", "摘要", "公告", "说明"])
                    )[:200]
                    result["source_chain"].append(f"earnings_quick:{quick_source}")

            # Dividend details (cash dividend, pre-tax)
            dividend_df, dividend_source, dividend_errors = self._call_df_candidates([
                ("stock_fhps_detail_em", {"symbol": stock_code}),
                ("stock_history_dividend_detail", {"symbol": stock_code, "indicator": "分红", "date": ""}),
                ("stock_dividend_cninfo", {"symbol": stock_code}),
            ])
            result["errors"].extend(dividend_errors)
            if dividend_df is not None:
                dividend_payload = _build_dividend_payload(dividend_df, stock_code, max_events=5)
                if dividend_payload:
                    result["earnings"]["dividend"] = dividend_payload
                    result["source_chain"].append(f"dividend:{dividend_source}")

        # Institution / top shareholders
        inst_df, inst_source, inst_errors = self._call_df_candidates([
            ("stock_institute_hold", {}),
            ("stock_institute_recommend", {}),
        ])
        result["errors"].extend(inst_errors)
        if inst_df is not None:
            row = _extract_latest_row(inst_df, stock_code)
            if row is not None:
                inst_change = _safe_float(_pick_by_keywords(row, ["增减", "变化", "变动", "持股变化"]))
                result["institution"]["institution_holding_change"] = inst_change
                result["source_chain"].append(f"institution:{inst_source}")

        top10_df, top10_source, top10_errors = self._call_df_candidates([
            ("stock_gdfx_top_10_em", {"symbol": stock_code}),
            ("stock_gdfx_top_10_em", {}),
            ("stock_zh_a_gdhs_detail_em", {"symbol": stock_code}),
            ("stock_zh_a_gdhs_detail_em", {}),
        ])
        result["errors"].extend(top10_errors)
        if top10_df is not None:
            row = _extract_latest_row(top10_df, stock_code)
            if row is not None:
                holder_change = _safe_float(_pick_by_keywords(row, ["增减", "变化", "持股变化", "变动"]))
                result["institution"]["top10_holder_change"] = holder_change
                result["source_chain"].append(f"top10:{top10_source}")

        has_content = bool(result["growth"] or result["earnings"] or result["institution"])
        result["status"] = "partial" if has_content else "not_supported"
        return result

    def get_capital_flow(
        self,
        stock_code: str,
        top_n: int = 5,
        stock_flow: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Return stock + sector capital flow.

        stock_flow: 预取的个股资金流（当前来自 Tushare moneyflow），字段与 akshare
        口径对齐（金额单位：元）。非空时跳过 akshare 个股资金流候选链，板块排行
        仍走 akshare（fail-open）；为 None 或空时按原有 akshare 候选链获取。
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "stock_flow": {},
            "sector_rankings": {"top": [], "bottom": []},
            "source_chain": [],
            "errors": [],
        }

        if stock_flow and any(v is not None for v in stock_flow.values()):
            result["stock_flow"] = dict(stock_flow)
            result["source_chain"].append("capital_stock:tushare_moneyflow")
        else:
            stock_df, stock_source, stock_errors = self._call_df_candidates([
                ("stock_individual_fund_flow", {"stock": stock_code}),
                ("stock_individual_fund_flow", {"symbol": stock_code}),
                ("stock_individual_fund_flow", {}),
                ("stock_main_fund_flow", {"symbol": stock_code}),
                ("stock_main_fund_flow", {}),
            ])
            result["errors"].extend(stock_errors)
            if stock_df is not None:
                row = _extract_latest_row(stock_df, stock_code)
                if row is not None:
                    net_inflow = _safe_float(_pick_by_keywords(row, ["主力净流入", "净流入", "净额"]))
                    inflow_5d = _safe_float(_pick_by_keywords(row, ["5日", "五日"]))
                    inflow_10d = _safe_float(_pick_by_keywords(row, ["10日", "十日"]))
                    result["stock_flow"] = {
                        "main_net_inflow": net_inflow,
                        "inflow_5d": inflow_5d,
                        "inflow_10d": inflow_10d,
                    }
                    result["source_chain"].append(f"capital_stock:{stock_source}")

        sector_df, sector_source, sector_errors = self._call_df_candidates([
            ("stock_sector_fund_flow_rank", {}),
            ("stock_sector_fund_flow_summary", {}),
        ])
        result["errors"].extend(sector_errors)
        if sector_df is not None:
            name_col = next((c for c in sector_df.columns if any(k in str(c) for k in ("板块", "行业", "名称", "name"))), None)
            flow_col = next((c for c in sector_df.columns if any(k in str(c) for k in ("净流入", "主力", "flow", "净额"))), None)
            if name_col and flow_col:
                work_df = sector_df[[name_col, flow_col]].copy()
                work_df[flow_col] = pd.to_numeric(work_df[flow_col], errors="coerce")
                work_df = work_df.dropna(subset=[flow_col])
                top_df = work_df.nlargest(top_n, flow_col)
                bottom_df = work_df.nsmallest(top_n, flow_col)
                result["sector_rankings"] = {
                    "top": [{"name": _safe_str(r[name_col]), "net_inflow": float(r[flow_col])} for _, r in top_df.iterrows()],
                    "bottom": [{"name": _safe_str(r[name_col]), "net_inflow": float(r[flow_col])} for _, r in bottom_df.iterrows()],
                }
                result["source_chain"].append(f"capital_sector:{sector_source}")

        has_content = bool(result["stock_flow"] or result["sector_rankings"]["top"] or result["sector_rankings"]["bottom"])
        result["status"] = "partial" if has_content else "not_supported"
        return result

    def get_dragon_tiger_flag(self, stock_code: str, lookback_days: int = 20) -> Dict[str, Any]:
        """
        Return dragon-tiger signal in lookback window.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "is_on_list": False,
            "recent_count": 0,
            "latest_date": None,
            "source_chain": [],
            "errors": [],
        }

        df, source, errors = self._call_df_candidates([
            ("stock_lhb_stock_statistic_em", {}),
            ("stock_lhb_detail_em", {}),
            ("stock_lhb_jgmmtj_em", {}),
        ])
        result["errors"].extend(errors)
        if df is None:
            return result

        # Try code filter
        code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码"))]
        target = _normalize_code(stock_code)
        matched = pd.DataFrame()
        for col in code_cols:
            try:
                series = df[col].astype(str).map(_normalize_code)
                cur = df[series == target]
                if not cur.empty:
                    matched = cur
                    break
            except Exception:
                continue
        if matched.empty:
            result["source_chain"].append(f"dragon_tiger:{source}")
            result["status"] = "ok" if code_cols else "partial"
            return result

        date_col = next((c for c in matched.columns if any(k in str(c) for k in ("日期", "上榜", "交易日", "time"))), None)
        parsed_dates: List[datetime] = []
        if date_col is not None:
            for val in matched[date_col].astype(str).tolist():
                try:
                    parsed_dates.append(pd.to_datetime(val).to_pydatetime())
                except Exception:
                    continue
        now = datetime.now()
        start = now - timedelta(days=max(1, lookback_days))
        recent_dates = [d for d in parsed_dates if start <= d <= now]

        result["is_on_list"] = bool(recent_dates)
        result["recent_count"] = len(recent_dates) if recent_dates else int(len(matched))
        result["latest_date"] = max(recent_dates).date().isoformat() if recent_dates else (
            max(parsed_dates).date().isoformat() if parsed_dates else None
        )
        result["status"] = "ok"
        result["source_chain"].append(f"dragon_tiger:{source}")
        return result
