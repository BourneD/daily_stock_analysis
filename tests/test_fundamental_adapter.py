# -*- coding: utf-8 -*-
"""
Tests for fundamental adapter helpers.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.fundamental_adapter import (
    AkshareFundamentalAdapter,
    _build_dividend_payload,
    _extract_latest_row,
    _parse_dividend_plan_to_per_share,
    build_financial_bundle_from_tushare,
)


class TestFundamentalAdapter(unittest.TestCase):
    def test_parse_dividend_plan_to_per_share_supports_cn_patterns(self) -> None:
        self.assertAlmostEqual(_parse_dividend_plan_to_per_share("10派3元(含税)"), 0.3, places=6)
        self.assertAlmostEqual(_parse_dividend_plan_to_per_share("每10股派发2.5元"), 0.25, places=6)
        self.assertAlmostEqual(_parse_dividend_plan_to_per_share("每股派0.8元"), 0.8, places=6)
        self.assertIsNone(_parse_dividend_plan_to_per_share("仅送股，不现金分红"))

    def test_extract_latest_row_returns_none_when_code_mismatch(self) -> None:
        df = pd.DataFrame(
            {
                "股票代码": ["600000", "000001"],
                "值": [1, 2],
            }
        )
        row = _extract_latest_row(df, "600519")
        self.assertIsNone(row)

    def test_extract_latest_row_fallback_when_no_code_column(self) -> None:
        df = pd.DataFrame({"值": [1, 2]})
        row = _extract_latest_row(df, "600519")
        self.assertIsNotNone(row)
        self.assertEqual(row["值"], 1)

    def test_capital_flow_uses_prefetched_stock_flow_and_skips_akshare_stock_candidates(self) -> None:
        adapter = AkshareFundamentalAdapter()
        calls = []

        def _fake_call_df_candidates(candidates):
            calls.append([name for name, _kwargs in candidates])
            return None, None, []

        prefetched = {"main_net_inflow": 12345.0, "inflow_5d": -2000.0, "inflow_10d": 500.0}
        with patch.object(adapter, "_call_df_candidates", side_effect=_fake_call_df_candidates):
            result = adapter.get_capital_flow("600519", stock_flow=prefetched)

        self.assertEqual(result["stock_flow"], prefetched)
        self.assertEqual(result["source_chain"], ["capital_stock:tushare_moneyflow"])
        self.assertEqual(result["status"], "partial")
        # 只调用板块排行候选，不再尝试 akshare 个股资金流
        self.assertEqual(calls, [["stock_sector_fund_flow_rank", "stock_sector_fund_flow_summary"]])

    def test_capital_flow_empty_prefetched_falls_back_to_akshare_candidates(self) -> None:
        adapter = AkshareFundamentalAdapter()
        with patch.object(adapter, "_call_df_candidates", return_value=(None, None, [])):
            result = adapter.get_capital_flow("600519", stock_flow={})
        self.assertEqual(result["stock_flow"], {})
        self.assertEqual(result["source_chain"], [])
        self.assertEqual(result["status"], "not_supported")

    def test_fundamental_bundle_uses_prefetched_tushare_financials(self) -> None:
        """预取财报 payload 非空时，跳过 akshare 财报候选链，仅机构/十大股东走 akshare。"""
        adapter = AkshareFundamentalAdapter()
        calls = []

        def _fake_call_df_candidates(candidates):
            calls.append([name for name, _kwargs in candidates])
            return None, None, []

        prefetched = {
            "status": "partial",
            "growth": {"revenue_yoy": 12.0, "net_profit_yoy": 9.5, "roe": 18.2, "gross_margin": 40.1},
            "earnings": {
                "financial_report": {
                    "report_date": "2026-06-30",
                    "revenue": 1.0e10,
                    "net_profit_parent": 3.0e9,
                    "operating_cash_flow": 5.0e9,
                    "roe": 18.2,
                }
            },
            "source_chain": ["growth:tushare_fina_indicator", "earnings_financial:tushare_financials"],
            "errors": [],
        }
        with patch.object(adapter, "_call_df_candidates", side_effect=_fake_call_df_candidates):
            result = adapter.get_fundamental_bundle("600519", financial_bundle=prefetched)

        self.assertEqual(result["growth"], prefetched["growth"])
        self.assertEqual(result["earnings"]["financial_report"]["revenue"], 1.0e10)
        self.assertEqual(result["source_chain"], prefetched["source_chain"])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(calls, [
            ["stock_institute_hold", "stock_institute_recommend"],
            ["stock_gdfx_top_10_em", "stock_gdfx_top_10_em", "stock_zh_a_gdhs_detail_em", "stock_zh_a_gdhs_detail_em"],
        ])

    def test_fundamental_bundle_prefetched_without_core_falls_back_to_akshare(self) -> None:
        """只有预告没有核心财报时，走回完整 akshare 候选链。"""
        adapter = AkshareFundamentalAdapter()
        calls = []

        def _fake_call_df_candidates(candidates):
            calls.append([name for name, _kwargs in candidates])
            return None, None, []

        prefetched = {"status": "partial", "growth": {}, "earnings": {"forecast_summary": "预增"}, "source_chain": [], "errors": []}
        with patch.object(adapter, "_call_df_candidates", side_effect=_fake_call_df_candidates):
            result = adapter.get_fundamental_bundle("600519", financial_bundle=prefetched)

        self.assertEqual(result["status"], "not_supported")
        self.assertEqual(calls[0][0], "stock_financial_abstract")
        self.assertEqual(len(calls), 6)

    def test_build_financial_bundle_from_tushare(self) -> None:
        """Tushare 财报原始数据映射为 akshare 口径 payload（金额单位元、分红按每股解析）。"""
        frames = {
            "fina_indicator": pd.DataFrame({
                "end_date": ["20260630", "20260331"],
                "roe": [18.2, 4.5],
                "grossprofit_margin": [40.1, 39.0],
                "or_yoy": [12.0, 8.0],
                "netprofit_yoy": [9.5, 6.0],
            }),
            "income": pd.DataFrame({
                "end_date": ["20260630"],
                "total_revenue": [1.0e10],
                "n_income_attr_p": [3.0e9],
            }),
            "cashflow": pd.DataFrame({
                "end_date": ["20260630"],
                "n_cashflow_act": [5.0e9],
            }),
            "forecast": pd.DataFrame({
                "end_date": ["20261231"],
                "type": ["预增"],
                "summary": ["业绩大幅增长"],
            }),
            "express": pd.DataFrame({
                "end_date": ["20260630"],
                "revenue": [1.0e10],
                "n_income": [3.0e9],
                "yoy_net_profit": [25.5],
                "perf_summary": ["经营稳健"],
            }),
            "dividend": pd.DataFrame({
                "end_date": ["20251231"],
                "ex_date": ["20260710"],
                "cash_div_tax": [0.8],
            }),
        }
        bundle = build_financial_bundle_from_tushare(frames, "600519")

        self.assertEqual(bundle["status"], "partial")
        self.assertEqual(bundle["growth"]["roe"], 18.2)
        self.assertEqual(bundle["growth"]["revenue_yoy"], 12.0)
        self.assertEqual(bundle["growth"]["gross_margin"], 40.1)
        financial_report = bundle["earnings"]["financial_report"]
        self.assertEqual(financial_report["report_date"], "2026-06-30")
        self.assertEqual(financial_report["revenue"], 1.0e10)
        self.assertEqual(financial_report["net_profit_parent"], 3.0e9)
        self.assertEqual(financial_report["operating_cash_flow"], 5.0e9)
        self.assertIn("预增", bundle["earnings"]["forecast_summary"])
        self.assertIn("业绩快报", bundle["earnings"]["quick_report_summary"])
        dividend = bundle["earnings"]["dividend"]
        self.assertEqual(dividend["events"][0]["cash_dividend_per_share"], 0.8)
        self.assertIn("growth:tushare_fina_indicator", bundle["source_chain"])
        self.assertIn("earnings_financial:tushare_financials", bundle["source_chain"])
        self.assertIn("dividend:tushare_dividend", bundle["source_chain"])

    def test_build_financial_bundle_from_tushare_prefers_consolidated_rows(self) -> None:
        """同一报告期存在母公司/合并口径时，优先取合并报表（comp_type==1）。"""
        frames = {
            "income": pd.DataFrame({
                "end_date": ["20260630", "20260630"],
                "comp_type": [4, 1],
                "total_revenue": [5.0e9, 1.0e10],
            }),
        }
        bundle = build_financial_bundle_from_tushare(frames, "600519")
        self.assertEqual(bundle["earnings"]["financial_report"]["revenue"], 1.0e10)

    def test_build_financial_bundle_from_tushare_empty_returns_not_supported(self) -> None:
        bundle = build_financial_bundle_from_tushare({}, "600519")
        self.assertEqual(bundle["status"], "not_supported")
        self.assertEqual(bundle["growth"], {})
        self.assertEqual(bundle["earnings"], {})

    def test_dragon_tiger_no_match_with_code_column_is_ok(self) -> None:
        adapter = AkshareFundamentalAdapter()
        df = pd.DataFrame(
            {
                "股票代码": ["600000"],
                "日期": ["2026-01-01"],
            }
        )
        with patch.object(adapter, "_call_df_candidates", return_value=(df, "stock_lhb_stock_statistic_em", [])):
            result = adapter.get_dragon_tiger_flag("600519")
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["is_on_list"])
        self.assertEqual(result["recent_count"], 0)

    def test_dragon_tiger_match_is_ok(self) -> None:
        adapter = AkshareFundamentalAdapter()
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["600519"],
                "日期": [today],
            }
        )
        with patch.object(adapter, "_call_df_candidates", return_value=(df, "stock_lhb_stock_statistic_em", [])):
            result = adapter.get_dragon_tiger_flag("600519")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["is_on_list"])
        self.assertGreaterEqual(result["recent_count"], 1)

    def test_fundamental_bundle_includes_financial_report_and_dividend_payload(self) -> None:
        adapter = AkshareFundamentalAdapter()
        now = datetime.now()
        within_ttm = (now - timedelta(days=30)).strftime("%Y-%m-%d")
        future_day = (now + timedelta(days=10)).strftime("%Y-%m-%d")
        old_day = (now - timedelta(days=500)).strftime("%Y-%m-%d")
        fin_df = pd.DataFrame(
            {
                "股票代码": ["600519"],
                "报告期": [within_ttm],
                "营业总收入": [1000.0],
                "归母净利润": [300.0],
                "经营活动产生的现金流量净额": [500.0],
                "净资产收益率": [18.2],
                "营业收入同比": [12.0],
                "净利润同比": [9.5],
            }
        )
        forecast_df = pd.DataFrame({"股票代码": ["600519"], "预告": ["预增"]})
        quick_df = pd.DataFrame({"股票代码": ["600519"], "快报": ["快报摘要"]})
        dividend_df = pd.DataFrame(
            {
                "股票代码": ["600519", "600519", "600519", "600519"],
                "除息日": [within_ttm, within_ttm, future_day, old_day],
                "分配方案": ["10派3元(含税)", "10派3元(含税)", "10派5元", "10派1元"],
            }
        )

        with patch.object(
            adapter,
            "_call_df_candidates",
            side_effect=[
                (fin_df, "stock_financial_abstract", []),
                (forecast_df, "stock_yjyg_em", []),
                (quick_df, "stock_yjkb_em", []),
                (dividend_df, "stock_fhps_detail_em", []),
                (None, None, []),
                (None, None, []),
            ],
        ):
            result = adapter.get_fundamental_bundle("600519")

        financial_report = result["earnings"].get("financial_report", {})
        self.assertEqual(financial_report.get("report_date"), within_ttm)
        self.assertEqual(financial_report.get("revenue"), 1000.0)
        self.assertEqual(financial_report.get("net_profit_parent"), 300.0)
        self.assertEqual(financial_report.get("operating_cash_flow"), 500.0)
        self.assertEqual(financial_report.get("roe"), 18.2)

        dividend_payload = result["earnings"].get("dividend", {})
        events = dividend_payload.get("events", [])
        self.assertEqual(len(events), 2)  # duplicate + future day filtered
        self.assertEqual(dividend_payload.get("ttm_event_count"), 1)
        self.assertAlmostEqual(dividend_payload.get("ttm_cash_dividend_per_share"), 0.3, places=6)

    def test_build_dividend_payload_returns_empty_when_code_not_matched(self) -> None:
        now = datetime.now().strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["000001"],
                "除息日": [now],
                "分配方案": ["10派3元(含税)"],
            }
        )

        payload = _build_dividend_payload(df, stock_code="600519")
        self.assertEqual(payload, {})

    def test_build_dividend_payload_skips_after_tax_plan(self) -> None:
        now = datetime.now().strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["600519"],
                "除息日": [now],
                "分配方案": ["10派3元(税后)"],
            }
        )

        payload = _build_dividend_payload(df, stock_code="600519")
        self.assertEqual(payload, {})

    def test_build_dividend_payload_ttm_window_boundary(self) -> None:
        now = datetime.now()
        day_365 = (now - timedelta(days=365)).strftime("%Y-%m-%d")
        day_366 = (now - timedelta(days=366)).strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["600519", "600519"],
                "除息日": [day_365, day_366],
                "分配方案": ["10派3元(含税)", "10派5元(含税)"],
            }
        )

        payload = _build_dividend_payload(df, stock_code="600519")
        self.assertEqual(payload.get("ttm_event_count"), 1)
        self.assertAlmostEqual(payload.get("ttm_cash_dividend_per_share"), 0.3, places=6)


if __name__ == "__main__":
    unittest.main()
