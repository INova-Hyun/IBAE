__all__ = [
    "JetAnalyzerV8Simple",
    "run_v8_preview",
    "JetAnalyzerV9Qt",
    "run_v9_preview",
    "QtTrendReportCompareDialog",
    "run_trend_report_compare_qt",
]


def __getattr__(name: str):
    if name in {"JetAnalyzerV8Simple", "run_v8_preview"}:
        from .analyzer_v8 import JetAnalyzerV8Simple, run_v8_preview

        return {"JetAnalyzerV8Simple": JetAnalyzerV8Simple, "run_v8_preview": run_v8_preview}[name]
    if name in {"JetAnalyzerV9Qt", "run_v9_preview"}:
        from .analyzer_v9_qt import JetAnalyzerV9Qt, run_v9_preview

        return {"JetAnalyzerV9Qt": JetAnalyzerV9Qt, "run_v9_preview": run_v9_preview}[name]
    if name in {"QtTrendReportCompareDialog", "run_trend_report_compare_qt"}:
        from .trend_report_compare_qt import QtTrendReportCompareDialog, run_trend_report_compare_qt

        return {
            "QtTrendReportCompareDialog": QtTrendReportCompareDialog,
            "run_trend_report_compare_qt": run_trend_report_compare_qt,
        }[name]
    raise AttributeError(name)
