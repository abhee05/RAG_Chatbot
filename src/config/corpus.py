from dataclasses import dataclass


@dataclass(frozen=True)
class FundSource:
    category: str
    fund_name: str
    source_url: str

    @property
    def slug(self) -> str:
        return self.source_url.rstrip("/").split("/")[-1]


CORPUS: tuple[FundSource, ...] = (
    FundSource(
        category="Large-cap",
        fund_name="HDFC Large Cap Fund Direct Growth",
        source_url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    ),
    FundSource(
        category="Flexi-cap",
        fund_name="HDFC Flexi Cap Fund Direct Growth",
        source_url="https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
    ),
    FundSource(
        category="ELSS",
        fund_name="HDFC ELSS Tax Saver Fund Direct Plan Growth",
        source_url="https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    ),
    FundSource(
        category="Small-cap",
        fund_name="HDFC Small Cap Fund Direct Growth",
        source_url="https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    ),
    FundSource(
        category="Hybrid",
        fund_name="HDFC Balanced Advantage Fund Direct Growth",
        source_url="https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
    ),
)

ALLOWED_SOURCE_URLS = frozenset(fund.source_url for fund in CORPUS)
