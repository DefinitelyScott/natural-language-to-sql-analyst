"""Resolve a natural-language question to SQL.

Two backends:

* ``OfflineBackend`` — a deterministic, rule-based matcher over a catalog of
  known analytical question patterns. Requires no network or API key, so the
  test suite and CI use it. It is intentionally small and transparent.
* ``LLMBackend`` — sends the schema + question to an OpenAI-compatible chat
  model and returns the SQL it produces. Used when ``OPENAI_API_KEY`` is set.

Both return a raw SQL string; validation and execution happen in ``runner``.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol, runtime_checkable


class Backend(Protocol):
    def to_sql(self, question: str, schema: str) -> str: ...


@runtime_checkable
class RepairingBackend(Backend, Protocol):
    """A backend that can revise SQL it wrote, given the error it raised.

    Optional: a backend is repairable only if implementing ``repair`` can
    actually change the outcome. ``LLMBackend`` qualifies — a model that
    hallucinated a column name will often fix it when shown the error.
    ``OfflineBackend`` deliberately does not: its SQL is hand-written and
    keyed to a fixed rule, so re-asking the same question returns the same
    string and a retry could only burn time. ``generator.answer_question``
    checks for this protocol at runtime (``isinstance``, which for a
    ``runtime_checkable`` Protocol tests only that the methods exist) and
    skips the repair step entirely when a backend does not implement it.

    It extends :class:`Backend` rather than standing alone because repairing is
    a capability *added to* generating, never a substitute for it: every holder
    of one of these calls ``to_sql`` on it too — ``cache.CachingBackend`` stores
    a ``RepairingBackend`` and immediately generates through it. Declaring only
    ``repair`` made that a type error the checker was right to flag, and the fix
    is to state the real contract rather than to widen the annotation. The
    runtime check tightens with it: ``isinstance`` now requires both methods, so
    a class with ``repair`` but no ``to_sql`` no longer passes as repairable.
    """

    def repair(self, question: str, schema: str, sql: str, error: str) -> str: ...


class NoRuleMatchError(ValueError):
    """No rule in the offline catalog matches the question.

    Subclasses :class:`ValueError` so existing callers that catch ``ValueError``
    around ``to_sql`` keep working unchanged. It exists as its own type so the
    CLI can distinguish "the catalog does not cover this question" — the one
    failure a nearest-question suggestion can help with — from every other
    ``ValueError`` a backend might raise, without matching on message text.
    """


#: Phrases that scope a question to a period — a relative window ("in the last
#: 7 days", "this quarter", "past few months") or an explicit calendar year
#: ("in 2023"). Used only as the body of :data:`_UNSCOPED_ONLY`.
#:
#: The single optional ``\w+`` between the qualifier and the unit is what lets
#: one alternative cover "last month", "last 7 days" and "past few weeks"
#: without enumerating the fillers.
_PERIOD_SCOPED = (
    r"\b(?:last|past|previous|this|current)\s+(?:\w+\s+)?"
    r"(?:day|week|month|quarter|year)s?\b"
    r"|\b(?:19|20)\d{2}\b"
)

#: Prefix for a rule whose SQL aggregates over a whole table with no date
#: filter, e.g. ``SELECT COUNT(*) FROM orders``.
#:
#: Those rules are registered last and phrased broadly on purpose ("how many
#: orders"), so a question that *scopes* the same aggregate to a period falls
#: through to them and gets answered with the unscoped total. The answer looks
#: right — it is a plausible number, correctly labelled "order_count" — while
#: silently ignoring the window the user asked about, which is the most
#: dangerous way for this catalog to be wrong.
#:
#: A negative lookahead anchored at ``\A`` is the narrowest fix available: the
#: rule keeps matching every unscoped phrasing it used to, and declines only
#: when the question carries a period the SQL does not honour. Declining routes
#: the question to ``NoRuleMatchError``, so the CLI says it has no rule and
#: suggests nearer catalog questions instead of inventing an answer.
#:
#: Periods the catalog *does* implement are unaffected: their rules ("orders in
#: the last 30 days", "revenue by month in 2024") are registered earlier and
#: win under first-match resolution before these are consulted.
_UNSCOPED_ONLY = rf"\A(?!.*(?:{_PERIOD_SCOPED}))"


# --------------------------------------------------------------------------- #
# Offline rule-based backend
# --------------------------------------------------------------------------- #
class OfflineBackend:
    """Map a question to SQL via lightweight keyword rules.

    This is not meant to be a general NL parser. It recognizes a fixed catalog
    of common analytics questions so the project is runnable and verifiable
    offline. Each rule is a (matcher, sql) pair.
    """

    def __init__(self) -> None:
        self._rules: list[tuple[re.Pattern[str], str]] = [
            # Monthly cohort retention: for each acquisition cohort (customers
            # grouped by the month of their *first* order), what share of that
            # cohort ordered again N months later. This is the standard retention
            # grid -- one row per (cohort, month offset) cell -- and it is the
            # only rule here that measures behavior *relative to each customer's
            # own start date* rather than against the calendar.
            #
            # It is registered first because its vocabulary (cohort / retention /
            # retained) is the narrowest in the catalog and no other rule uses
            # any of those words, so it cannot shadow anything -- while several
            # broad rules below would otherwise swallow it. "How many customers
            # are retained after their first purchase?" is a retention question,
            # but it also contains "how many customers", which the customer-count
            # rule matches; first-rule-wins ordering is what keeps it here.
            #
            # Three CTEs keep each step separable: ``first_order`` assigns every
            # customer their cohort; ``cohort_size`` is the denominator (the
            # cohort's headcount, computed once so it is not re-derived per
            # cell); ``activity`` lists the distinct months each customer was
            # active in, tagged with their cohort.
            #
            # The month offset is deliberately arithmetic on the 'YYYY-MM'
            # strings rather than a date function: SQLite has no month-difference
            # builtin, and julianday() measures *days*, which would make offsets
            # drift across months of unequal length. Converting each month to an
            # absolute month number (year * 12 + month) and subtracting gives an
            # exact whole-month distance. COUNT(DISTINCT customer_id) is used
            # rather than COUNT(*) so the numerator is unambiguously "customers",
            # independent of how ``activity`` happens to be deduplicated.
            #
            # Offset 0 is every cohort's own first month, so its retention is
            # 100% by construction -- that row is the baseline the later
            # percentages are read against, not a result.
            (
                re.compile(r"\bcohorts?\b|\bretention\b|\bretained\b", re.I),
                """
                WITH first_order AS (
                    SELECT customer_id,
                           strftime('%Y-%m', MIN(order_date)) AS cohort_month
                    FROM orders
                    GROUP BY customer_id
                ),
                cohort_size AS (
                    SELECT cohort_month, COUNT(*) AS customers
                    FROM first_order
                    GROUP BY cohort_month
                ),
                activity AS (
                    SELECT DISTINCT f.cohort_month AS cohort_month,
                           o.customer_id AS customer_id,
                           strftime('%Y-%m', o.order_date) AS active_month
                    FROM orders o
                    JOIN first_order f ON f.customer_id = o.customer_id
                )
                SELECT a.cohort_month,
                       (CAST(substr(a.active_month, 1, 4) AS INTEGER) * 12
                        + CAST(substr(a.active_month, 6, 2) AS INTEGER))
                       - (CAST(substr(a.cohort_month, 1, 4) AS INTEGER) * 12
                          + CAST(substr(a.cohort_month, 6, 2) AS INTEGER))
                           AS month_offset,
                       s.customers AS cohort_size,
                       COUNT(DISTINCT a.customer_id) AS active_customers,
                       ROUND(
                           100.0 * COUNT(DISTINCT a.customer_id) / s.customers, 1
                       ) AS retention_pct
                FROM activity a
                JOIN cohort_size s ON s.cohort_month = a.cohort_month
                GROUP BY a.cohort_month, month_offset
                ORDER BY a.cohort_month, month_offset
                """,
            ),
            # Time to first order, by signup cohort: for each month of signups,
            # how many of those customers ever placed an order, and how long the
            # ones who did took to do it. This is the activation view -- the only
            # rule in the catalog anchored on ``customers.signup_date`` as a
            # per-customer clock rather than as a calendar bucket to count in.
            #
            # Registered second, immediately after cohort retention, for the same
            # reason: its vocabulary is narrow (a duration word plus "first
            # order"/"first purchase", or "activation rate") and no other rule
            # uses it, while broader rules further down would otherwise swallow
            # phrasings like "how long before a new customer places their first
            # order". Retention keeps priority over it because a question that
            # says both "retained" and "first purchase" is a retention question.
            #
            # LEFT JOIN, not JOIN: a customer who never ordered is the entire
            # point of the activation rate, and an inner join would silently drop
            # them and report 100% for every month. COUNT(*) therefore counts
            # signups and COUNT(f.customer_id) counts the subset that converted,
            # since COUNT of a column skips NULLs.
            #
            # avg_days_to_first_order is likewise an average over converted
            # customers only, for the same NULL-skipping reason -- there is no
            # defensible number of days to attribute to someone who has not
            # ordered yet. It is NULL for a month in which nobody converted.
            #
            # Reading caveat, and the reason this is grouped by signup month at
            # all: the most recent cohorts are censored. Someone who signed up
            # weeks before the last order in the database has had far less time
            # to convert than a cohort from a year earlier, so a declining
            # activation_pct down the final rows is an artifact of the window,
            # not a trend. The same goes the other way: a cohort that signed up
            # before the earliest order in the database carries the gap to that
            # date inside its average, which is why the 2023 cohorts of the
            # sample data average far more days than the 2024 ones. Splitting by
            # cohort is what makes both effects legible; a single blended
            # average would bury them in one number.
            (
                re.compile(
                    r"\b(?:time|days?|long|lag)\b[^?]*"
                    r"\bfirst\s+(?:order|purchase)\b"
                    r"|\bsign\s?-?up\s+to\s+(?:their\s+)?first\s+"
                    r"(?:order|purchase)\b"
                    r"|\bactivation\s+rate\b",
                    re.I,
                ),
                """
                WITH first_order AS (
                    SELECT customer_id, MIN(order_date) AS first_order_date
                    FROM orders
                    GROUP BY customer_id
                )
                SELECT strftime('%Y-%m', c.signup_date) AS signup_month,
                       COUNT(*) AS customers,
                       COUNT(f.customer_id) AS activated,
                       ROUND(100.0 * COUNT(f.customer_id) / COUNT(*), 1)
                           AS activation_pct,
                       ROUND(
                           AVG(julianday(f.first_order_date)
                               - julianday(c.signup_date)),
                           1
                       ) AS avg_days_to_first_order
                FROM customers c
                LEFT JOIN first_order f ON f.customer_id = c.id
                GROUP BY signup_month
                ORDER BY signup_month
                """,
            ),
            # Acquisition mix: which category each customer's *first* order came
            # from, aggregated into the share of the customer base each category
            # brought in. Activation above asks whether and how fast a signup
            # converts; this asks what they converted *on*, which is the question
            # behind deciding where to spend to win new customers.
            #
            # Registered third, behind the two rules above and ahead of everything
            # else, on the same narrow-vocabulary reasoning: it needs "category"
            # next to a first-purchase notion, which no other rule uses, while the
            # broad "revenue by category" and category-share rules further down
            # would otherwise swallow any phrasing containing the word. Retention
            # and activation keep priority because a question that says both
            # "retention" and "first order" is a retention question.
            #
            # Three CTEs, each doing one thing:
            #   * ``first_order`` picks one order per customer. ROW_NUMBER() over
            #     (order_date, id) rather than MIN(order_date), because two orders
            #     can share a date and MIN would then match both, silently
            #     double-counting that customer. The id tiebreaker makes the choice
            #     deterministic -- the same idiom the new-vs-returning rule uses.
            #   * ``category_spend`` totals that order's line items per category.
            #   * ``primary_category`` keeps the highest-spending category per
            #     customer, breaking ties by category name so the result is stable.
            #
            # That last step is an *attribution choice* and the one thing to defend
            # here: 78 of the 115 customers in the sample data have a first order
            # spanning more than one category, so "the" acquiring category does not
            # exist in the data and has to be defined. Attributing the customer to
            # where they spent the most treats the largest line as the reason for
            # the visit, and -- unlike counting the customer once per category
            # present -- partitions the base exactly once, so ``customers_acquired``
            # sums to the number of customers who have ordered and the percentages
            # sum to 100. The alternative is defensible too, but it yields a column
            # that sums to ~196% here, which reads as a bug in a report even when it
            # is not. The tradeoff is that a narrowly-lost second category is
            # invisible; this is a mix question, not a basket question, and the
            # market-basket rule already covers what else rides along in an order.
            (
                re.compile(
                    r"\bcategor(?:y|ies)\b[^?]*"
                    r"\b(?:buy|bought|purchase[ds]?|order(?:ed)?|start(?:s|ed)?)\b"
                    r"[^?]*\bfirst\b|"
                    r"\bfirst\b[^?]*\b(?:buy|bought|purchase[ds]?|order(?:ed)?)\b"
                    r"[^?]*\bcategor(?:y|ies)\b|"
                    r"\bfirst\s+(?:order|purchase)\b[^?]*\bcategor(?:y|ies)\b|"
                    r"\bcategor(?:y|ies)\b[^?]*\bfirst\s+(?:order|purchase)\b|"
                    r"\bacquisition\s+categor(?:y|ies)\b|"
                    r"\bcategor(?:y|ies)\b[^?]*\bacquires?\b",
                    re.I,
                ),
                """
                WITH first_order AS (
                    SELECT order_id, customer_id
                    FROM (
                        SELECT o.id AS order_id,
                               o.customer_id AS customer_id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY o.customer_id
                                   ORDER BY o.order_date, o.id
                               ) AS seq
                        FROM orders o
                    )
                    WHERE seq = 1
                ),
                category_spend AS (
                    SELECT f.customer_id AS customer_id,
                           p.category AS category,
                           SUM(oi.quantity * oi.unit_price) AS category_total
                    FROM first_order f
                    JOIN order_items oi ON oi.order_id = f.order_id
                    JOIN products p ON p.id = oi.product_id
                    GROUP BY f.customer_id, p.category
                ),
                primary_category AS (
                    SELECT customer_id, category
                    FROM (
                        SELECT customer_id,
                               category,
                               ROW_NUMBER() OVER (
                                   PARTITION BY customer_id
                                   ORDER BY category_total DESC, category
                               ) AS spend_rank
                        FROM category_spend
                    )
                    WHERE spend_rank = 1
                )
                SELECT category,
                       COUNT(*) AS customers_acquired,
                       ROUND(
                           100.0 * COUNT(*)
                           / (SELECT COUNT(*) FROM primary_category), 1
                       ) AS pct_of_customers
                FROM primary_category
                GROUP BY category
                ORDER BY customers_acquired DESC, category
                """,
            ),
            (
                re.compile(r"total sales by month.*2024", re.I),
                """
                SELECT strftime('%Y-%m', o.order_date) AS month,
                       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                FROM orders o
                JOIN order_items oi ON oi.order_id = o.id
                WHERE o.order_date >= '2024-01-01' AND o.order_date < '2025-01-01'
                GROUP BY month
                ORDER BY month
                """,
            ),
            # Monthly revenue per category -- a category time series, not the
            # single-number-per-category breakdown the plain "revenue by
            # category" rule returns. Analysts ask this to see *which* category
            # is carrying (or dragging) a total that looks flat overall, so the
            # per-category month-over-month change is reported alongside the
            # level.
            #
            # It is registered here, ahead of both the month-over-month growth
            # rule below and the plain "revenue by category" rule further down,
            # because either would otherwise answer these questions with the
            # wrong shape: "monthly revenue by category" currently matched the
            # category rule and silently dropped the month dimension, and
            # "revenue by category month over month" matched the growth rule and
            # silently dropped the category dimension. Both returned a plausible
            # table, which is what makes the shadowing worth guarding: the
            # answer looks right until you notice a dimension is missing. The
            # pattern demands a category word *and* a monthly/trend word, in
            # either order, so a single-dimension question still falls through
            # to its own rule.
            #
            # ``LAG`` is partitioned by category so each category's series is
            # compared against itself; without the PARTITION BY the lag would
            # walk across the category boundary and compare, say, Fitness's
            # January against Electronics' December. The prior month is lifted
            # into its own CTE column rather than repeating the window
            # expression, so the absolute and percentage change are provably
            # derived from the same prior value. The first month of each
            # category is NULL in both change columns -- there is no prior month
            # to compare against, and NULL is the honest answer rather than 0.
            #
            # A category with no sales in some month produces no row for it, so
            # the lag then reaches back to the last month that *did* have sales.
            # On this dataset every category sells every month, so it does not
            # arise; the ``month`` column is returned precisely so a reader can
            # see a gap rather than have it silently smoothed over.
            (
                re.compile(
                    r"categor(?:y|ies).*(?:monthly|by\s+month|per\s+month|"
                    r"each\s+month|over\s+time|month[- ]over[- ]month|trend)|"
                    r"(?:monthly|by\s+month|per\s+month|each\s+month|over\s+time|"
                    r"month[- ]over[- ]month|trend).*categor(?:y|ies)",
                    re.I,
                ),
                """
                WITH monthly AS (
                    SELECT p.category AS category,
                           strftime('%Y-%m', o.order_date) AS month,
                           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    JOIN products p ON p.id = oi.product_id
                    WHERE o.order_date >= '2024-01-01' AND o.order_date < '2025-01-01'
                    GROUP BY category, month
                ),
                with_prior AS (
                    SELECT category,
                           month,
                           revenue,
                           LAG(revenue) OVER (
                               PARTITION BY category
                               ORDER BY month
                           ) AS prior_revenue
                    FROM monthly
                )
                SELECT category,
                       month,
                       revenue,
                       ROUND(revenue - prior_revenue, 2) AS revenue_change,
                       ROUND(100.0 * (revenue - prior_revenue) / prior_revenue, 1)
                           AS revenue_change_pct
                FROM with_prior
                ORDER BY category, month
                """,
            ),
            # Month-over-month revenue growth for the business as a whole. A
            # ``LAG`` window function over a monthly-revenue CTE yields each
            # month's change from the previous month; the first month's change is
            # NULL because there is no prior month to compare against. It is
            # placed ahead of the broad "total revenue" rule so "revenue growth"
            # phrasings are not shadowed by it, and behind the per-category rule
            # above so a question naming a category keeps that dimension.
            (
                re.compile(
                    r"month[- ]over[- ]month|(revenue|sales)\s+growth|"
                    r"growth.*(revenue|sales)",
                    re.I,
                ),
                """
                WITH monthly AS (
                    SELECT strftime('%Y-%m', o.order_date) AS month,
                           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    WHERE o.order_date >= '2024-01-01' AND o.order_date < '2025-01-01'
                    GROUP BY month
                )
                SELECT month,
                       revenue,
                       ROUND(revenue - LAG(revenue) OVER (ORDER BY month), 2)
                           AS revenue_change
                FROM monthly
                ORDER BY month
                """,
            ),
            # Cumulative (running-total) revenue by month for 2024. A first CTE
            # totals each month's revenue; the outer query then carries a running
            # sum with SUM(revenue) OVER (ORDER BY month ROWS BETWEEN UNBOUNDED
            # PRECEDING AND CURRENT ROW) -- an *explicit window frame* that sums
            # every month up to and including the current one. This differs from
            # the other window rules here: LAG (month-over-month) looks one row
            # back, and the category-share rule's SUM(...) OVER () has an empty,
            # unordered frame that spans the whole result; an ordered ROWS frame
            # is what turns a plain total into a progressive running total. The
            # final row's cumulative value therefore equals total 2024 revenue.
            # It requires a cumulative/running-total phrasing and is registered
            # ahead of the broad "total revenue" rule so it is not shadowed.
            (
                re.compile(
                    r"cumulative\s+(revenue|sales)|"
                    r"running\s+(total|sum)\s+(of\s+)?(revenue|sales)|"
                    r"(revenue|sales)\s+running\s+total|"
                    r"(revenue|sales)\s+to\s+date",
                    re.I,
                ),
                """
                WITH monthly AS (
                    SELECT strftime('%Y-%m', o.order_date) AS month,
                           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    WHERE o.order_date >= '2024-01-01' AND o.order_date < '2025-01-01'
                    GROUP BY month
                )
                SELECT month,
                       revenue,
                       ROUND(
                           SUM(revenue) OVER (
                               ORDER BY month
                               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                           ), 2
                       ) AS cumulative_revenue
                FROM monthly
                ORDER BY month
                """,
            ),
            # 7-day moving average of daily revenue: the smoothed daily trend.
            # Daily revenue is noisy (a handful of orders per day), so the
            # trailing average is what makes the direction readable. Three
            # idioms appear here and nowhere else in the catalog:
            #
            #   * A *recursive CTE* (``calendar``) generates one row per
            #     calendar day from the first order date to the last. It is
            #     needed for correctness, not decoration: a "7-day" average
            #     computed with ROWS over only the days that appear in the
            #     orders table would silently widen whenever a day had no
            #     orders — the frame counts rows, not days, so missing days
            #     make "6 PRECEDING" reach further back than a week. Anchoring
            #     the spine to MIN/MAX(order_date) keeps the result
            #     reproducible from the data alone, like the other
            #     data-anchored rules.
            #   * A *LEFT JOIN* from the spine to the per-day revenue, with
            #     COALESCE(revenue, 0) filling the gaps. An inner join would
            #     drop the zero-revenue days and reintroduce exactly the bug
            #     the spine exists to fix; a day with no orders contributes
            #     0 to the average, which is the honest reading.
            #   * A *bounded* window frame: ROWS BETWEEN 6 PRECEDING AND
            #     CURRENT ROW is a sliding 7-day window, in contrast to the
            #     cumulative rule's UNBOUNDED PRECEDING frame (which only ever
            #     grows) and the LAG rules (which look at a single prior row).
            #     Because the spine is gap-free, rows and days coincide and
            #     the frame really is one week.
            #
            # The first six rows average over fewer than seven days (the
            # window is truncated at the start of the data); SQL window
            # frames do this by construction and reporting a partial-window
            # average is the standard convention for a leading edge. The
            # matcher owns the moving/rolling/trailing-average and "smoothed"
            # vocabulary, which no other rule uses, so it neither shadows nor
            # is shadowed by the AOV or cumulative-revenue rules.
            (
                re.compile(
                    r"(moving|rolling|trailing)\s+(average|avg|mean)|"
                    r"\b(7|seven)[-\s]?day\s+(average|avg|mean)|"
                    r"smoothed\s+(daily\s+)?(revenue|sales)",
                    re.I,
                ),
                """
                WITH RECURSIVE calendar(day) AS (
                    SELECT (SELECT MIN(order_date) FROM orders)
                    UNION ALL
                    SELECT date(day, '+1 day')
                    FROM calendar
                    WHERE day < (SELECT MAX(order_date) FROM orders)
                ),
                daily AS (
                    SELECT o.order_date AS day,
                           SUM(oi.quantity * oi.unit_price) AS revenue
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY o.order_date
                )
                SELECT c.day,
                       ROUND(COALESCE(d.revenue, 0), 2) AS revenue,
                       ROUND(
                           AVG(COALESCE(d.revenue, 0)) OVER (
                               ORDER BY c.day
                               ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                           ), 2
                       ) AS avg_7day_revenue
                FROM calendar c
                LEFT JOIN daily d ON d.day = c.day
                ORDER BY c.day
                """,
            ),
            # The single highest-spending customer within each region: the classic
            # "greatest-N-per-group" (top-1-per-partition) problem. A first CTE
            # totals each customer's spend and carries their region; the second
            # ranks customers *inside* each region with
            # ROW_NUMBER() OVER (PARTITION BY region ORDER BY total_spent DESC);
            # the outer query then keeps only rank 1 per region. PARTITION BY
            # restarts the numbering for every region, which is what makes this a
            # per-group ranking rather than one global ranking -- distinct from
            # the other window rules here (LAG for month-over-month, SUM() OVER ()
            # for category share), neither of which partitions. customer_id is a
            # deterministic tiebreaker in the ORDER BY so ties resolve the same
            # way on every run. This rule requires a per-region phrase together
            # with a superlative about customer spend, and is registered ahead of
            # the plain "top 5 customers by spend" rule below so a per-region
            # question is not shadowed by the global top-spenders rule.
            #
            # The first two alternatives key on a top/best/highest word next to
            # "customer". The third exists because that vocabulary is not the only
            # way to ask: "for every region, which customer spent the most?" says
            # the same thing with "spent ... most" and no superlative adjective,
            # and used to fall through to the global top-spenders rule below --
            # which answered it with one overall ranking, silently dropping the
            # per-region grouping. Requiring "each/every/per region" *and*
            # "customer" *and* "spent/spend/spending" *and* "most" keeps the
            # branch narrow enough not to reclaim the global rule's questions,
            # none of which mention a region.
            (
                re.compile(
                    r"(top|best|highest)[-\s]*(spending|spender)?\s*customers?.*"
                    r"(in|per|by|within|for)\s+(each\s+)?region|"
                    r"region.*(top|best|highest)[-\s]*(spending|spender)?\s*customers?|"
                    r"(?:each|every|per)\s+regions?\b.*\bcustomers?\b.*\bspen[dt]"
                    r"(?:ing|s)?\b.*\bmost\b|"
                    r"\bcustomers?\b.*\bspen[dt](?:ing|s)?\b.*\bmost\b.*"
                    r"(?:in|per|within|for)\s+(?:each|every)\s+regions?\b",
                    re.I,
                ),
                """
                WITH customer_spend AS (
                    SELECT c.id AS customer_id,
                           c.name AS name,
                           c.region AS region,
                           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spent
                    FROM customers c
                    JOIN orders o ON o.customer_id = c.id
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY c.id
                ),
                ranked AS (
                    SELECT region,
                           name,
                           total_spent,
                           ROW_NUMBER() OVER (
                               PARTITION BY region
                               ORDER BY total_spent DESC, customer_id
                           ) AS rn
                    FROM customer_spend
                )
                SELECT region, name, total_spent
                FROM ranked
                WHERE rn = 1
                ORDER BY region
                """,
            ),
            # Segment customers into spend quartiles (four equal-sized tiers by
            # lifetime spend). A first CTE totals each customer's spend; the
            # second assigns a quartile with NTILE(4) OVER (ORDER BY total_spent
            # DESC, customer_id) -- NTILE splits the ordered rows into four
            # groups as equal in size as possible, so quartile 1 is the
            # top-spending 25% of customers and quartile 4 the bottom 25%. The
            # outer query then rolls each quartile up into a one-row summary
            # (customer count, total and average spend). It differs from the
            # ROW_NUMBER top-per-region rule (which ranks every row to pick a
            # single winner) in that NTILE only needs the bucket number, not a
            # full ranking; the RFM rule below is the only other NTILE user, and
            # it buckets on three measures rather than one. customer_id is a
            # deterministic tiebreaker so tied spends fall on the same side of a
            # bucket boundary on every run. The matcher requires a quartile/tier
            # or "segment customers by spend" phrasing, so it does not shadow the
            # plain top-spenders rule below it, and its "quartile" wording will
            # not collide with the "revenue by quarter" rule.
            (
                re.compile(
                    r"\bquartiles?\b|"
                    r"(spend|spending)\s+tiers?|"
                    r"segment\s+customers?\b.*\b(spend|spending|value|revenue)",
                    re.I,
                ),
                """
                WITH customer_spend AS (
                    SELECT c.id AS customer_id,
                           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spent
                    FROM customers c
                    JOIN orders o ON o.customer_id = c.id
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY c.id
                ),
                bucketed AS (
                    SELECT customer_id,
                           total_spent,
                           NTILE(4) OVER (ORDER BY total_spent DESC, customer_id)
                               AS quartile
                    FROM customer_spend
                )
                SELECT quartile,
                       COUNT(*) AS customers,
                       ROUND(SUM(total_spent), 2) AS total_spent,
                       ROUND(AVG(total_spent), 2) AS avg_spent
                FROM bucketed
                GROUP BY quartile
                ORDER BY quartile
                """,
            ),
            # Revenue concentration (Pareto / "80-20") across the customer base:
            # customers are ranked by lifetime revenue and split into five equal
            # groups with NTILE(5), then each quintile reports its revenue, its
            # share of total revenue, and the *cumulative* share through that
            # quintile. Reading down the last column answers the question the
            # quintiles exist for -- "what share of revenue comes from the top
            # 20% / 40% / ... of customers" -- in one pass.
            #
            # It shares NTILE with the spend-quartiles rule above but answers a
            # different question: quartiles report each tier's own spend (how
            # much a tier is worth), while this reports each tier's share of the
            # whole (how concentrated the business is). Concentration is the
            # question a running share answers and a per-bucket total does not,
            # which is why the cumulative column is here and not there.
            #
            # Two window functions over the same pre-aggregated CTE keep the
            # arithmetic readable: SUM(revenue) OVER () is the grand total (the
            # denominator for both percentages), and the same sum with an
            # explicit ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW frame,
            # ordered by quintile, is the running numerator. Aggregating into
            # ``per_quintile`` first means neither window has to nest an
            # aggregate inside itself.
            #
            # customer_id breaks ties in the NTILE ordering so two customers with
            # identical revenue always fall on the same side of a bucket boundary
            # across runs. The matcher owns the pareto / 80-20 / concentration /
            # quintile vocabulary, which no other rule uses -- note "quintile"
            # is deliberately distinct from the quartiles rule's "quartile".
            (
                re.compile(
                    r"\bpareto\b|"
                    r"\b80[/\-\s]?20\b|"
                    r"(revenue|sales|spend(?:ing)?)\s+concentration|"
                    r"concentration\s+of\s+(revenue|sales|spend(?:ing)?)|"
                    r"top\s*20\s*%|"
                    r"\bquintiles?\b",
                    re.I,
                ),
                """
                WITH customer_revenue AS (
                    SELECT o.customer_id AS customer_id,
                           SUM(oi.quantity * oi.unit_price) AS revenue
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY o.customer_id
                ),
                ranked AS (
                    SELECT revenue,
                           NTILE(5) OVER (ORDER BY revenue DESC, customer_id)
                               AS quintile
                    FROM customer_revenue
                ),
                per_quintile AS (
                    SELECT quintile,
                           COUNT(*) AS customers,
                           SUM(revenue) AS revenue
                    FROM ranked
                    GROUP BY quintile
                )
                SELECT quintile,
                       customers,
                       ROUND(revenue, 2) AS revenue,
                       ROUND(100.0 * revenue / SUM(revenue) OVER (), 2)
                           AS revenue_share_pct,
                       ROUND(
                           100.0 * SUM(revenue) OVER (
                               ORDER BY quintile
                               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                           ) / SUM(revenue) OVER (),
                           2
                       ) AS cumulative_share_pct
                FROM per_quintile
                ORDER BY quintile
                """,
            ),
            # RFM segmentation: score every buying customer on the three classic
            # customer-value dimensions at once -- Recency (how long since their
            # last order), Frequency (how many orders), Monetary (how much they
            # spent) -- and label them with the combined RFM cell. Two other rules
            # already look at one of these lenses in isolation: the at-risk rule
            # is recency-only (a filter), and the spend-quartile rule is
            # monetary-only (four buckets). This rule is what composes them, and
            # the composition is the point: a customer can be a heavy spender and
            # still be lapsing, which neither single-lens rule can express.
            #
            # ``customer_rfm`` reduces the order history to one row per customer
            # holding all three raw measures. Recency is measured against the
            # dataset's own MAX(order_date) rather than wall-clock today(), for
            # the same reproducibility reason as the at-risk and last-30-days
            # rules -- anchoring to the data keeps the answer stable for anyone
            # who clones the repo. julianday() is the right function here (unlike
            # in the cohort rule, which needs whole *months*) because recency is
            # naturally counted in days, and CAST(... AS INTEGER) truncates the
            # half-day artifact of differencing two date-only julian values.
            # COUNT(DISTINCT o.id) is required rather than COUNT(*): the join to
            # order_items multiplies each order by its line count, so COUNT(*)
            # would measure items and silently inflate Frequency.
            #
            # ``scored`` then buckets each measure into fifths with NTILE(5).
            # The three window functions deliberately do not sort the same way:
            # Frequency and Monetary are ordered ASC so that more is better and
            # 5 is the best score, while Recency is ordered DESC because for
            # recency a *smaller* number is better -- the largest gap since the
            # last order lands in bucket 1. Getting that inversion wrong is the
            # classic RFM bug, and it is invisible in the output because the
            # scores still look plausible. customer_id is a tiebreaker in every
            # window so customers with identical measures fall on the same side
            # of a bucket boundary on every run. The scores cannot be
            # concatenated in the same SELECT that computes them (a window
            # function's result is not addressable by alias in its own select
            # list), which is why ``scored`` is a separate CTE and ``rfm_cell``
            # is built in the outer query.
            #
            # The INNER JOIN means only customers who have ordered are scored: an
            # RFM cell is undefined for someone with no recency and no frequency,
            # and treating them as the worst-scoring segment would mix
            # acquisition into a retention metric. The matcher owns the RFM
            # acronym and the "recency, frequency, monetary" phrasing, which no
            # other rule keys on.
            (
                re.compile(
                    r"\brfm\b|"
                    r"recency\s*,?\s*(and\s+)?frequency|"
                    r"frequency\s*,?\s*(and\s+)?monetary",
                    re.I,
                ),
                """
                WITH customer_rfm AS (
                    SELECT o.customer_id AS customer_id,
                           CAST(
                               julianday((SELECT MAX(order_date) FROM orders))
                               - julianday(MAX(o.order_date)) AS INTEGER
                           ) AS recency_days,
                           COUNT(DISTINCT o.id) AS frequency,
                           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS monetary
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY o.customer_id
                ),
                scored AS (
                    SELECT r.customer_id AS customer_id,
                           c.name AS name,
                           r.recency_days AS recency_days,
                           r.frequency AS frequency,
                           r.monetary AS monetary,
                           NTILE(5) OVER (
                               ORDER BY r.recency_days DESC, r.customer_id
                           ) AS r_score,
                           NTILE(5) OVER (
                               ORDER BY r.frequency ASC, r.customer_id
                           ) AS f_score,
                           NTILE(5) OVER (
                               ORDER BY r.monetary ASC, r.customer_id
                           ) AS m_score
                    FROM customer_rfm r
                    JOIN customers c ON c.id = r.customer_id
                )
                SELECT name,
                       recency_days,
                       frequency,
                       monetary,
                       r_score,
                       f_score,
                       m_score,
                       r_score || f_score || m_score AS rfm_cell
                FROM scored
                ORDER BY monetary DESC, customer_id
                """,
            ),
            # Average revenue per customer (ARPU): mean lifetime spend per paying
            # customer. A first CTE rolls the money up to one row per customer
            # (their total spend); the outer query then AVGs those per-customer
            # totals. The per-customer rollup is essential and is the whole point
            # of this rule: ARPU divides revenue by *customers*, whereas average
            # order value (AOV) divides the same revenue by *orders* -- so on data
            # where customers place several orders each, ARPU is many times larger
            # than AOV, and averaging the raw order-item rows would give neither.
            # The denominator is customers who have actually ordered (the INNER
            # JOIN drops never-buyers), which is the standard "per paying customer"
            # definition and is reproducible from the data alone. The matcher owns
            # the "per customer" revenue/spend phrasings and the ARPU acronym,
            # which no other rule keys on; it is registered ahead of the broad
            # "total revenue" and top-spenders rules so a per-customer question is
            # not shadowed by them.
            (
                re.compile(
                    r"\barpu\b|"
                    r"(average|avg|mean).*(revenue|spend|sales).*per\s+customer|"
                    r"(revenue|spend|sales)\s+per\s+customer|"
                    r"per[- ]customer\s+(revenue|spend|sales)",
                    re.I,
                ),
                """
                WITH customer_revenue AS (
                    SELECT o.customer_id AS customer_id,
                           SUM(oi.quantity * oi.unit_price) AS revenue
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY o.customer_id
                )
                SELECT COUNT(*) AS paying_customers,
                       ROUND(SUM(revenue), 2) AS total_revenue,
                       ROUND(AVG(revenue), 2) AS avg_revenue_per_customer
                FROM customer_revenue
                """,
            ),
            (
                re.compile(r"customers?\b.*\bspent|top\s*(5|five)\s*customers", re.I),
                """
                SELECT c.name,
                       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spent
                FROM customers c
                JOIN orders o ON o.customer_id = c.id
                JOIN order_items oi ON oi.order_id = o.id
                GROUP BY c.id
                ORDER BY total_spent DESC
                LIMIT 5
                """,
            ),
            # Revenue broken out by two dimensions at once: customer region and
            # product category. This is the only rule that groups by two columns,
            # and it needs all four tables -- customers (region) -> orders ->
            # order_items (the money) -> products (category) -- so the region and
            # category labels meet on the same order-item rows. The matcher
            # requires BOTH "region" and "category" words and is registered ahead
            # of the single-dimension "revenue by region" and "revenue by
            # category" rules so a combined question is not shadowed by whichever
            # one-dimension rule would otherwise match first.
            (
                re.compile(
                    r"(revenue|sales).*(region.*categor|categor.*region)|"
                    r"(region.*categor|categor.*region).*(revenue|sales)",
                    re.I,
                ),
                """
                SELECT c.region,
                       p.category,
                       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                FROM customers c
                JOIN orders o ON o.customer_id = c.id
                JOIN order_items oi ON oi.order_id = o.id
                JOIN products p ON p.id = oi.product_id
                GROUP BY c.region, p.category
                ORDER BY c.region, revenue DESC
                """,
            ),
            # Categories whose revenue is above the average category revenue.
            # A first CTE totals each category's revenue; the outer query then
            # keeps only the categories whose revenue exceeds the mean, obtained
            # with a scalar subquery SELECT AVG(revenue) FROM category_revenue
            # that re-reads the same CTE. This "above-average" filter is a
            # distinct idiom from the window rules here: SUM(revenue) OVER ()
            # (category share) annotates every row with the grand total but keeps
            # all rows, whereas this scalar subquery is used in the WHERE clause
            # to *drop* rows below the mean. The comparison is against the rounded
            # per-category revenue, so the threshold matches the revenue values
            # shown. The matcher requires an above/over-average phrase next to a
            # category word and is registered ahead of both the category-share
            # and the plain "revenue by category" rules so those bare questions
            # are not shadowed.
            (
                re.compile(
                    r"categor(y|ies).*(above|over|greater than|more than|beat)"
                    r"[-\s]*(the\s+)?average|"
                    r"(above|over)[-\s]*average.*categor",
                    re.I,
                ),
                """
                WITH category_revenue AS (
                    SELECT p.category AS category,
                           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                    FROM products p
                    JOIN order_items oi ON oi.product_id = p.id
                    GROUP BY p.category
                )
                SELECT category, revenue
                FROM category_revenue
                WHERE revenue > (SELECT AVG(revenue) FROM category_revenue)
                ORDER BY revenue DESC
                """,
            ),
            # Revenue by category with a grand-total row appended -- the shape a
            # report is actually delivered in, where the reader wants the parts
            # and the whole in one table. Postgres and friends would write this
            # as GROUP BY ROLLUP(category), but SQLite has neither ROLLUP nor
            # GROUPING SETS, so the total is produced as a second SELECT over
            # the same CTE and stapled on with UNION ALL. This is the catalog's
            # only *compound* SELECT: every other rule returns rows from a single
            # query, while this one unions two result sets that are at different
            # levels of aggregation.
            #
            # Deriving the total from the CTE rather than re-scanning
            # `order_items` is what guarantees the report foots -- the total is
            # by construction the sum of the rows printed above it, not a second
            # independent measurement that could disagree at the cent. The CTE
            # rounds once, so the outer SUM adds already-rounded figures; the
            # outer ROUND then only mops up binary-float representation error
            # from that addition. The alternative (round only at the end) is
            # marginally more accurate but can print a total a cent away from
            # the sum of its own visible parts, which reads as an arithmetic bug
            # to anyone checking the column by hand.
            #
            # ORDER BY (category = 'Total') exploits SQLite evaluating a boolean
            # as 0 or 1: every real category sorts 0 and the total sorts 1, so
            # the total lands last whatever the revenue ordering does, without
            # carrying a sort-key column through to the output. Revenue DESC
            # then orders the categories themselves. The union is wrapped in a
            # second CTE because SQLite only accepts a bare column name or an
            # ordinal in a compound SELECT's own ORDER BY -- an expression there
            # fails with "1st ORDER BY term does not match any column in the
            # result set". Sorting the union's output as an ordinary table
            # sidesteps that restriction and keeps the sort key out of the
            # result.
            #
            # The matcher requires explicit total/rollup/subtotal vocabulary and
            # is registered ahead of both category rules below, so a plain
            # "revenue by category" still routes to the simple rule and only a
            # question that asks for the total row gets the compound query.
            (
                re.compile(
                    r"(grand\s+)?total\s+(row|line)\b|"
                    r"\brollup\b|\bsub[-\s]?totals?\b|"
                    r"(with|including|include|add|append|plus)\s+"
                    r"(a\s+)?(grand\s+)?total\b",
                    re.I,
                ),
                """
                WITH category_revenue AS (
                    SELECT p.category AS category,
                           SUM(oi.quantity) AS units_sold,
                           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                    FROM products p
                    JOIN order_items oi ON oi.product_id = p.id
                    GROUP BY p.category
                ),
                report AS (
                    SELECT category, units_sold, revenue
                    FROM category_revenue
                    UNION ALL
                    SELECT 'Total', SUM(units_sold), ROUND(SUM(revenue), 2)
                    FROM category_revenue
                )
                SELECT category, units_sold, revenue
                FROM report
                ORDER BY (category = 'Total'), revenue DESC
                """,
            ),
            # Category purchase penetration: how many distinct customers bought
            # from each category, and what share of the buyer base that is. This
            # is a *reach* measure, not a value measure -- the complement of the
            # category-share rule directly below, which splits revenue. The two
            # can disagree in the way that matters commercially: a category can
            # take a small slice of revenue while nearly every customer touches
            # it (a cheap, universal add-on), or carry a large slice off a narrow
            # set of buyers (a big-ticket item). Revenue share alone cannot tell
            # those apart, which is why penetration is reported in customers.
            #
            # It is registered ahead of the category-share rule because "what
            # share of customers bought from each category" contains both a
            # share word and a category word, so that rule would otherwise
            # answer it -- with revenue percentages, which look like a valid
            # answer to a question that was about people. The pattern demands a
            # customer word *and* a purchase verb alongside the category word
            # (or the unambiguous "penetration" / "cross-sell" vocabulary), so a
            # question about revenue share still falls through to the rule below.
            #
            # Two details carry the correctness of the metric:
            #
            # * ``category_buyers`` is DISTINCT on (category, customer_id), so a
            #   customer who bought a category twenty times counts once. Without
            #   it COUNT(*) would count order *lines* and the "percentage" could
            #   exceed 100.
            # * The denominator is the number of customers who placed any order,
            #   not COUNT(*) FROM customers. Dividing by the full customer table
            #   would fold never-buying customers into the base and understate
            #   every category's reach -- the question is which of our *buyers*
            #   reach a category, and non-buyers reach none of them by
            #   definition. ``total_buyers`` is returned rather than left
            #   implicit so the denominator is visible in the output and the
            #   percentages can be re-derived from the columns shown.
            #
            # On this synthetic dataset the four categories all land near 96%:
            # orders draw products uniformly, so with 900 orders across 120
            # customers almost everyone eventually touches every category. The
            # spread is real but small; the ``buyers`` column is what carries the
            # ranking here, and a real catalog with niche lines would separate
            # far more.
            (
                re.compile(
                    r"\bpenetration\b|"
                    r"cross[-\s]?sell(?:ing)?\b.*categor(?:y|ies)|"
                    r"categor(?:y|ies).*\bcross[-\s]?sell(?:ing)?\b|"
                    r"\b(?:customers?|buyers?|shoppers?)\b.*"
                    r"\b(?:bought|buy|purchased?|ordered|order)\b.*"
                    r"\bcategor(?:y|ies)\b",
                    re.I,
                ),
                """
                WITH category_buyers AS (
                    SELECT DISTINCT p.category AS category,
                           o.customer_id AS customer_id
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    JOIN products p ON p.id = oi.product_id
                ),
                buyer_base AS (
                    SELECT COUNT(DISTINCT customer_id) AS buyers
                    FROM orders
                )
                SELECT b.category,
                       COUNT(*) AS buyers,
                       (SELECT buyers FROM buyer_base) AS total_buyers,
                       ROUND(
                           100.0 * COUNT(*) / (SELECT buyers FROM buyer_base), 1
                       ) AS penetration_pct
                FROM category_buyers b
                GROUP BY b.category
                ORDER BY penetration_pct DESC, b.category
                """,
            ),
            # Each category's revenue as a share (percentage) of total revenue.
            # A CTE first computes per-category revenue; the outer query divides
            # each category by the grand total obtained with SUM(revenue) OVER ()
            # -- a window function with an empty OVER () that sums across every
            # row, i.e. the whole result. The 100.0 literal forces float division
            # so the percentages are not integer-truncated. Requiring a
            # share/percentage word in the matcher, and registering this rule
            # ahead of the plain "revenue by category" rule below, keeps a bare
            # "revenue by category" question routed to that simpler rule.
            (
                re.compile(
                    r"(percent(age)?|share|proportion).*categor|"
                    r"categor.*(percent(age)?|share|proportion)",
                    re.I,
                ),
                """
                WITH category_revenue AS (
                    SELECT p.category AS category,
                           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                    FROM products p
                    JOIN order_items oi ON oi.product_id = p.id
                    GROUP BY p.category
                )
                SELECT category,
                       revenue,
                       ROUND(100.0 * revenue / SUM(revenue) OVER (), 2)
                           AS pct_of_total
                FROM category_revenue
                ORDER BY revenue DESC
                """,
            ),
            (
                re.compile(r"(revenue|sales).*(by|per).*(category)", re.I),
                """
                SELECT p.category,
                       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                FROM products p
                JOIN order_items oi ON oi.product_id = p.id
                GROUP BY p.category
                ORDER BY revenue DESC
                """,
            ),
            # Average order value bucketed by month -- the AOV time-series. Like
            # the other order-value rules, order value is a per-order quantity,
            # so each order's total is computed one level down (in the
            # ``order_totals`` CTE, grouping by order id) before averaging;
            # averaging the raw order_items rows would weight the mean by
            # line-item count rather than by order. The CTE carries each order's
            # month up so the outer query can AVG per month. This differs from
            # the two AOV rules below it: the region rule groups the same
            # per-order totals by customer region, and the plain rule averages
            # them over all orders at once, whereas this one groups them by
            # calendar month to show the trend across the year. The matcher
            # requires a month/over-time phrase and is registered ahead of the
            # plain "average order value" rule so a monthly question is not
            # shadowed by it, while a bare "average order value" still routes to
            # the plain rule.
            (
                re.compile(
                    r"(average|avg).*order value.*"
                    r"(by\s+month|per\s+month|each\s+month|monthly|over\s+time)|"
                    r"monthly.*(average|avg).*order value",
                    re.I,
                ),
                """
                WITH order_totals AS (
                    SELECT o.id AS order_id,
                           strftime('%Y-%m', o.order_date) AS month,
                           SUM(oi.quantity * oi.unit_price) AS order_total
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    WHERE o.order_date >= '2024-01-01' AND o.order_date < '2025-01-01'
                    GROUP BY o.id
                )
                SELECT month,
                       ROUND(AVG(order_total), 2) AS avg_order_value
                FROM order_totals
                GROUP BY month
                ORDER BY month
                """,
            ),
            # Average order value broken out by customer region. Order value is a
            # per-order quantity, so it must be computed one level down (each
            # order's total in the subquery) before averaging -- averaging the
            # raw order_items rows would weight the mean by line-item count, not
            # by order. The subquery carries customer_id up so the outer query can
            # join to customers for the region label and then AVG per region.
            # Requiring the word "region" and registering this rule ahead of the
            # plain "average order value" rule below keeps a bare "average order
            # value" question routed to that simpler overall rule.
            (
                re.compile(r"(average|avg).*order value.*region", re.I),
                """
                SELECT c.region,
                       ROUND(AVG(order_total), 2) AS avg_order_value
                FROM (
                    SELECT o.id AS order_id,
                           o.customer_id AS customer_id,
                           SUM(oi.quantity * oi.unit_price) AS order_total
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY o.id
                ) t
                JOIN customers c ON c.id = t.customer_id
                GROUP BY c.region
                ORDER BY avg_order_value DESC
                """,
            ),
            (
                re.compile(r"(average|avg).*order value", re.I),
                """
                SELECT ROUND(AVG(order_total), 2) AS average_order_value
                FROM (
                    SELECT o.id, SUM(oi.quantity * oi.unit_price) AS order_total
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY o.id
                )
                """,
            ),
            # Median order value: the *middle* order total, which SQLite has no
            # built-in function for. A first CTE rolls each order up to its total
            # (the same per-order rollup the average-order-value rules use); the
            # subquery then walks the totals in ascending order and keeps only the
            # middle row(s) with LIMIT/OFFSET:
            #   OFFSET (COUNT(*) - 1) / 2  skips the bottom half, and
            #   LIMIT  2 - COUNT(*) % 2    takes 1 row when the count is odd and
            #                              2 rows when it is even.
            # AVG over that 1-or-2-row window is the median by definition: the
            # single middle value, or the mean of the two middle values. This is
            # deliberately paired with the average rule rather than replacing it —
            # order totals are right-skewed (a few large orders pull the mean up),
            # so the median is the more representative "typical order" and the gap
            # between the two is itself informative. The matcher requires the word
            # "median", which no other rule uses, so it neither shadows nor is
            # shadowed by the average-order-value rules.
            (
                re.compile(
                    r"median\s+(order\s+(value|total)|basket)|"
                    r"order\s+(value|total).*median",
                    re.I,
                ),
                """
                WITH order_totals AS (
                    SELECT o.id AS order_id,
                           SUM(oi.quantity * oi.unit_price) AS order_total
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY o.id
                )
                SELECT ROUND(AVG(order_total), 2) AS median_order_value
                FROM (
                    SELECT order_total
                    FROM order_totals
                    ORDER BY order_total
                    LIMIT 2 - (SELECT COUNT(*) FROM order_totals) % 2
                    OFFSET (SELECT (COUNT(*) - 1) / 2 FROM order_totals)
                )
                """,
            ),
            # Average basket size: the mean number of *units* per order, where an
            # order's unit count is SUM(quantity) across its line items. Like the
            # average-order-value rule above, the per-order rollup happens in the
            # subquery and the AVG is taken over those order totals -- averaging
            # the raw order_items rows instead would weight the mean by how many
            # line items an order has, not by order. This differs from average
            # order value only in the numerator: value sums quantity * unit_price
            # (money), basket size sums quantity (units). The matcher requires an
            # items/units-per-order or basket-size phrase, so it does not shadow
            # the "order value" rule (which owns the "order value" wording) or the
            # broad order/product count rules.
            (
                re.compile(
                    r"(average|avg|mean)\s+(number\s+of\s+)?(items?|units?)\s+per\s+order|"
                    r"(items?|units?)\s+per\s+order|"
                    r"average\s+basket\s+size|basket\s+size",
                    re.I,
                ),
                """
                SELECT ROUND(AVG(order_units), 2) AS avg_units_per_order
                FROM (
                    SELECT o.id AS order_id,
                           SUM(oi.quantity) AS order_units
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY o.id
                )
                """,
            ),
            (
                # Guarded by _UNSCOPED_ONLY: this counts every row in
                # `customers`, so "how many customers churned last month?" is
                # not a question it can answer.
                re.compile(_UNSCOPED_ONLY + r".*how many (?:customers|users)", re.I),
                "SELECT COUNT(*) AS customer_count FROM customers",
            ),
            # The best-selling product (by units) *within each category*: another
            # "greatest-N-per-group" ranking, the product-and-category analogue of
            # the top-customer-per-region rule above. A first CTE rolls units up to
            # one row per product, carrying its category; the second ranks products
            # *inside* each category with
            # ROW_NUMBER() OVER (PARTITION BY category ORDER BY units_sold DESC);
            # the outer query keeps only rank 1 per category. PARTITION BY restarts
            # the numbering for every category, which is what turns one global
            # ranking into a per-category one. product_id breaks ties in the
            # ORDER BY so a category whose top two products are level resolves the
            # same way on every run. This rule requires BOTH a best/top product
            # phrase AND a category word, and is registered ahead of the broad
            # "best selling product" rule below so a per-category question is not
            # shadowed by the single-product global ranking.
            (
                re.compile(
                    r"(best|top)[-\s]*(selling\s+)?products?.*"
                    r"(in|per|by|within|for)\s+(each\s+)?categor(y|ies)|"
                    r"categor(y|ies).*(best|top)[-\s]*(selling\s+)?products?",
                    re.I,
                ),
                """
                WITH product_units AS (
                    SELECT p.category AS category,
                           p.id AS product_id,
                           p.name AS name,
                           SUM(oi.quantity) AS units_sold
                    FROM products p
                    JOIN order_items oi ON oi.product_id = p.id
                    GROUP BY p.id
                ),
                ranked AS (
                    SELECT category,
                           name,
                           units_sold,
                           ROW_NUMBER() OVER (
                               PARTITION BY category
                               ORDER BY units_sold DESC, product_id
                           ) AS rn
                    FROM product_units
                )
                SELECT category, name, units_sold
                FROM ranked
                WHERE rn = 1
                ORDER BY category
                """,
            ),
            (
                re.compile(r"(best|top).*selling product", re.I),
                """
                SELECT p.name,
                       SUM(oi.quantity) AS units_sold
                FROM products p
                JOIN order_items oi ON oi.product_id = p.id
                GROUP BY p.id
                ORDER BY units_sold DESC
                LIMIT 1
                """,
            ),
            # "Top products by revenue" ranks products by sales value
            # (quantity * unit_price), which is distinct from the units-sold
            # ranking above: a cheap high-volume item can top units-sold while a
            # pricier item tops revenue. Requiring the word "revenue" keeps this
            # rule from shadowing the best-selling (units) rule.
            (
                re.compile(
                    r"top.*products?.*revenue|products?.*by revenue|"
                    r"highest[- ]revenue products?",
                    re.I,
                ),
                """
                SELECT p.name,
                       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                FROM products p
                JOIN order_items oi ON oi.product_id = p.id
                GROUP BY p.id
                ORDER BY revenue DESC
                LIMIT 5
                """,
            ),
            # Products frequently bought together (market-basket / affinity
            # analysis): the pairs of products that co-occur in the most orders.
            # This is the only rule that *self-joins* a table -- order_items to
            # itself on the same order_id -- so each matched row is two line
            # items from one order. The join condition oi1.product_id <
            # oi2.product_id does two jobs at once: it drops the trivial
            # self-pairing of a line item with itself, and it emits each unordered
            # pair {A, B} only once (as A,B, never also B,A), so pairs are not
            # double-counted. COUNT(DISTINCT oi1.order_id) counts *orders*
            # containing both products rather than raw joined rows, which keeps
            # the tally correct even when an order lists the same product on more
            # than one line item. product_a/product_b are added to the ORDER BY
            # after the co-occurrence count so ties resolve the same way on every
            # run. The matcher owns the "bought/purchased together", "market
            # basket", and "product pairs/affinity" phrasings, none of which any
            # other rule uses, so it neither shadows nor is shadowed by the
            # product- or basket-size rules.
            (
                re.compile(
                    r"bought\s+together|purchased\s+together|"
                    r"market[-\s]basket|product\s+(pairs|affinity)|"
                    r"frequently\s+bought",
                    re.I,
                ),
                """
                SELECT p1.name AS product_a,
                       p2.name AS product_b,
                       COUNT(DISTINCT oi1.order_id) AS orders_together
                FROM order_items oi1
                JOIN order_items oi2
                     ON oi2.order_id = oi1.order_id
                    AND oi1.product_id < oi2.product_id
                JOIN products p1 ON p1.id = oi1.product_id
                JOIN products p2 ON p2.id = oi2.product_id
                GROUP BY p1.name, p2.name
                ORDER BY orders_together DESC, product_a, product_b
                LIMIT 5
                """,
            ),
            # Average customer lifespan: the mean number of days between a
            # customer's first and last order. A first CTE collapses the orders
            # table to one row per customer whose active_days is
            # julianday(MAX(order_date)) - julianday(MIN(order_date)) -- julianday
            # converts each date to a fractional day number so the two can be
            # subtracted, which is how date differences are taken in SQLite (there
            # is no DATEDIFF). This is the only rule that does date arithmetic
            # rather than date *formatting* (the strftime rules bucket by month,
            # quarter, or weekday; this one measures an elapsed span). The outer
            # query then AVGs those per-customer spans -- a nested aggregate: MIN
            # and MAX run per customer inside the CTE, AVG runs across customers
            # outside it, which is why the CTE is needed rather than a single flat
            # query. A customer with exactly one order has first == last and so
            # contributes a span of 0, correctly pulling the average toward the
            # behaviour of one-time buyers. The matcher owns the lifespan/tenure
            # phrasings, which no other rule uses, so it neither shadows nor is
            # shadowed by the customer-count or repeat-customer rules.
            (
                re.compile(
                    r"customer\s+(lifespan|tenure|lifetime)|"
                    r"(lifespan|tenure)\s+of\s+(a\s+|the\s+)?customers?|"
                    r"how\s+long\s+.*customers?\s+(stay|remain|are)\s+active|"
                    r"average\s+(active\s+)?(customer\s+)?lifespan|"
                    r"days\s+between\s+(a\s+)?customers?['’]?s?\s+"
                    r"first\s+and\s+last\s+order",
                    re.I,
                ),
                """
                WITH customer_span AS (
                    SELECT customer_id,
                           julianday(MAX(order_date))
                               - julianday(MIN(order_date)) AS active_days
                    FROM orders
                    GROUP BY customer_id
                )
                SELECT ROUND(AVG(active_days), 1) AS avg_customer_lifespan_days
                FROM customer_span
                """,
            ),
            # At-risk (lapsed) customers: those who have bought before but whose
            # most recent order is now old -- the "Recency" lens of RFM analysis.
            # A first CTE collapses the orders table to one row per customer
            # carrying MAX(order_date) (their latest order); the outer query then
            # keeps only customers whose latest order predates a cutoff 90 days
            # before the newest order in the data. Two deliberate design choices,
            # noted so they can be defended:
            #   * The cutoff is anchored to the dataset's own MAX(order_date), not
            #     the wall-clock today(), so the answer is reproducible for anyone
            #     who clones the repo -- the same reasoning the "orders in the last
            #     30 days" rule uses. This rule reuses that date(..., '-N day')
            #     idiom but applies it as a per-customer recency filter rather than
            #     a single global count.
            #   * The JOIN to orders is inner, so only customers with at least one
            #     order are considered. A customer who never ordered is an
            #     acquisition question, not a churn one; "at-risk" implies a prior
            #     relationship that has since gone quiet.
            # The matcher owns the churn/lapsed/inactive/at-risk and "customers
            # haven't ordered" phrasings, none of which other rules use, so it
            # neither shadows nor is shadowed by the repeat-customer or
            # recent-orders rules. It is placed ahead of the broad aggregate rules
            # for the same first-rule-wins reason as the other specific rules.
            (
                re.compile(
                    r"at[-\s]?risk\s+customers?|"
                    r"churn(?:ed|ing)?\s+customers?|"
                    r"lapsed\s+customers?|"
                    r"inactive\s+customers?|"
                    r"customers?\s+(?:who\s+)?"
                    r"(?:haven'?t|hasn'?t|have\s+not|has\s+not|not)\s+"
                    r"(?:placed\s+an?\s+order|ordered)|"
                    r"customers?\b.*\bno\s+orders?\s+in\s+the\s+last",
                    re.I,
                ),
                """
                WITH customer_last_order AS (
                    SELECT c.id AS customer_id,
                           c.name AS name,
                           MAX(o.order_date) AS last_order_date
                    FROM customers c
                    JOIN orders o ON o.customer_id = c.id
                    GROUP BY c.id
                )
                SELECT customer_id, name, last_order_date
                FROM customer_last_order
                WHERE last_order_date
                      < date((SELECT MAX(order_date) FROM orders), '-90 day')
                ORDER BY last_order_date, customer_id
                """,
            ),
            # Revenue split between orders from *new* customers (their very first
            # order) and *returning* customers (every order after the first) --
            # the standard new-vs-returning revenue attribution. A first CTE rolls
            # each order up to its total, carrying the customer and order date; the
            # second labels every order by whether it is that customer's first with
            # ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date, order_id)
            # == 1 -> 'new', else 'returning'. This reuses the same per-customer
            # ROW_NUMBER idiom as the top-spender-per-region rule but to a different
            # end: that rule *keeps* rank 1 to pick a single winner per group, while
            # this one *labels* rank 1 vs the rest and keeps every order, so the two
            # revenue buckets together sum to total revenue. order_id is a
            # deterministic tiebreaker so two orders a customer placed on the same
            # day resolve the same way on every run (only one can be the 'new' one).
            # The matcher owns the new-vs-returning / first-time-vs-repeat contrast
            # phrasings and is registered ahead of the plain "(repeat|returning)
            # customers" count rule below so a question naming "returning customers"
            # in a revenue-split context is not shadowed by that simpler counter.
            (
                re.compile(
                    r"new\s+(?:vs\.?|versus|and|or|&)\s+returning|"
                    r"returning\s+(?:vs\.?|versus|and|or|&)\s+new|"
                    r"first[-\s]?time\s+(?:vs\.?|versus|and|or|&)\s+(?:returning|repeat)|"
                    r"(?:revenue|sales|spend)\s+from\s+new\s+"
                    r"(?:and|vs\.?|versus|&)\s+returning",
                    re.I,
                ),
                """
                WITH order_totals AS (
                    SELECT o.id AS order_id,
                           o.customer_id AS customer_id,
                           o.order_date AS order_date,
                           SUM(oi.quantity * oi.unit_price) AS order_total
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY o.id
                ),
                classified AS (
                    SELECT order_total,
                           CASE
                               WHEN ROW_NUMBER() OVER (
                                        PARTITION BY customer_id
                                        ORDER BY order_date, order_id
                                    ) = 1
                               THEN 'new'
                               ELSE 'returning'
                           END AS customer_type
                    FROM order_totals
                )
                SELECT customer_type,
                       COUNT(*) AS orders,
                       ROUND(SUM(order_total), 2) AS revenue
                FROM classified
                GROUP BY customer_type
                ORDER BY customer_type
                """,
            ),
            # Repeat purchase rate per product: of the customers who ever bought
            # a product, what share came back and bought it again. This is the
            # product-level counterpart of the whole-business repeat-customer
            # count registered directly below, and it answers a different
            # question -- repeat customers measures whether the *business*
            # retains buyers, this measures whether an *individual product*
            # earns a second purchase, which is what separates a consumable
            # from a one-off (a coffee mug from a standing desk mat).
            #
            # The load-bearing modelling decision is COUNT(DISTINCT o.id): a
            # "repeat" buyer is one who bought the product on two separate
            # *orders*, not one who put two units in a single basket. Counting
            # order_items rows instead would score buying three mugs at once as
            # repeat behavior, which inverts the metric's meaning -- one basket
            # is one purchase decision, and the whole point of the measure is
            # whether the customer made a second one later.
            #
            # The denominator is the product's own buyer base, not the customer
            # table, so the rate is comparable across products with very
            # different reach; ``buyers`` and ``repeat_buyers`` are both
            # returned so the percentage can be re-derived from the printed
            # rows and a rate computed off a thin buyer base is visible as such
            # rather than hidden behind a confident-looking percentage.
            #
            # Registered ahead of the broad "(repeat|returning) customers" rule
            # on purpose. Without this rule, "which products have the most
            # repeat customers?" falls through to that one and is answered with
            # a single business-wide count -- a plausible number that silently
            # drops the product dimension the question was about. The pattern
            # here requires a product/item word, so the plain "how many repeat
            # customers are there?" phrasing still reaches the rule below.
            (
                re.compile(
                    r"repeat[-\s]?(?:purchase|buy|buying|order)\s*rate|"
                    r"re-?purchase\s*rate|rebuy\s*rate|"
                    r"(?:products?|items?)\b.*"
                    r"\brepeat\s+(?:buyers?|purchasers?|purchases?|customers?)|"
                    r"repeat\s+(?:buyers?|purchasers?|purchases?)\b.*"
                    r"\b(?:products?|items?)|"
                    r"(?:which|what)\s+products?\b.*"
                    r"\b(?:buy|bought|purchase|purchased|order|ordered)\b.*\bagain\b|"
                    r"products?\b.*\bbought\s+more\s+than\s+once",
                    re.I,
                ),
                """
                WITH product_customer AS (
                    SELECT oi.product_id AS product_id,
                           o.customer_id AS customer_id,
                           COUNT(DISTINCT o.id) AS orders_with_product
                    FROM order_items oi
                    JOIN orders o ON o.id = oi.order_id
                    GROUP BY oi.product_id, o.customer_id
                )
                SELECT p.name AS product,
                       COUNT(*) AS buyers,
                       SUM(CASE WHEN pc.orders_with_product > 1 THEN 1 ELSE 0 END)
                           AS repeat_buyers,
                       ROUND(
                           100.0
                           * SUM(CASE WHEN pc.orders_with_product > 1 THEN 1 ELSE 0 END)
                           / COUNT(*),
                           1
                       ) AS repeat_rate_pct
                FROM product_customer pc
                JOIN products p ON p.id = pc.product_id
                GROUP BY p.id
                ORDER BY repeat_rate_pct DESC, product
                """,
            ),
            (
                re.compile(r"(repeat|returning) customers", re.I),
                """
                SELECT COUNT(*) AS repeat_customers
                FROM (
                    SELECT customer_id
                    FROM orders
                    GROUP BY customer_id
                    HAVING COUNT(*) > 1
                )
                """,
            ),
            (
                re.compile(r"orders.*last (30|thirty) days", re.I),
                """
                SELECT COUNT(*) AS recent_orders
                FROM orders
                WHERE order_date >= date((SELECT MAX(order_date) FROM orders), '-30 day')
                """,
            ),
            (
                re.compile(r"(revenue|sales).*by region", re.I),
                """
                SELECT c.region,
                       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                FROM customers c
                JOIN orders o ON o.customer_id = c.id
                JOIN order_items oi ON oi.order_id = o.id
                GROUP BY c.region
                ORDER BY revenue DESC
                """,
            ),
            # Revenue by price tier -- fixed-threshold *banding* of a continuous
            # variable: each product falls into Budget / Mid-range / Premium by
            # comparing its price against dollar cutoffs. This is the complement
            # of the NTILE(4) spend-quartile rule: NTILE builds equal-*count*
            # buckets whose boundaries move with the data, while a CASE band has
            # fixed, meaningful boundaries ("under $20") whose *populations*
            # move with the data. An analyst reaches for bands when the
            # thresholds themselves carry business meaning (a pricing strategy,
            # a free-shipping cutoff) and for quantiles when they only need
            # equal-sized cohorts.
            #
            # The tier is derived from p.price (the catalog list price), not
            # oi.unit_price (the transacted price): a product's tier is a
            # property of the product, so it must not straddle tiers if a
            # historical sale happened at a different price. Revenue still sums
            # oi.unit_price, because money earned is a property of the
            # transaction. In the sample DB the two are equal, but keeping the
            # roles distinct is what makes the query correct on data where they
            # are not.
            #
            # SQLite allows GROUP BY on the SELECT alias, so the CASE is not
            # repeated. ORDER BY MIN(p.price) sorts the tiers cheapest-first --
            # by their actual contents rather than alphabetically (which would
            # interleave them: Budget, Mid-range, Premium only sorts correctly
            # by accident of these labels).
            (
                re.compile(
                    r"(revenue|sales)\b.*\bprice\s+(tier|band|bracket)s?|"
                    r"\bprice\s+(tier|band|bracket)s?\b.*\b(revenue|sales)|"
                    r"by\s+price\s+(tier|band|bracket)s?",
                    re.I,
                ),
                """
                SELECT CASE
                           WHEN p.price < 20 THEN 'Budget (under $20)'
                           WHEN p.price < 40 THEN 'Mid-range ($20-$39.99)'
                           ELSE 'Premium ($40 and up)'
                       END AS price_tier,
                       COUNT(DISTINCT p.id) AS products,
                       SUM(oi.quantity) AS units_sold,
                       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                FROM order_items oi
                JOIN products p ON p.id = oi.product_id
                GROUP BY price_tier
                ORDER BY MIN(p.price)
                """,
            ),
            (
                re.compile(r"(monthly )?new customers.*(by month|2024)", re.I),
                """
                SELECT strftime('%Y-%m', signup_date) AS month,
                       COUNT(*) AS new_customers
                FROM customers
                WHERE signup_date >= '2024-01-01' AND signup_date < '2025-01-01'
                GROUP BY month
                ORDER BY month
                """,
            ),
            # Unique (active) customers per month -- the "monthly active buyers"
            # metric: how many *distinct* customers placed at least one order in
            # each month, as opposed to the raw order count. COUNT(DISTINCT
            # customer_id) collapses a customer's multiple orders in a month down
            # to one, so a repeat buyer is counted once per month. It requires
            # a distinctness
            # word (unique/distinct/active) plus a customer/buyer word, so it does
            # not shadow the plain "how many customers" count or the "new
            # customers by month" signup rule above it (which counts signups, not
            # buyers).
            (
                re.compile(
                    r"(unique|distinct|active)\s+(customers?|buyers?|shoppers?)"
                    r".*(month|2024)|"
                    r"monthly.*(unique|distinct|active)\s+(customers?|buyers?)",
                    re.I,
                ),
                """
                SELECT strftime('%Y-%m', o.order_date) AS month,
                       COUNT(DISTINCT o.customer_id) AS unique_customers
                FROM orders o
                WHERE o.order_date >= '2024-01-01' AND o.order_date < '2025-01-01'
                GROUP BY month
                ORDER BY month
                """,
            ),
            # Revenue bucketed by calendar quarter for 2024. SQLite has no
            # quarter function, so the quarter number is derived from the month
            # with integer arithmetic: (month + 2) / 3 maps months 1-3 -> 1,
            # 4-6 -> 2, 7-9 -> 3, 10-12 -> 4. Kept with the other time-series
            # rules and ahead of the broad "total revenue" rule below so that
            # quarterly phrasings are not shadowed by it.
            (
                re.compile(
                    r"(revenue|sales).*(by|per).*quarter|quarterly (revenue|sales)",
                    re.I,
                ),
                """
                SELECT '2024-Q' || (
                           (CAST(strftime('%m', o.order_date) AS INTEGER) + 2) / 3
                       ) AS quarter,
                       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                FROM orders o
                JOIN order_items oi ON oi.order_id = o.id
                WHERE o.order_date >= '2024-01-01' AND o.order_date < '2025-01-01'
                GROUP BY quarter
                ORDER BY quarter
                """,
            ),
            # Customers who ordered in *every* quarter of 2024: the catalog's
            # only relational division (a "for all" query). Every other rule
            # asks "which rows satisfy X" -- an EXISTS-shaped question; this one
            # asks "which customers satisfy X for ALL members of a set" (all
            # four quarters), which no plain WHERE clause can express.
            #
            # The division is done as HAVING COUNT(DISTINCT quarter) = 4 rather
            # than the textbook double-NOT-EXISTS: one aggregate over a grouped
            # join reads as "covered 4 distinct quarters", while nested negated
            # quantifiers hide the same test inside two layers of negation. The
            # DISTINCT is load-bearing -- a customer with five orders all in Q1
            # has COUNT(*) = 5 but COUNT(DISTINCT quarter) = 1, and it is
            # quarters covered, not orders placed, that the question asks about.
            #
            # The 4 is hardcoded because it is a fact of the calendar, not of
            # the data: a year has four quarters whether or not anyone ordered
            # in all of them. Deriving the denominator from the data (COUNT of
            # quarters seen in orders) would quietly weaken the test in exactly
            # the case where it matters -- a dead quarter would shrink the
            # requirement to 3 and admit customers the question excludes.
            #
            # The quarter number reuses the (month + 2) / 3 integer-arithmetic
            # derivation from the revenue-by-quarter rule above, so the two
            # rules cannot disagree about which quarter a date belongs to.
            #
            # ORDER BY is readability only (name, with id as a deterministic
            # tiebreaker since the sample generator can produce duplicate
            # names); the answer is a *set* of customers, so the gold row is
            # marked "ordered": false.
            (
                re.compile(r"(every|each|all\s+(four|4))\s+quarters?\b", re.I),
                """
                SELECT c.id AS customer_id,
                       c.name AS customer,
                       c.region,
                       COUNT(o.id) AS orders_2024
                FROM customers c
                JOIN orders o ON o.customer_id = c.id
                WHERE o.order_date >= '2024-01-01' AND o.order_date < '2025-01-01'
                GROUP BY c.id, c.name, c.region
                HAVING COUNT(DISTINCT
                           (CAST(strftime('%m', o.order_date) AS INTEGER) + 2) / 3
                       ) = 4
                ORDER BY c.name, c.id
                """,
            ),
            # First-half vs second-half 2024 revenue for every product: the
            # "what is trending up, what is trending down" question, which is
            # asked of products far more often than of the business as a whole.
            #
            # This is the catalog's only *conditional aggregation* (a pivot):
            # two date ranges become two side-by-side columns rather than two
            # rows, because SUM(CASE WHEN ... THEN ... ELSE 0 END) applies a
            # different filter per output column within one GROUP BY. The
            # alternative -- grouping by half and reading two rows per product,
            # or self-joining a filtered aggregate to itself -- makes the
            # subtraction that the question actually asks for (h2 - h1) awkward:
            # it cannot be expressed until the two halves sit in the same row.
            # This form scans the join once and lands both halves together.
            #
            # ELSE 0 rather than ELSE NULL is deliberate. A product sold in only
            # one half must report 0.00 for the other, not NULL: NULL would
            # propagate through the subtraction and drop exactly the products
            # whose change is most extreme -- the ones that appeared or vanished
            # -- from a result whose whole purpose is to surface them.
            #
            # pct_change divides by NULLIF(h1_revenue, 0) so a product with no
            # first-half sales yields NULL rather than a meaningless percentage
            # (SQLite returns NULL for x/0 rather than raising, so the guard is
            # about stating the intent, not about avoiding an error). Its
            # absolute change is still reported, which is the honest reading:
            # growth from zero has a magnitude but no percentage.
            #
            # The matcher owns the half/H1/H2 vocabulary, which no other rule
            # uses, and the ordering tiebreak on product name keeps two products
            # with an identical change in a stable order across runs.
            (
                re.compile(
                    r"(first|1st|second|2nd)\s+half\s+of\s+(the\s+)?(year|2024)|"
                    r"(first|1st)\s+half\b.*\b(second|2nd)\s+half\b|"
                    r"\bh1\b.*\bh2\b|"
                    r"half[-\s]?over[-\s]?half",
                    re.I,
                ),
                """
                WITH product_halves AS (
                    SELECT p.name AS product,
                           SUM(CASE WHEN o.order_date < '2024-07-01'
                                    THEN oi.quantity * oi.unit_price
                                    ELSE 0 END) AS h1_revenue,
                           SUM(CASE WHEN o.order_date >= '2024-07-01'
                                    THEN oi.quantity * oi.unit_price
                                    ELSE 0 END) AS h2_revenue
                    FROM products p
                    JOIN order_items oi ON oi.product_id = p.id
                    JOIN orders o ON o.id = oi.order_id
                    WHERE o.order_date >= '2024-01-01'
                      AND o.order_date < '2025-01-01'
                    GROUP BY p.id
                )
                SELECT product,
                       ROUND(h1_revenue, 2) AS h1_revenue,
                       ROUND(h2_revenue, 2) AS h2_revenue,
                       ROUND(h2_revenue - h1_revenue, 2) AS revenue_change,
                       ROUND(
                           100.0 * (h2_revenue - h1_revenue)
                           / NULLIF(h1_revenue, 0), 1
                       ) AS pct_change
                FROM product_halves
                ORDER BY revenue_change DESC, product
                """,
            ),
            # Revenue split by day of the week (Sunday..Saturday). SQLite's
            # strftime('%w', ...) returns the weekday as a digit 0-6 with
            # Sunday == 0; a CASE expression turns that digit into a readable
            # name. Grouping and ordering use the numeric weekday (not the name)
            # so the rows come back in calendar order rather than alphabetically.
            # Kept with the other time-series rules and ahead of the broad
            # "total revenue" rule so day-of-week phrasings are not shadowed.
            (
                re.compile(
                    r"(revenue|sales).*(day of (the )?week|day[- ]of[- ]week|"
                    r"weekday|days? of the week)|by weekday",
                    re.I,
                ),
                """
                SELECT CASE CAST(strftime('%w', o.order_date) AS INTEGER)
                           WHEN 0 THEN 'Sunday'
                           WHEN 1 THEN 'Monday'
                           WHEN 2 THEN 'Tuesday'
                           WHEN 3 THEN 'Wednesday'
                           WHEN 4 THEN 'Thursday'
                           WHEN 5 THEN 'Friday'
                           WHEN 6 THEN 'Saturday'
                       END AS weekday,
                       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                FROM orders o
                JOIN order_items oi ON oi.order_id = o.id
                GROUP BY CAST(strftime('%w', o.order_date) AS INTEGER)
                ORDER BY CAST(strftime('%w', o.order_date) AS INTEGER)
                """,
            ),
            # Average time between a customer's consecutive orders (purchase
            # cadence): a repeat-purchase engagement metric. The single CTE lines
            # every order up next to that same customer's previous order using
            # LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date).
            # This is the only rule with a *partitioned* window: the earlier LAG
            # rule (month-over-month) has one unpartitioned series, whereas here
            # PARTITION BY restarts the LAG for each customer so gaps never span
            # two different customers. The order_date difference is taken in days
            # via julianday(), which converts an ISO date to a Julian day number
            # so the subtraction yields whole days. A customer's first order has
            # no prior order, so its LAG is NULL and gap_days is NULL; the outer
            # WHERE drops those rows, leaving one gap per repeat purchase, and AVG
            # is the mean of all such gaps. The ORDER BY breaks ties on o.id so
            # two orders a customer placed on the same day resolve identically on
            # every run (their gap is 0). The matcher owns the "time/days/gap
            # between orders" and "purchase/reorder cadence|frequency" phrasings,
            # none of which any other rule uses, so it neither shadows nor is
            # shadowed by the customer-lifespan or broad order-count rules.
            (
                re.compile(
                    r"(time|days?|gap|interval)\s+between\s+(orders|purchases)|"
                    r"between\s+(consecutive\s+)?(orders|purchases)|"
                    r"(order|purchase|reorder|repurchase)\s+"
                    r"(cadence|frequency|interval)|"
                    r"how\s+(often|frequently)\s+.*(order|purchase)",
                    re.I,
                ),
                """
                WITH order_gaps AS (
                    SELECT o.customer_id,
                           julianday(o.order_date) - julianday(
                               LAG(o.order_date) OVER (
                                   PARTITION BY o.customer_id
                                   ORDER BY o.order_date, o.id
                               )
                           ) AS gap_days
                    FROM orders o
                )
                SELECT ROUND(AVG(gap_days), 1) AS avg_days_between_orders
                FROM order_gaps
                WHERE gap_days IS NOT NULL
                """,
            ),
            # Distribution of orders per customer: a purchase-frequency histogram
            # and the foundation of frequency-based (RFM) segmentation. This is a
            # *nested aggregation* -- the only rule that groups by the output of a
            # prior GROUP BY. The inner CTE collapses the orders table to one row
            # per customer carrying that customer's order count; the outer query
            # then groups those per-customer counts, so each result row reads
            # "this many customers placed exactly this many orders." Ordering by
            # order_count returns the histogram buckets low-to-high. It is a
            # distribution, not the single-number cadence metric owned by the
            # between-orders rule (which this does not use "frequency" wording to
            # avoid), and it is registered ahead of the broad order-count rule
            # below so "orders per customer" phrasings are not shadowed by it.
            (
                re.compile(
                    r"distribution\s+of\s+(the\s+)?(number\s+of\s+)?orders|"
                    r"orders?\s+per\s+customer|"
                    r"(number|count)\s+of\s+orders\s+per\s+customer|"
                    r"how\s+many\s+orders\s+(do|does)\s+(each|every|a)\s+customers?|"
                    r"order[-\s]?count\s+(distribution|histogram|breakdown)|"
                    r"(distribution|histogram|breakdown)\s+of\s+order\s+counts?",
                    re.I,
                ),
                """
                WITH orders_per_customer AS (
                    SELECT customer_id, COUNT(*) AS order_count
                    FROM orders
                    GROUP BY customer_id
                )
                SELECT order_count,
                       COUNT(*) AS customers
                FROM orders_per_customer
                GROUP BY order_count
                ORDER BY order_count
                """,
            ),
            # Multi-category orders: how many orders mix products from more
            # than one category, and what share of all orders that is. This is
            # a *basket-breadth* measure -- where basket size (units per order)
            # asks how much a customer buys at once, this asks how widely, which
            # is the quantity a cross-sell effort is trying to move.
            #
            # COUNT(DISTINCT p.category) is what makes the per-order count a
            # breadth: two Electronics items in one order are one category, not
            # two, so only genuinely mixed baskets clear the > 1 bar. The CTE
            # keeps "how many categories does each order touch" as its own
            # reusable per-order fact, and the *outer* WHERE applies the
            # more-than-one cut -- the filter belongs to this question, not to
            # the fact. The share's denominator is the orders table itself
            # rather than the CTE, so the reported percentage is honestly "of
            # all orders" even for a hypothetical order with no items (which
            # would be absent from the CTE).
            #
            # The natural phrasing contains "how many orders", which the broad
            # order-count rule below also matches; registering this rule ahead
            # of it is what routes the question here (first-rule-wins), and
            # `explain` reports the broad rule as shadowed for exactly this
            # question.
            (
                re.compile(
                    r"orders?\b.*\b(more than one|multiple|two or more|several)\b"
                    r".*categor|cross[-\s]categor",
                    re.I,
                ),
                """
                WITH categories_per_order AS (
                    SELECT oi.order_id,
                           COUNT(DISTINCT p.category) AS category_count
                    FROM order_items oi
                    JOIN products p ON p.id = oi.product_id
                    GROUP BY oi.order_id
                )
                SELECT COUNT(*) AS multi_category_orders,
                       ROUND(
                           100.0 * COUNT(*) / (SELECT COUNT(*) FROM orders), 1
                       ) AS pct_of_orders
                FROM categories_per_order
                WHERE category_count > 1
                """,
            ),
            # The 10 largest orders by value -- the catalog's only *drill-down*:
            # every other rule aggregates away individual rows (by month, by
            # category, by customer), while this one surfaces specific orders,
            # with date and customer attached so an outlier can be followed up
            # ("who placed it, and when?"). That is the natural next question
            # after any aggregate looks off -- a spike in a monthly total is
            # usually one or two unusually large orders, and this is the query
            # that finds them.
            #
            # GROUP BY o.id is safe here even though order_date and the customer
            # name are selected bare: o.id is the orders table's primary key, so
            # both are single-valued within each group -- there is exactly one
            # date and one customer per order, so nothing is being arbitrarily
            # picked. Ordering is by the rounded alias (what the user sees) with
            # o.id as a deterministic tiebreaker, so two orders with equal totals
            # rank the same way on every run. The matcher owns the
            # largest/biggest/highest-value orders phrasings, which no other rule
            # uses; "top N orders" is claimed only with a following "orders" so
            # it cannot collide with the top-customers or top-products rules.
            (
                re.compile(
                    r"(largest|biggest|highest[-\s]?value)\s+(\d+\s+)?orders?|"
                    r"top\s+\d+\s+orders\b|"
                    r"orders?\b.*\bhighest\s+(value|total)",
                    re.I,
                ),
                """
                SELECT o.id AS order_id,
                       o.order_date,
                       c.name AS customer,
                       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS order_total
                FROM orders o
                JOIN customers c ON c.id = o.customer_id
                JOIN order_items oi ON oi.order_id = o.id
                GROUP BY o.id
                ORDER BY order_total DESC, o.id
                LIMIT 10
                """,
            ),
            # The rules below are intentionally placed last. Matching is
            # first-rule-wins, so these broad aggregate phrasings ("how many
            # orders") do not shadow the more specific rules above (e.g. orders
            # in the last 30 days), which should take precedence.
            #
            # Each also carries the _UNSCOPED_ONLY guard: being both last and
            # broad is what makes them the catch-all a period-scoped question
            # lands on, and every one of them aggregates over a whole table
            # with no date filter. See _UNSCOPED_ONLY for why declining beats
            # answering there.
            (
                re.compile(
                    _UNSCOPED_ONLY
                    + r".*(?:total revenue|overall revenue|how much revenue)",
                    re.I,
                ),
                """
                SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue
                FROM order_items oi
                """,
            ),
            (
                re.compile(
                    _UNSCOPED_ONLY
                    + r".*(?:how many orders|number of orders|total orders"
                    r"|order count)",
                    re.I,
                ),
                "SELECT COUNT(*) AS order_count FROM orders",
            ),
            (
                re.compile(
                    _UNSCOPED_ONLY
                    + r".*(?:how many products|number of products|product count"
                    r"|products in (?:the )?catalog)",
                    re.I,
                ),
                "SELECT COUNT(*) AS product_count FROM products",
            ),
        ]

    def rule_count(self) -> int:
        """Return the number of question patterns registered in the catalog."""
        return len(self._rules)

    def rule_pattern(self, index: int) -> str:
        """Return the regex source of the rule at ``index``.

        Used to name a rule in a test failure message; a pattern is far more
        recognizable than a bare index when a catalog invariant breaks.
        """
        return self._rules[index][0].pattern

    def matching_rule_indexes(self, question: str) -> list[int]:
        """Return the index of every rule whose matcher matches ``question``.

        Resolution is first-rule-wins, so only ``[0]`` of this list decides the
        SQL. The rest is what makes the catalog's ordering auditable: a rule
        that matches but never wins is shadowed by a broader rule registered
        ahead of it, and a rule that never appears first for any question is
        unreachable. ``tests/test_rule_catalog.py`` asserts both properties
        across the gold set.
        """
        return [
            index
            for index, (matcher, _) in enumerate(self._rules)
            if matcher.search(question)
        ]

    def to_sql(self, question: str, schema: str) -> str:  # noqa: ARG002
        # Deliberately reuses ``matching_rule_indexes`` instead of short-circuiting
        # on the first match: routing and the ordering diagnostic then share one
        # implementation and cannot drift apart. Scanning ~40 small regexes is not
        # a meaningful cost next to executing the query.
        matches = self.matching_rule_indexes(question)
        if not matches:
            raise NoRuleMatchError(
                "Offline backend has no rule for this question. "
                "Set OPENAI_API_KEY and use --llm for open-ended questions."
            )
        _, sql = self._rules[matches[0]]
        return " ".join(sql.split())


# --------------------------------------------------------------------------- #
# LLM backend
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = """You are a careful analytics engineer. Given a SQLite schema
and a question, return a single read-only SQL query that answers it.

Rules:
- Output ONLY the SQL, no prose, no markdown fences.
- Use only SELECT (or WITH ... SELECT). Never modify data.
- Use the exact table and column names from the schema.
- Prefer explicit JOINs and clear column aliases.
"""

_REPAIR_PROMPT = """You are a careful analytics engineer. A SQLite query you
wrote for a question failed. Rewrite it so that it runs and still answers the
same question.

Rules:
- Output ONLY the corrected SQL, no prose, no markdown fences.
- Use only SELECT (or WITH ... SELECT). Never modify data.
- Use only the exact table and column names from the schema.
- Fix the reported error. Do not change what the query is trying to measure.
"""


def _prompt_fingerprint(prompt: str) -> str:
    """Return a short, stable digest of a prompt.

    Used to make the prompt part of a cache key without storing the prompt
    itself in every cache entry. Truncated to 12 hex characters: the digest
    only has to distinguish successive revisions of one file, not resist an
    adversary, and a short one keeps ``cache_identity`` readable when it turns
    up in a cache file someone is inspecting by hand.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


class LLMBackend:
    """OpenAI-compatible chat backend. Imports the client lazily."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        from openai import OpenAI  # lazy import; optional dependency

        self._client = OpenAI(api_key=api_key)
        self._model = model

    @property
    def cache_identity(self) -> str:
        """Everything about this backend's configuration that shapes its SQL.

        The model name and a fingerprint of the system prompt, which together
        with the question and schema determine ``to_sql``'s output at
        ``temperature=0``. Satisfies :class:`nl2sql.cache.CacheableBackend`, and
        exists so that ``cache`` never has to reach into this class to discover
        how it is configured.

        The *repair* prompt is deliberately excluded: repairs are not cached, so
        including it would invalidate every stored entry on an edit that cannot
        change any of them.
        """
        return f"{self._model}/{_prompt_fingerprint(_SYSTEM_PROMPT)}"

    def to_sql(self, question: str, schema: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
            ],
        )
        sql = resp.choices[0].message.content or ""
        return _strip_fences(sql).strip()

    def repair(self, question: str, schema: str, sql: str, error: str) -> str:
        """Return a rewritten query, given the SQL that failed and its error.

        The failed SQL and the error text are the whole point: without them the
        model is just being asked the same question again at temperature 0 and
        would return the same query. With them, the common LLM text-to-SQL
        failure modes — a column that does not exist, a table joined on the
        wrong key, a function SQLite does not have — become directly
        correctable, because the engine has already named what is wrong.

        The original question and schema are re-sent rather than relying on a
        conversation history so this call is stateless: one repair is
        independent of any other, which keeps it cheap to reason about and
        makes the backend safe to reuse across questions.

        This method has no authority of its own. Whatever it returns goes back
        through the same validator and read-only connection as the first
        attempt, so a repair cannot widen what the system is willing to run.
        """
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": _REPAIR_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Schema:\n{schema}\n\n"
                        f"Question: {question}\n\n"
                        f"SQL that failed:\n{sql}\n\n"
                        f"Error:\n{error}"
                    ),
                },
            ],
        )
        return _strip_fences(resp.choices[0].message.content or "").strip()


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:sql)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def get_backend(use_llm: bool) -> Backend:
    """Factory: pick the LLM backend when requested, else offline."""
    if use_llm:
        return LLMBackend()
    return OfflineBackend()
