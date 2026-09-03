"""Host and URL routing for hard-negative mining (scope.md 2, 4.2).

scope.md 2 says false positives concentrate on press releases, SEO copy, corporate
blogs, templated product descriptions and non-native English. No off-the-shelf corpus
supplies those at scale, so they are mined out of pre-2022 CommonCrawl-derived text by
URL. This module is the router that decides what a URL is.

Registered domains come from tldextract's bundled public-suffix snapshot with network
lookups disabled. That matters for reproducibility: the suffix list decides whether
`foo.co.uk` groups as `foo.co.uk` or `co.uk`, and grouping decides train/test splits.
Pinning the tldextract version pins the snapshot, so a split cannot silently change
underneath a corpus.
"""

from __future__ import annotations

import re
from functools import cache, lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from ..data.genre import HardNegativeKind

HOSTS_DIR = Path(__file__).resolve().parents[3] / "configs" / "hosts"


@lru_cache(maxsize=1)
def _extractor():  # type: ignore[no-untyped-def]
    import tldextract

    # suffix_list_urls=() disables the network fetch, so every run uses the snapshot
    # shipped with the pinned tldextract version.
    return tldextract.TLDExtract(suffix_list_urls=(), fallback_to_snapshot=True)


def registered_domain(url: str) -> str | None:
    """eTLD+1 for a URL, e.g. https://news.acme.co.uk/x -> acme.co.uk."""
    parts = _extractor()(url)
    if not parts.domain or not parts.suffix:
        return None
    return f"{parts.domain}.{parts.suffix}".lower()


def subdomain(url: str) -> str:
    return str(_extractor()(url).subdomain).lower()


@cache
def load_host_list(name: str) -> frozenset[str]:
    """Read configs/hosts/<name>.txt. Blank lines and # comments ignored."""
    path = HOSTS_DIR / f"{name}.txt"
    if not path.exists():
        return frozenset()
    entries = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().lower()
        if line:
            entries.add(line)
    return frozenset(entries)


# Structural rules, which generalize far beyond any host list.
PR_PATH = re.compile(
    r"/(press[-_]?releases?|news[-_]?releases?|newsroom"
    r"|media[-_](centre|center|room)|investor[-_]relations|investors?/news)/",
    re.I,
)
PR_SUBDOMAIN = re.compile(r"^(news|press|media|newsroom|investors?|ir)$", re.I)
PRODUCT_PATH = re.compile(r"/(products?|p|dp/B0|itm|collections/[^/]+/products)/", re.I)
FORUM_PATH = re.compile(
    r"(/(forums?|showthread|viewtopic|threads?|topic|t)/"
    r"|\.php\?[^&]*\b(t|topic|showtopic|f)="
    r"|/index\.php\?topic=)",
    re.I,
)
BLOG_PATH = re.compile(r"/blogs?/", re.I)
# Paths that mark a host as a company rather than a person. Used to tell a corporate
# blog from a personal one, which no path pattern alone can do.
CORPORATE_PATH = re.compile(
    r"/(pricing|products?|features|solutions|customers|case[-_]stud|enterprise"
    r"|request[-_]a?[-_]?demo|book[-_]a?[-_]?demo|our[-_]team|careers)/",
    re.I,
)


def is_corporate_host(paths: set[str]) -> bool:
    """Does this host also serve company pages?

    Computed once per host over all its URLs in a dump, which is the cheapest reliable
    corporate-vs-personal blog signal that needs no external data.
    """
    return any(CORPORATE_PATH.search(p) for p in paths)


def classify_url(
    url: str,
    *,
    corporate_host: bool = False,
) -> tuple[HardNegativeKind | None, str] | tuple[None, str]:
    """Route a URL to a hard-negative kind. Returns (kind, rule) for the audit trail."""
    domain = registered_domain(url)
    if domain is None:
        return None, ""
    split = urlsplit(url)
    path = split.path + ("?" + split.query if split.query else "")

    if domain in load_host_list("press_release"):
        return "press_release", f"host:press_release:{domain}"
    if PR_PATH.search(path):
        return "press_release", "path:PR_PATH"
    if PR_SUBDOMAIN.match(subdomain(url).split(".")[-1] or ""):
        return "press_release", "subdomain:PR_SUBDOMAIN"

    if domain in load_host_list("seo_marketing"):
        return "seo_marketing", f"host:seo_marketing:{domain}"

    if domain in load_host_list("product"):
        return "product_template", f"host:product:{domain}"
    if PRODUCT_PATH.search(path):
        return "product_template", "path:PRODUCT_PATH"

    if BLOG_PATH.search(path):
        # A /blog/ path on a blogging platform is a personal blog, not a corporate one.
        if domain in load_host_list("personal_blog_platforms"):
            return None, "path:BLOG_PATH+personal_platform"
        if corporate_host:
            return "corporate_blog", "path:BLOG_PATH+corporate_host"

    if domain in load_host_list("nonnative_forum") or FORUM_PATH.search(path):
        return "non_native_forum", "path:FORUM_PATH"

    return None, ""
