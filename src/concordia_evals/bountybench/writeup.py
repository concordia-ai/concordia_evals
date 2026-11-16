"""Rendering of the captured bug bounty reports into the prompt text.

Ported from `resources/utils.py` of the upstream repository
(https://github.com/bountybench/bountybench, commit `8ece6aab`).
"""

import html
import re
from typing import Any


def _soup(markup: str) -> Any:
    # Deferred import: bs4 is only needed when a writeup is rendered.
    from bs4 import BeautifulSoup

    return BeautifulSoup(markup, "html.parser")


def format_huntr_writeup(writeup: str) -> str:
    """Render a captured huntr.com report."""
    soup = _soup(writeup)

    formatted_content = ["======================", "Writeup:\n"]

    # Title and status
    title = soup.find("h1", id="title")
    if title:
        formatted_content.append("Title:")
        formatted_content.append(title.text.strip())
        status_span = soup.find("span", class_="mr-1 text-xs font-medium")
        if status_span:
            formatted_content.append(status_span.text.strip())
        report_date = soup.find("p", class_="ml-3 mt-1 self-center text-xs font-medium")
        if report_date:
            formatted_content.append(report_date.text.strip())
        formatted_content.append("")

    # Description and Proof of Concept
    markdown_body = soup.find("div", class_="markdown-body")
    if markdown_body:
        for child in markdown_body.children:
            if child.name and child.name.startswith("h"):
                formatted_content.append(f"\n{child.text}")
            elif child.name in ["p", "pre"]:
                formatted_content.append(html.unescape(child.text.strip()))
            elif child.name in ["ul", "ol"]:
                for li in child.find_all("li"):
                    formatted_content.append(f"- {li.text.strip()}")

    # Occurences Title
    occurrences_title = soup.find("h1", id="occurrence-title")
    if occurrences_title:
        formatted_content.append("\nOccurrences:")
        clean_code_ref = soup.find("div", class_="permalinkMD mb-3 inline-block")
        if clean_code_ref:
            formatted_content.append(clean_code_ref.text.strip())
        else:
            clean_code_ref = soup.find("div", class_="permalinkMD")
            if clean_code_ref:
                filename_tag = clean_code_ref.select_one("a.permalink p")
                content_tag = clean_code_ref.select_one("div.markdown-body")

                if filename_tag and content_tag:
                    filename = filename_tag.text.strip()
                    content = content_tag.text.strip()
                    formatted_content.append(f"{filename} {content}")

    # Reference Title
    reference_title = soup.find("h2", id="reference-title")
    if reference_title:
        formatted_content.append("\nReference:")
        next_elem = reference_title.find_next_sibling()
        while next_elem and next_elem.name not in ["ul", "ol"]:
            next_elem = next_elem.find_next_sibling()
        if next_elem:
            for li in next_elem.find_all("li"):
                formatted_content.append(f"- {li.text.strip()}")

    formatted_content.append("======================")
    return "\n".join(formatted_content)


def format_hackerone_writeup(writeup: str) -> str:
    """Render a captured hackerone.com report."""
    soup = _soup(writeup)

    formatted_content = ["======================", "Writeup:\n"]

    # Title and report number
    title_element = soup.find(
        "div",
        class_="false report-heading__report-title spec-report-title flex items-center",
    )
    if title_element:
        title_div = title_element.find("div", class_="break-word")
        if title_div:
            formatted_content.append("Title:")
            formatted_content.append(title_div.text.strip())

    # Get report ID
    report_status = soup.find("div", class_="report-status")
    if report_status:
        report_id = report_status.find("a", class_="report-status-indicator")
        if report_id:
            formatted_content.append(report_id.text.strip())

    # Get severity
    severity_element = soup.find("span", class_="severity-label__text")
    if severity_element:
        severity_rating = severity_element.find("span", class_="spec-severity-rating")
        if severity_rating:
            formatted_content.append(f"Severity: {severity_rating.text.strip()}")

    formatted_content.append("")

    # Get the summary content
    summary = soup.find("div", class_="spec-full-summary-content")
    if summary:
        formatted_content.append("Summary:")
        # Extract paragraphs from summary
        paragraphs = summary.find_all("div", class_="vertical-spacing interactive_markdown__p")
        for p in paragraphs:
            formatted_content.append(html.unescape(p.text.strip()))
        formatted_content.append("")

    # Get vulnerability information
    vuln_info = soup.find("div", class_="spec-vulnerability-information")
    if vuln_info:
        for wrapper in vuln_info.find_all("div", class_="interactive-markdown"):
            for child in wrapper.find_all(recursive=False):
                cls = child.get("class", [])
                ids = child.get("id", [])
                if "interactive_markdown__p" in cls:
                    formatted_content.append(html.unescape(child.get_text()))

                elif "interactive-markdown__code" in cls:
                    code_div = child.find("div", class_="interactive-markdown__code__content")
                    if not code_div:
                        continue

                    # Remove line numbers
                    for num in code_div.find_all(
                        "span", class_=re.compile(r"(linenumber|line-number)")
                    ):
                        num.decompose()

                    # Get the code text (separator="" works best here)
                    code_text = code_div.get_text(separator="").strip()
                    formatted_content.append("Code:\n" + code_text + "\n")

                # Handle the impact section header
                elif "user-content-impact" in ids:
                    formatted_content.append(child.get_text(strip=True) + ":")

                elif "user-content-steps-to-reproduce" in ids:
                    formatted_content.append(child.get_text(strip=True))

    formatted_content.append("======================")
    return "\n".join(formatted_content)


def read_writeup(writeup_html: str, bounty_link: str | None) -> str:
    """Render the stored bounty report, selecting the parser from the report URL."""
    if not bounty_link:
        return writeup_html
    if "huntr" in bounty_link:
        return format_huntr_writeup(writeup_html)
    if "hackerone" in bounty_link:
        return format_hackerone_writeup(writeup_html)
    raise ValueError("Only huntr and hackerone bounty reports can currently be parsed.")
