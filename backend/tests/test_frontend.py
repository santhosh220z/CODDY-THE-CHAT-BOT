"""Static checks on the frontend.

No browser runs in CI, so these cover the failure modes that would otherwise
only show up as a blank page or a silent security regression: an element id
that JS references but the HTML never declares, a template literal
interpolated straight into innerHTML, or a CDN dependency creeping back in.
"""

import pathlib
import re

import pytest

FRONTEND = pathlib.Path(__file__).resolve().parents[2] / "frontend"
HTML = FRONTEND / "index.html"
JS = FRONTEND / "js" / "app.js"
CSS = FRONTEND / "css" / "style.css"


@pytest.fixture(scope="module")
def html():
    return HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def js():
    return JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def js_code():
    """app.js with comments removed.

    Without this the assertions below match prose - a comment reading
    "never innerHTML: the filename is user-controlled" trips a check that
    innerHTML is never assigned from a filename.
    """
    source = JS.read_text(encoding="utf-8")
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    source = re.sub(r"//[^\n]*", "", source)
    return source


@pytest.fixture(scope="module")
def css():
    return CSS.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------
def test_every_element_id_referenced_by_js_exists(html, js):
    declared = set(re.findall(r'\bid="([^"]+)"', html))
    referenced = set(re.findall(r"\$\('([^']+)'\)", js)) | set(
        re.findall(r"getElementById\('([^']+)'\)", js)
    )
    assert referenced, "no ids referenced - the id lookup pattern may have changed"
    assert not referenced - declared, f"missing ids: {sorted(referenced - declared)}"


def test_local_assets_exist(html):
    for match in re.findall(r'(?:src|href)="(/static/[^"]+)"', html):
        assert (FRONTEND / match[len("/static/"):]).exists(), f"missing {match}"


def test_vendored_libraries_are_present():
    assert (FRONTEND / "js" / "vendor" / "purify.min.js").exists()
    assert (FRONTEND / "js" / "vendor" / "marked.min.js").exists()


def test_script_tags_are_balanced(html):
    assert html.count("<script") == html.count("</script>")


def test_css_braces_are_balanced(css):
    assert css.count("{") == css.count("}")


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
def test_no_third_party_origins(html):
    """The app runs locally; a CDN is a supply-chain risk on a page that
    renders untrusted document text."""
    assert not re.findall(r'(?:src|href)="https?://', html)


def test_no_interpolated_template_goes_into_innerhtml(js_code):
    # `el.innerHTML = ` + backtick + `...${value}...` + backtick + ` is dangerous.
    assert not re.findall(r"\.innerHTML\s*=\s*`[^`]*\$\{", js_code)


def test_model_output_is_sanitised(js_code):
    assert "DOMPurify.sanitize(" in js_code
    # Must fail closed rather than render raw HTML if the library is missing.
    assert "if (window.DOMPurify)" in js_code
    assert re.search(r"\.innerHTML\s*=\s*renderMarkdown\(", js_code)


def test_user_controlled_values_use_textcontent(js_code):
    assert re.search(r"sender === 'user'\) text\.textContent", js_code)
    # Session titles are derived from the user's own chat query.
    assert "setText(open, session.title)" in js_code
    # Uploaded filenames are user-controlled too. The name sits in its own
    # cell and the size in a separate mono column.
    assert "setText(label, doc.name)" in js_code
    # And must never be assigned to innerHTML anywhere.
    assert not re.search(r"innerHTML[^;]*doc\.name", js_code)


def test_no_inline_event_handlers(js):
    """Inline handlers inside sanitised markup are the wrong tool; the app
    uses delegated listeners instead."""
    assert "onclick=" not in js
    assert "onerror=" not in js
    assert "addEventListener" in js


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
def test_sse_reader_buffers_between_network_reads(js):
    """Frames do not align with reads. Dropping the remainder truncates
    answers at random points."""
    assert "frames.pop()" in js
    assert "decoder.decode(value, { stream: true })" in js
    assert "decoder.decode()" in js, "must flush the decoder at end of stream"


def test_sse_reader_handles_errors(js):
    assert "event.error" in js


# ---------------------------------------------------------------------------
# Accessibility
# ---------------------------------------------------------------------------
def test_basic_accessibility_markers(html):
    assert "<label" in html
    assert 'role="dialog"' in html
    assert 'aria-modal="true"' in html
    assert html.count("aria-label=") >= 2
    assert "placeholder=" in html
    assert 'lang="en"' in html


def test_reduced_motion_is_respected(css):
    assert "prefers-reduced-motion" in css


def test_responsive_breakpoint_exists(css):
    assert "@media" in css