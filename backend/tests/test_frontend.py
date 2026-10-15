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


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------
def _token_block(css, selector):
    """The custom properties declared for one selector."""
    pattern = re.escape(selector) + r"\s*\{(.*?)\}"
    match = re.search(pattern, css, re.DOTALL)
    assert match, f"no rule found for {selector}"
    return set(re.findall(r"^\s*(--[a-z0-9-]+)\s*:", match.group(1), re.MULTILINE))


# Tokens that are genuinely the same in both themes, so the dark block does not
# redeclare them. Everything else is a colour and must be restated, otherwise
# it silently renders with its light value on a dark surface.
THEME_INVARIANT_TOKENS = {
    "--r-sm",
    "--r",
    "--r-lg",
    "--serif",
    "--sans",
    "--mono",
    "--spine-w",
}


def test_dark_theme_defines_every_colour_token(css):
    """R-34: a theme that ships broken is a defect.

    The failure this catches is the common one: a colour token is added later
    for a new component, the dark block is never updated, and that one
    property keeps its light value on a dark surface.
    """
    light = _token_block(css, ":root")
    dark = _token_block(css, ':root[data-theme="dark"]')

    assert dark, "no dark theme block"
    assert light - dark - THEME_INVARIANT_TOKENS == set(), (
        f"tokens missing from the dark theme: "
        f"{sorted(light - dark - THEME_INVARIANT_TOKENS)}"
    )


def test_both_themes_declare_color_scheme(css):
    """Form controls and scrollbars follow the page only if this is set."""
    assert "color-scheme: light" in css
    assert re.search(r'data-theme="dark"\][^{]*\{[^}]*color-scheme:\s*dark', css, re.DOTALL)


def test_theme_is_resolved_before_first_paint(html):
    """The head bootstrap must come after the stylesheet but before the body.

    Moving it into the body reintroduces the flash of the wrong theme.
    """
    assert "coddy_theme" in html
    assert "prefers-color-scheme: dark" in html
    head = html.split("</head>")[0]
    assert head.index("style.css") < head.index("coddy_theme")


def test_theme_toggle_is_wired_and_labelled(html, js):
    assert 'id="theme-toggle"' in html
    # Both glyphs ship; CSS shows the one matching the destination theme.
    assert "icon-sun" in html and "icon-moon" in html
    assert 'aria-label="Switch to dark theme"' in html
    assert "$('theme-toggle').addEventListener('click'" in js
    assert "localStorage.setItem(THEME_KEY, next)" in js
    # No aria-pressed alongside a label that already names the action, which
    # would contradict itself.
    assert "aria-pressed" not in html


def test_theme_toggle_is_reachable_without_opening_the_drawer(html):
    """The spine is an off canvas drawer on narrow screens.

    A control that lives only inside it cannot be reached while the drawer is
    closed, so the toggle belongs to the masthead.
    """
    masthead = html.split('class="masthead"')[1].split("</header>")[0]
    assert 'id="theme-toggle"' in masthead

# ---------------------------------------------------------------------------
# Streaming, state and session invariants
# ---------------------------------------------------------------------------
def test_pending_frame_is_cancelled_before_the_final_render(js_code):
    """A queued animation frame outlives the stream loop.

    It fired after the sanitised markdown was written and replaced it with
    raw text, so every answer ended unformatted.
    """
    assert "cancelAnimationFrame" in js_code
    assert re.search(r"frame\s*=\s*requestAnimationFrame", js_code)
    # Scoped to sendMessage: appendMessage also renders markdown, and finding
    # that first would compare the wrong two offsets.
    send = js_code[js_code.index("async function sendMessage"):]
    cancel = send.index("cancelPendingFrame();", send.index("while (!done)"))
    render = send.index(".innerHTML = renderMarkdown(")
    assert cancel < render, "frame cancelled after the final render"


def test_streaming_request_refreshes_an_expired_token(js_code):
    """/api/chat needs the raw Response to read the body, so it cannot use
    api(). Without repeating the refresh-and-replay it 401s forever after
    ACCESS_TOKEN_EXPIRE_MINUTES, with no way back but a reload."""
    assert "async function postChat" in js_code
    post_chat = js_code[js_code.index("async function postChat"):]
    assert "localSession()" in post_chat, "streaming path skips the token refresh"
    assert post_chat.count("send()") >= 2, "the request is never replayed"


def test_rename_limit_matches_the_api(js, js_code):
    """A client limit looser than the server's produces a rejection, and the
    rejection is the path that used to leave the row stuck as a bare input."""
    assert "input.maxLength = 200" in js_code
    assert "input.maxLength = 300" not in js_code
    # The row is replaced wholesale, so a failure must put it back.
    assert "li.replaceChildren(input)" in js_code
    rename = js_code[js_code.index("function beginRename"):js_code.index("async function openSession")]
    assert "loadSessions()" in rename.split("catch")[-1], "failure path does not restore the row"


def test_opening_a_conversation_is_refused_mid_stream(js_code):
    """openSession emptied the column, so switching mid-answer detached the
    node the tokens were still being written into."""
    open_session = js_code[js_code.index("async function openSession"):]
    open_session = open_session[: open_session.index("\nfunction ")]
    assert "if (state.busy) return;" in open_session
    # The session is claimed only once the history has actually arrived, so a
    # failure cannot leave the composer posting to a conversation on screen.
    assert open_session.index("await api(") < open_session.index("state.sessionId = sessionId")
    # And the column is put back rather than left blank.
    assert "messages.replaceChildren(emptyState())" in open_session


def test_health_polling_is_bounded(js_code):
    """With prewarm off the model never loads until the first question, so an
    unbounded poll requested /api/health every six seconds for the life of the
    page while reporting a load that was not happening."""
    assert "STATUS_POLL_LIMIT" in js_code
    assert "status-retry" in js_code, "no manual re-check once polling gives up"
    assert "scheduleStatus(STATUS_POLL_MS)" in js_code


def test_failure_inside_a_row_restores_the_list_before_reporting(js_code):
    """loadSessions clears the list-state on success, so a fault written first
    would be wiped. The restore has to happen, then the message."""
    assert js_code.count("loadSessions().then(") >= 2
    assert js_code.count("loadDocuments().then(") >= 1


def test_boot_paths_do_not_duplicate_the_enter_sequence(js):
    """The fresh-boot and stored-token paths were the same five lines twice."""
    assert "async function enterApp(" in js
    assert js.count("await Promise.all([loadSessions(), loadDocuments(), loadStatus()])") == 1
    assert "state.user" not in js, "assigned twice and never read"

def test_working_label_names_the_wait_it_is_in(js_code):
    """The first question said "Searching your documents" while it was actually
    blocked on a 20 second model load, so a slow start was indistinguishable
    from a hung request."""
    assert "modelReady" in js_code
    assert "this only happens on the first question" in js_code
    # The wording must follow the readiness the health probe already reports,
    # not a new event invented to carry it.
    assert "state.modelReady = true" in js_code
    assert "state.modelReady = false" in js_code