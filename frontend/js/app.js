'use strict';

/* CODDY frontend.

   Two things here are load-bearing and easy to regress:

   1. renderMarkdown() sanitises before the result ever reaches innerHTML.
      Assistant output quotes uploaded documents, so it is untrusted input.
      DOMPurify is the boundary. Do not remove it, and do not swap it for
      marked alone.

   2. The SSE reader carries a buffer between network reads. Frames do not
      align with chunk boundaries, so splitting each read independently drops
      or truncates events.
*/

const TOKEN_KEY = 'coddy_token';

const state = {
  token: localStorage.getItem(TOKEN_KEY) || '',
  user: null,
  mode: 'login',
  sessionId: null,
  busy: false,
};

// ---------------------------------------------------------------------------
// DOMPurify policy
// ---------------------------------------------------------------------------
const PURIFY_CONFIG = {
  ALLOWED_TAGS: [
    'p', 'br', 'hr', 'div', 'span', 'blockquote', 'pre', 'code',
    'strong', 'b', 'em', 'i', 'u', 's', 'del', 'mark', 'sub', 'sup',
    'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'a', 'img', 'table', 'thead', 'tbody', 'tr', 'th', 'td',
  ],
  ALLOWED_ATTR: ['class', 'href', 'title', 'alt', 'lang'],
  // Blocks javascript: and data: URLs so a model cannot emit a live link.
  ALLOWED_URI_REGEXP: /^(?:https?|mailto):/i,
  FORBID_TAGS: ['style', 'form', 'input', 'iframe', 'object', 'embed', 'svg', 'math', 'link'],
  FORBID_ATTR: ['onerror', 'onload', 'onclick', 'onmouseover', 'style', 'srcset'],
  ADD_ATTR: ['target', 'rel'],
};

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;',
  }[ch]));
}

function sanitizeHtml(html) {
  if (window.DOMPurify) return DOMPurify.sanitize(html, PURIFY_CONFIG);
  // Fail closed. If the vendored library failed to load, show inert text
  // rather than injecting markup.
  console.error('DOMPurify unavailable - rendering escaped text instead.');
  return escapeHtml(html);
}

function renderMarkdown(text) {
  if (!text) return '';
  let html = window.marked
    ? marked.parse(String(text), { breaks: true, gfm: true })
    : escapeHtml(text);
  return sanitizeHtml(html);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

function setText(el, value) {
  el.textContent = value == null ? '' : String(value);
}

async function api(path, options = {}) {
  const opts = { ...options, headers: { ...(options.headers || {}) } };
  if (state.token) opts.headers.Authorization = `Bearer ${state.token}`;
  if (opts.body && !(opts.body instanceof FormData)) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(opts.body);
  }

  const response = await fetch(path, opts);

  if (response.status === 401) {
    signOut();
    throw new Error('Session expired. Sign in again.');
  }
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const data = await response.json();
      if (data && data.detail) {
        detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
      }
    } catch (_) { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------
function showAuth() {
  $('auth-screen').classList.remove('hidden');
  $('app').classList.add('hidden');
}

function showApp() {
  $('auth-screen').classList.add('hidden');
  $('app').classList.remove('hidden');
}

function signOut() {
  state.token = '';
  state.user = null;
  state.sessionId = null;
  localStorage.removeItem(TOKEN_KEY);
  showAuth();
}

async function submitAuth(event) {
  event.preventDefault();
  const email = $('auth-email').value.trim();
  const password = $('auth-password').value;
  const errorEl = $('auth-error');
  errorEl.textContent = '';

  if (password.length > 72) {
    errorEl.textContent = 'Password must be at most 72 bytes.';
    return;
  }

  const button = $('auth-submit');
  button.disabled = true;
  try {
    const path = state.mode === 'login' ? '/api/auth/login' : '/api/auth/register';
    const data = await api(path, { method: 'POST', body: { email, password } });

    if (state.mode === 'register') {
      $('auth-hint').textContent = 'Account created. Signing you in…';
      const login = await api('/api/auth/login', { method: 'POST', body: { email, password } });
      state.token = login.access_token;
    } else {
      state.token = data.access_token;
    }

    localStorage.setItem(TOKEN_KEY, state.token);
    $('auth-password').value = '';
    await enterApp();
  } catch (err) {
    errorEl.textContent = err.message;
  } finally {
    button.disabled = false;
  }
}

async function enterApp() {
  state.user = await api('/api/auth/profile');
  showApp();
  await Promise.all([loadSessions(), loadDocuments(), loadStatus()]);
  await createSession();
}

function wireAuth() {
  $('auth-form').addEventListener('submit', submitAuth);

  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.remove('is-active'));
      tab.classList.add('is-active');
      state.mode = tab.dataset.mode;
      $('auth-submit').textContent = state.mode === 'login' ? 'Sign in' : 'Create account';
      $('auth-error').textContent = '';
      $('auth-hint').textContent = '';
      $('auth-password').autocomplete = state.mode === 'login' ? 'current-password' : 'new-password';
    });
  });

  $('logout-btn').addEventListener('click', signOut);
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------
async function loadSessions() {
  try {
    const sessions = await api('/api/chat/sessions');
    const list = $('session-list');
    list.replaceChildren();

    if (!sessions.length) {
      const li = document.createElement('li');
      li.className = 'muted small';
      setText(li, 'No conversations yet');
      list.appendChild(li);
      return;
    }

    sessions.forEach((session) => {
      const li = document.createElement('li');
      li.className = state.sessionId === session.session_id ? 'is-active' : '';
      li.addEventListener('click', () => openSession(session.session_id));

      const label = document.createElement('span');
      label.className = 'label';
      setText(label, session.title);            // user-controlled: textContent

      const actions = document.createElement('span');
      actions.className = 'row-actions';

      const rename = document.createElement('button');
      setText(rename, 'Rename');
      rename.addEventListener('click', (event) => {
        event.stopPropagation();
        const next = window.prompt('New title', session.title);
        if (next && next.trim()) {
          api(`/api/chat/sessions/${session.session_id}`, { method: 'PUT', body: { title: next.trim() } })
            .then(loadSessions)
            .catch((err) => window.alert(err.message));
        }
      });

      const remove = document.createElement('button');
      remove.className = 'danger';
      setText(remove, 'Delete');
      remove.addEventListener('click', (event) => {
        event.stopPropagation();
        if (!window.confirm('Delete this conversation?')) return;
        api(`/api/chat/sessions/${session.session_id}`, { method: 'DELETE' })
          .then(() => {
            if (state.sessionId === session.session_id) createSession();
            else loadSessions();
          })
          .catch((err) => window.alert(err.message));
      });

      actions.append(rename, remove);
      li.append(label, actions);
      list.appendChild(li);
    });
  } catch (err) {
    console.error('loadSessions failed', err);
  }
}

async function createSession() {
  const session = await api('/api/chat/sessions', { method: 'POST', body: { title: 'New conversation' } });
  state.sessionId = session.session_id;
  setText($('chat-title'), session.title);
  $('messages').replaceChildren(emptyState());
  $('empty-state').classList.remove('hidden');
  await loadSessions();
  return session.session_id;
}

function emptyState() {
  const wrap = document.createElement('div');
  wrap.className = 'empty';
  const heading = document.createElement('h3');
  setText(heading, 'Ask about your documents');
  const body = document.createElement('p');
  body.className = 'muted';
  setText(body, 'Upload a PDF, DOCX, TXT, CSV or Markdown file, then ask a question. Answers are grounded in what you uploaded and cite the source.');
  wrap.append(heading, body);
  return wrap;
}

async function openSession(sessionId) {
  state.sessionId = sessionId;
  const messages = $('messages');
  messages.replaceChildren();

  try {
    const history = await api(`/api/chat/sessions/${sessionId}/messages`);
    setText($('chat-title'), history.length ? history[0].content.slice(0, 50) : 'Conversation');

    if (!history.length) {
      messages.appendChild(emptyState());
    } else {
      history.forEach((message) => appendMessage(message.sender, message.content, message.sources));
    }
    await loadSessions();
    scrollToBottom();
  } catch (err) {
    window.alert(err.message);
  }
}

$('new-chat-btn').addEventListener('click', () => {
  if (state.busy) return;
  createSession().catch((err) => window.alert(err.message));
});

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------
async function loadDocuments() {
  try {
    const documents = await api('/api/documents');
    const list = $('document-list');
    list.replaceChildren();
    setText($('doc-count'), `${documents.length} document${documents.length === 1 ? '' : 's'}`);

    if (!documents.length) {
      const li = document.createElement('li');
      li.className = 'muted small';
      setText(li, 'Nothing uploaded yet');
      list.appendChild(li);
      return;
    }

    documents.forEach((doc) => {
      const li = document.createElement('li');

      const label = document.createElement('span');
      label.className = 'label';
      // setText, never innerHTML: the filename is user-controlled.
      setText(label, `${doc.name} · ${(doc.size_bytes / 1024).toFixed(0)}KB`);
      label.title = doc.name;

      const actions = document.createElement('span');
      actions.className = 'row-actions';

      const reindex = document.createElement('button');
      setText(reindex, 'Reindex');
      reindex.addEventListener('click', (event) => {
        event.stopPropagation();
        reindexButton(doc, reindex);
      });

      const remove = document.createElement('button');
      remove.className = 'danger';
      setText(remove, 'Delete');
      remove.addEventListener('click', (event) => {
        event.stopPropagation();
        if (!window.confirm(`Delete "${doc.name}"?`)) return;
        api(`/api/documents/${doc.id}`, { method: 'DELETE' })
          .then(loadDocuments)
          .catch((err) => window.alert(err.message));
      });

      actions.append(reindex, remove);
      li.append(label, actions);
      list.appendChild(li);
    });
  } catch (err) {
    console.error('loadDocuments failed', err);
  }
}

async function reindexButton(doc, button) {
  const original = button.textContent;
  button.disabled = true;
  setText(button, 'Working…');
  try {
    await api(`/api/documents/${doc.id}/reindex`, { method: 'POST' });
  } catch (err) {
    window.alert(err.message);
  } finally {
    button.disabled = false;
    setText(button, original);
  }
}

$('file-input').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const hint = $('upload-hint');
  const form = new FormData();
  form.append('file', file);
  setText(hint, `Indexing ${file.name}…`);

  try {
    const result = await api('/api/documents', { method: 'POST', body: form });
    setText(hint, `${result.name} → ${result.chunks} chunks`);
    await loadDocuments();
  } catch (err) {
    setText(hint, err.message);
  } finally {
    event.target.value = '';
    setTimeout(() => setText(hint, ''), 6000);
  }
});

async function loadStatus() {
  const el = $('model-status');
  try {
    const health = await api('/api/health');
    const detail = health.details || {};
    const model = detail.model_id || health.llm;
    const vram = detail.vram_total_gb ? ` · ${detail.vram_total_gb}GB VRAM` : '';
    setText(el, `${model}${detail.quantization ? ` (${detail.quantization})` : ''}${vram}`);
  } catch (_) {
    setText(el, 'status unavailable');
  }
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------
function appendMessage(sender, content, sources) {
  $('empty-state')?.classList.add('hidden');

  const wrap = document.createElement('div');
  wrap.className = `msg ${sender === 'user' ? 'user' : ''}`;

  const body = document.createElement('div');
  body.className = 'msg-body';

  const role = document.createElement('div');
  role.className = 'msg-role';
  setText(role, sender === 'user' ? 'You' : 'CODDY');

  const text = document.createElement('div');
  text.className = 'msg-content';
  // Both branches are safe: user text is escaped, model text is sanitized.
  if (sender === 'user') text.textContent = content;
  else text.innerHTML = renderMarkdown(content);

  body.append(role, text);
  wrap.appendChild(body);
  $('messages').appendChild(wrap);
  scrollToBottom();

  return { wrap, text };
}

function appendCitations(card, citations) {
  if (!citations || !citations.length) return;

  const box = document.createElement('div');
  box.className = 'citations';

  const label = document.createElement('div');
  label.className = 'citations-label';
  setText(label, 'Sources');

  const chips = document.createElement('div');
  chips.className = 'citation-chips';

  citations.forEach((citation) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip';
    setText(chip, `[${citation.position}] ${citation.file_name} p${citation.page_number}`);
    chip.addEventListener('click', () => showCitation(citation));
    chips.appendChild(chip);
  });

  box.append(label, chips);
  card.querySelector('.msg-body').appendChild(box);
}

function showCitation(citation) {
  setText($('citation-title'), citation.file_name);
  setText($('citation-meta'), `Segment ${citation.position} · page ${citation.page_number}`);
  setText($('citation-body'), citation.excerpt || '');
  $('citation-modal').classList.remove('hidden');
}

$('citation-close').addEventListener('click', () => $('citation-modal').classList.add('hidden'));
$('citation-modal').addEventListener('click', (event) => {
  if (event.target === $('citation-modal')) $('citation-modal').classList.add('hidden');
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') $('citation-modal').classList.add('hidden');
});

function scrollToBottom() {
  const box = $('messages');
  box.scrollTop = box.scrollHeight;
}

// ---------------------------------------------------------------------------
// Chat streaming
// ---------------------------------------------------------------------------
$('composer').addEventListener('submit', sendMessage);

async function sendMessage(event) {
  event.preventDefault();
  if (state.busy) return;

  const input = $('query');
  const query = input.value.trim();
  if (!query) return;
  if (!state.sessionId) return;

  input.value = '';
  input.style.height = 'auto';
  appendMessage('user', query);

  state.busy = true;
  $('send-btn').disabled = true;
  setText($('chat-title'), query.slice(0, 50));

  const card = appendMessage('assistant', '');
  const content = card.text;

  const indicator = document.createElement('div');
  indicator.className = 'thinking';
  indicator.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  content.replaceChildren(indicator);

  let accumulated = '';
  let citations = null;

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${state.token}` },
      body: JSON.stringify({ query, session_id: state.sessionId }),
    });

    if (!response.ok) {
      let detail = `Request failed (${response.status})`;
      try {
        const data = await response.json();
        if (data && data.detail) detail = String(data.detail);
      } catch (_) { /* non-JSON body */ }
      throw new Error(detail);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');

    // SSE frames end with a blank line and can straddle network reads, so the
    // remainder is carried forward. Without this, any frame split across two
    // reads is dropped and the answer loses tokens.
    let buffer = '';
    let streamError = null;
    let done = false;

    while (!done) {
      const { value, done: finished } = await reader.read();
      done = finished;

      buffer += value ? decoder.decode(value, { stream: true }) : decoder.decode();

      const frames = buffer.split('\n\n');
      buffer = done ? '' : frames.pop();

      for (const frame of frames) {
        for (const line of frame.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6);
          if (!raw) continue;

          let event;
          try {
            event = JSON.parse(raw);
          } catch (_) {
            continue;
          }

          if (event.error) {
            streamError = event.error;
            done = true;
            break;
          }
          if (event.token) {
            accumulated += event.token;
            content.innerHTML = renderMarkdown(accumulated);
            scrollToBottom();
          }
          if (event.citations) citations = event.citations;
        }
        if (streamError) break;
      }
    }

    if (streamError) {
      content.replaceChildren();
      const fail = document.createElement('p');
      fail.className = 'error';
      setText(fail, streamError);
      content.appendChild(fail);
    } else if (!accumulated) {
      content.replaceChildren();
      const blank = document.createElement('p');
      blank.className = 'muted';
      setText(blank, 'The model returned an empty response.');
      content.appendChild(blank);
    }

    appendCitations(card.wrap, citations);
    await loadSessions();
  } catch (err) {
    content.replaceChildren();
    const fail = document.createElement('p');
    fail.className = 'error';
    setText(fail, err.message);
    content.appendChild(fail);
  } finally {
    state.busy = false;
    $('send-btn').disabled = false;
    input.focus();
  }
}

// Auto-grow the textarea.
$('query').addEventListener('input', (event) => {
  const el = event.target;
  el.style.height = 'auto';
  el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
});

$('query').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    $('composer').requestSubmit();
  }
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
(async function init() {
  wireAuth();
  if (!state.token) {
    showAuth();
    return;
  }
  try {
    await enterApp();
  } catch (_) {
    // Token is stale or rejected; fall back to the sign-in screen.
    signOut();
  }
})();