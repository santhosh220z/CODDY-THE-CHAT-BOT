'use strict';

/* CODDY frontend.

   Four things here are load-bearing and easy to regress:

   1. renderMarkdown() sanitises before the result ever reaches innerHTML.
      Assistant output quotes uploaded documents, so it is untrusted input.
      DOMPurify is the boundary. Do not remove it, and do not swap it for
      marked alone.

   2. The SSE reader carries a buffer between network reads. Frames do not
      align with chunk boundaries, so splitting each read independently drops
      or truncates events.

   3. Every list has three states: loading, empty, error. Each names the cause
      and the next action. A list that only renders the happy path tells the
      reader nothing when a fetch fails.

   4. There is no sign-in step. The server provisions one local account and
      hands out a session, so a 401 means the token expired rather than that
      the reader was never authenticated. It is refreshed once and the request
      retried, never treated as a reason to show a login screen.

   Streaming renders textContent per animation frame and parses markdown once
   at the end. Parsing the whole buffer on every token was quadratic and made
   half-finished markdown flicker as it re-parsed.
*/

const TOKEN_KEY = 'coddy_token';
const THEME_KEY = 'coddy_theme';

const state = {
  token: localStorage.getItem(TOKEN_KEY) || '',
  sessionId: null,
  busy: false,
  // Mirrors #model-dot. The first question used to say "Searching your
  // documents" while it was actually waiting on a 20 second model load, so a
  // slow start was indistinguishable from a hung request.
  modelReady: false,
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

function makeEl(tag, className, value) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (value != null) setText(el, value);
  return el;
}

/* The server provisions its own account, so a session is acquired rather than
   entered. Concurrent callers share one in-flight request. */
let sessionRequest = null;

function localSession() {
  if (!sessionRequest) {
    sessionRequest = fetch('/api/auth/local')
      .then((response) => {
        if (!response.ok) throw new Error(`The server returned ${response.status}.`);
        return response.json();
      })
      .then((data) => {
        state.token = data.access_token;
        localStorage.setItem(TOKEN_KEY, state.token);
        return data;
      })
      .finally(() => {
        sessionRequest = null;
      });
  }
  return sessionRequest;
}

async function api(path, options = {}, retried = false) {
  const opts = { ...options, headers: { ...(options.headers || {}) } };
  if (state.token) opts.headers.Authorization = `Bearer ${state.token}`;
  if (opts.body && !(opts.body instanceof FormData)) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(opts.body);
  }

  const response = await fetch(path, opts);

  // A 401 here means the stored token went stale. Refreshing once and
  // replaying keeps an expired session from throwing the reader out of a
  // local app that has no way to sign back in.
  if (response.status === 401 && !retried) {
    await localSession();
    return api(path, options, true);
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const data = await response.json();
      if (data && data.detail) {
        detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
      }
    } catch (_) { /* non-JSON error body */ }
    if (response.status === 401) {
      localStorage.removeItem(TOKEN_KEY);
      state.token = '';
    }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

/* The three states of a data view. An empty state says why it is empty and
   what fills it; an error state says what failed and offers the retry. */
function setListState(id, options) {
  const box = $(id);
  const opts = options || {};
  box.replaceChildren();
  box.classList.toggle('is-fault', opts.kind === 'error');
  box.classList.toggle('hidden', !opts.message);

  if (!opts.message) return;

  box.appendChild(makeEl('span', null, opts.message));
  if (opts.detail) box.appendChild(makeEl('span', 'state-detail', opts.detail));
  if (opts.action) {
    const btn = makeEl('button', 'btn btn-sm state-action', opts.action.label);
    btn.type = 'button';
    btn.addEventListener('click', opts.action.run);
    box.appendChild(btn);
  }
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------
function sessionFault(message, detail) {
  setListState('session-state', {
    kind: 'error',
    message,
    detail,
    action: { label: 'Try again', run: loadSessions },
  });
}

async function loadSessions() {
  setListState('session-state', { message: 'Loading conversations' });
  let sessions;
  try {
    sessions = await api('/api/chat/sessions');
  } catch (err) {
    $('session-list').replaceChildren();
    sessionFault('Could not load conversations.', err.message);
    return;
  }

  const list = $('session-list');
  list.replaceChildren();

  if (!sessions.length) {
    setListState('session-state', {
      message: 'No conversations yet.',
      detail: 'Ask a question and this becomes the first one.',
    });
    return;
  }

  setListState('session-state', {});
  sessions.forEach((session) => {
    const li = document.createElement('li');
    li.className = state.sessionId === session.session_id ? 'is-active' : '';
    li.dataset.kind = 'session';

    // The row's own control is a real button, so the list is keyboard
    // reachable. The action buttons sit beside it rather than inside it,
    // which keeps the markup valid.
    const open = makeEl('button', 'label');
    open.type = 'button';
    setText(open, session.title);            // user-controlled: textContent
    open.addEventListener('click', () => openSession(session.session_id));

    const actions = document.createElement('span');
    actions.className = 'row-actions';

    const rename = makeEl('button', null, 'Rename');
    rename.type = 'button';
    rename.addEventListener('click', (event) => {
      event.stopPropagation();
      beginRename(li, session);
    });

    const remove = makeEl('button', 'danger', 'Delete');
    remove.type = 'button';
    remove.addEventListener('click', (event) => {
      event.stopPropagation();
      armDelete(remove, () => {
        api(`/api/chat/sessions/${session.session_id}`, { method: 'DELETE' })
          .then(() => {
            // Returned, so a failure to replace the open conversation reaches
            // the catch below instead of becoming an unhandled rejection that
            // left the composer pointing at a deleted session.
            if (state.sessionId === session.session_id) return createSession();
            return loadSessions();
          })
          .catch((err) => {
            state.sessionId = null;
            $('messages').replaceChildren(emptyState());
            loadSessions().then(
              () => sessionFault('Could not delete that conversation.', err.message),
            );
          });
      });
    });

    actions.append(rename, remove);
    li.append(open, actions);
    list.appendChild(li);
  });
}

/* Two-step delete. The button arms, a second click commits, and it disarms
   itself after four seconds. window.confirm cannot be styled, is invisible to
   assistive tech on some platforms, and blocks the main thread. */
function armDelete(button, commit) {
  if (button.classList.contains('armed')) {
    clearTimeout(button._armTimer);
    button.classList.remove('armed');
    setText(button, button.dataset.rest);
    commit();
    return;
  }
  button.dataset.rest = button.textContent;
  button.classList.add('armed');
  setText(button, 'Confirm');
  button._armTimer = setTimeout(() => {
    button.classList.remove('armed');
    setText(button, button.dataset.rest);
  }, 4000);
}

/* Inline rename. Replaces window.prompt so the field can be styled, focused
   predictably, and committed with Enter or cancelled with Escape. */
function beginRename(li, session) {
  if (li.querySelector('.rename-input')) return;

  const input = document.createElement('input');
  input.className = 'rename-input';
  input.type = 'text';
  input.value = session.title;
  input.setAttribute('aria-label', 'Conversation title');
  // Matches the API's cap. A client limit looser than the server's one lets a
  // title through that the server rejects, and a rejection is the path that
  // used to leave this row stuck.
  input.maxLength = 200;
  li.replaceChildren(input);
  input.focus();
  input.select();

  let settled = false;
  const finish = (commit) => {
    if (settled) return;
    settled = true;
    const next = input.value.trim();
    if (!commit || !next || next === session.title) {
      loadSessions();
      return;
    }
    api(`/api/chat/sessions/${session.session_id}`, { method: 'PUT', body: { title: next } })
      .then(loadSessions)
      .catch((err) => loadSessions().then(
        () => sessionFault('Could not rename that conversation.', err.message),
      ));
  };

  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      finish(true);
    } else if (event.key === 'Escape') {
      event.preventDefault();
      finish(false);
    }
  });
  input.addEventListener('blur', () => finish(true));
}

async function createSession() {
  const session = await api('/api/chat/sessions', { method: 'POST', body: { title: 'New conversation' } });
  state.sessionId = session.session_id;
  setText($('chat-title'), session.title);
  $('messages').replaceChildren(emptyState());
  await loadSessions();
  return session.session_id;
}

function emptyState() {
  const wrap = document.createElement('div');
  wrap.className = 'empty';
  wrap.append(
    makeEl('h2', null, 'Ask about your documents'),
    makeEl('p', null,
      'Coddy answers only from the files you upload, and every answer names the page it came from.'),
  );
  const steps = makeEl('p', 'empty-steps');
  steps.append(
    makeEl('strong', null, 'PDF, DOCX, TXT, CSV or Markdown.'),
    document.createTextNode(' Nothing leaves this machine.'),
  );
  wrap.append(steps);
  return wrap;
}

async function openSession(sessionId) {
  // Switching conversations mid-stream would tear the column out from under
  // the tokens still arriving, leaving the answer to generate into a detached
  // node while the reader watches it vanish.
  if (state.busy) return;

  const messages = $('messages');
  setListState('session-state', { message: 'Opening conversation' });
  // A loading state rather than an empty column: the previous content is not
  // thrown away until the replacement has actually arrived.
  messages.replaceChildren(columnNotice('Opening the conversation'));

  let history;
  try {
    history = await api(`/api/chat/sessions/${sessionId}/messages`);
  } catch (err) {
    state.sessionId = null;
    messages.replaceChildren(emptyState());
    await loadSessions();
    return sessionFault('Could not open that conversation.', err.message);
  }

  state.sessionId = sessionId;
  setText($('chat-title'), history.length ? history[0].content.slice(0, 50) : 'Conversation');

  if (!history.length) {
    messages.replaceChildren(emptyState());
  } else {
    history.forEach((message) => appendMessage(message.sender, message.content, message.sources));
  }
  await loadSessions();
  scrollToBottom();
}

function columnNotice(text) {
  return makeEl('p', 'notice', text);
}

$('new-chat-btn').addEventListener('click', () => {
  if (state.busy) return;
  setNav(false);
  createSession().catch((err) => sessionFault('Could not start a conversation.', err.message));
});

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------
function documentFault(message, detail) {
  setListState('document-state', {
    kind: 'error',
    message,
    detail,
    action: { label: 'Try again', run: loadDocuments },
  });
}

async function loadDocuments() {
  setListState('document-state', { message: 'Loading documents' });
  let documents;
  try {
    documents = await api('/api/documents');
  } catch (err) {
    $('document-list').replaceChildren();
    documentFault('Could not load documents.', err.message);
    return;
  }

  const list = $('document-list');
  list.replaceChildren();
  setText($('doc-count'), `${documents.length} document${documents.length === 1 ? '' : 's'}`);

  if (!documents.length) {
    setListState('document-state', {
      message: 'No documents yet.',
      detail: 'Upload a file and it becomes the ground for every answer.',
      action: { label: 'Upload a file', run: () => $('file-input').click() },
    });
    return;
  }

  setListState('document-state', {});
  documents.forEach((doc) => {
    const li = document.createElement('li');
    li.dataset.kind = 'document';

    const label = makeEl('span', 'label');
    // setText, never innerHTML: the filename is user-controlled.
    setText(label, doc.name);
    label.title = doc.name;

    // The size sits in its own mono column rather than being appended to the
    // name. A catalogue reads as name plus figures, and the split is what
    // keeps this ledger from looking identical to the conversation ledger.
    const figures = makeEl('span', 'figures', `${Math.round(doc.size_bytes / 1024)} KB`);

    const actions = document.createElement('span');
    actions.className = 'row-actions';

    const reindex = makeEl('button', null, 'Reindex');
    reindex.type = 'button';
    reindex.addEventListener('click', (event) => {
      event.stopPropagation();
      reindexButton(doc, reindex);
    });

    const remove = makeEl('button', 'danger', 'Delete');
    remove.type = 'button';
    remove.addEventListener('click', (event) => {
      event.stopPropagation();
      armDelete(remove, () => {
        api(`/api/documents/${doc.id}`, { method: 'DELETE' })
          .then(loadDocuments)
          .catch((err) => loadDocuments().then(
            () => documentFault('Could not delete that document.', err.message),
          ));
      });
    });

    actions.append(reindex, remove);
    li.append(label, figures, actions);
    list.appendChild(li);
  });
}

async function reindexButton(doc, button) {
  const original = button.textContent;
  button.disabled = true;
  setText(button, 'Working');
  setText($('upload-hint'), `Reindexing ${doc.name}`);
  try {
    const result = await api(`/api/documents/${doc.id}/reindex`, { method: 'POST' });
    setText($('upload-hint'), `${doc.name} reindexed, ${result.chunks} chunks`);
  } catch (err) {
    documentFault('Could not reindex that document.', err.message);
  } finally {
    button.disabled = false;
    setText(button, original);
    setTimeout(() => setText($('upload-hint'), ''), 6000);
  }
}

$('upload-btn').addEventListener('click', () => $('file-input').click());

$('file-input').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const hint = $('upload-hint');
  const form = new FormData();
  form.append('file', file);
  setText(hint, `Indexing ${file.name}`);

  try {
    const result = await api('/api/documents', { method: 'POST', body: form });
    setText(hint, `${result.name} indexed, ${result.chunks} chunks`);
    await loadDocuments();
  } catch (err) {
    documentFault('Upload failed.', err.message);
  } finally {
    event.target.value = '';
    setTimeout(() => setText($('upload-hint'), ''), 8000);
  }
});

/* The status dot marks a real state the health probe reports, so it is polled
   only while the model is still loading. Once it is resident there is nothing
   left to watch and the timer stops.

   The poll is also bounded. With prewarm disabled the provider is never built
   until the first question, so `model_loaded` stays false indefinitely and an
   unbounded timer would request /api/health every six seconds for the life of
   the page while reporting a load that is not happening. After the cap it
   stops and offers a manual re-check instead. */
const STATUS_POLL_MS = 6000;
const STATUS_POLL_LIMIT = 40;

let statusTimer = null;
let statusAttempts = 0;

function scheduleStatus(delay) {
  clearTimeout(statusTimer);
  if (delay) statusTimer = setTimeout(loadStatus, delay);
}

function giveUpPolling() {
  $('status-retry').classList.remove('hidden');
}

$('status-retry').addEventListener('click', () => {
  $('status-retry').classList.add('hidden');
  statusAttempts = 0;
  loadStatus();
});

async function loadStatus() {
  const dot = $('model-dot');
  const label = $('model-status');
  try {
    const health = await api('/api/health');
    const detail = health.details || {};
    const model = detail.model_id || health.model || health.llm || 'model';

    if (health.model_loaded) {
      state.modelReady = true;
      dot.dataset.state = 'ready';
      const vram = detail.vram_total_gb ? `, ${detail.vram_total_gb}GB` : '';
      setText(label, `${model}${detail.quantization ? `, ${detail.quantization}` : ''}${vram}`);
      scheduleStatus(0);
      return;
    }

    state.modelReady = false;
    statusAttempts += 1;
    dot.dataset.state = 'warming';
    setText(label, `${model}, not loaded yet`);
    if (statusAttempts >= STATUS_POLL_LIMIT) {
      scheduleStatus(0);
      giveUpPolling();
      return;
    }
    scheduleStatus(STATUS_POLL_MS);
  } catch (_) {
    dot.dataset.state = 'unknown';
    setText(label, 'Model status unavailable');
    scheduleStatus(0);
  }
}

// ---------------------------------------------------------------------------
// Turns, citations and the dialog
// ---------------------------------------------------------------------------
function appendMessage(sender, content, sources) {
  const column = $('messages');
  // The empty state is replaced by the first turn, not left stacked above it.
  column.querySelector('.empty')?.remove();

  const turn = document.createElement('article');
  turn.className = sender === 'user' ? 'turn turn-user' : 'turn';

  const role = makeEl('div', 'turn-role', sender === 'user' ? 'You asked' : 'Coddy answered');
  const text = document.createElement('div');
  text.className = 'turn-content';
  // Both branches are safe: user text is escaped, model text is sanitized.
  if (sender === 'user') text.textContent = content;
  else text.innerHTML = renderMarkdown(content);

  turn.append(role, text);
  column.appendChild(turn);
  scrollToBottom();

  if (sources && sources.length) appendCitations(turn, sources);

  return { turn, content: text };
}

function appendCitations(turn, citations) {
  const box = document.createElement('div');
  box.className = 'sources';

  const label = makeEl('div', 'sources-label', `Sources, ${citations.length} passage${citations.length === 1 ? '' : 's'}`);
  const tabs = document.createElement('div');
  tabs.className = 'source-tabs';

  citations.forEach((citation) => {
    const tab = makeEl('button', 'source-tab', `${docLabel(citation.file_name)}, page ${citation.page_number}`);
    tab.type = 'button';
    tab.addEventListener('click', () => showCitation(citation));
    tabs.appendChild(tab);
  });

  box.append(label, tabs);
  turn.querySelector('.turn-content').after(box);
}

/* PDF, DOCX and the rest carry no extension in the chunk metadata, so the
   file name is shown as stored rather than reconstructed. */
function docLabel(fileName) {
  return fileName || 'Untitled file';
}

let lastDialogFocus = null;

function showCitation(citation) {
  setText($('citation-title'), docLabel(citation.file_name));
  setText($('citation-meta'), `Passage ${citation.position}, page ${citation.page_number}`);
  setText($('citation-body'), citation.excerpt || 'No text was stored for this passage.');
  lastDialogFocus = document.activeElement;
  $('citation-modal').classList.remove('hidden');
  $('citation-close').focus();
}

function closeDialog() {
  const modal = $('citation-modal');
  if (modal.classList.contains('hidden')) return;
  modal.classList.add('hidden');
  if (lastDialogFocus && typeof lastDialogFocus.focus === 'function') lastDialogFocus.focus();
}

$('citation-close').addEventListener('click', closeDialog);

$('citation-modal').addEventListener('mousedown', (event) => {
  if (event.target === $('citation-modal')) closeDialog();
});

// Tab stays inside the dialog while it is open.
$('citation-modal').addEventListener('keydown', (event) => {
  if (event.key !== 'Tab') return;
  const focusable = $('citation-modal').querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
  );
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
});

function scrollToBottom() {
  const box = $('messages');
  box.scrollTop = box.scrollHeight;
}

// ---------------------------------------------------------------------------
// Drawer (narrow screens)
// ---------------------------------------------------------------------------
let lastNavFocus = null;

function setNav(open) {
  const app = $('app');
  const toggle = $('nav-toggle');
  const navIsOpen = app.classList.contains('nav-open');
  if (open === navIsOpen) return;

  app.classList.toggle('nav-open', open);
  toggle.setAttribute('aria-expanded', String(open));
  toggle.setAttribute(
    'aria-label',
    open ? 'Hide conversations and documents' : 'Show conversations and documents',
  );

  if (open) {
    lastNavFocus = document.activeElement;
    $('new-chat-btn').focus();
  } else if (lastNavFocus && typeof lastNavFocus.focus === 'function') {
    lastNavFocus.focus();
  }
}

$('nav-toggle').addEventListener('click', () => setNav(!$('app').classList.contains('nav-open')));
$('nav-scrim').addEventListener('click', () => setNav(false));

document.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape') return;
  if (!$('citation-modal').classList.contains('hidden')) closeDialog();
  else if ($('app').classList.contains('nav-open')) setNav(false);
});

// ---------------------------------------------------------------------------
// Chat streaming
// ---------------------------------------------------------------------------
/* The rows are refused in openSession while a stream is running. Saying so is
   better than letting the click disappear without explanation. */
function setBusy(on) {
  state.busy = on;
  $('messages').setAttribute('aria-busy', String(on));
  document.querySelectorAll('#session-list button.label').forEach((btn) => {
    if (on) btn.setAttribute('aria-disabled', 'true');
    else btn.removeAttribute('aria-disabled');
  });
}

/* The streaming request cannot go through api(), because reading the body
   needs the raw Response. So it has to repeat the refresh-and-replay contract
   itself: without this, a token that expired mid-session left the composer
   permanently 401ing with no way back but a page reload. */
async function postChat(query) {
  const send = () => fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${state.token}` },
    body: JSON.stringify({ query, session_id: state.sessionId }),
  });

  const response = await send();
  if (response.status !== 401) return response;

  await localSession();
  return send();
}

$('composer').addEventListener('submit', sendMessage);

async function sendMessage(event) {
  event.preventDefault();
  if (state.busy) return;

  const input = $('query');
  const query = input.value.trim();
  if (!query) return;
  if (!state.sessionId) {
    sessionFault('No conversation is open.', 'Start a new one from the sidebar.');
    return;
  }

  input.value = '';
  input.style.height = 'auto';
  appendMessage('user', query);

  setBusy(true);
  $('send-btn').disabled = true;
  setText($('chat-title'), query.slice(0, 50));

  const card = appendMessage('assistant', '');
  const content = card.content;
  card.turn.setAttribute('aria-busy', 'true');
  // Say which wait this is. Loading the weights is a one-time cost and looks
  // like a hang if it is not named; the text is the whole feedback, since the
  // tokens arriving are the only motion this screen has.
  content.replaceChildren(makeEl(
    'p',
    'working',
    state.modelReady
      ? 'Searching your documents'
      : 'Loading the model, this only happens on the first question',
  ));

  let accumulated = '';
  let citations = null;
  // The handle is kept so it can be cancelled. A frame left pending when the
  // stream ends fires *after* the final render below and overwrites the
  // sanitised markdown with raw text.
  let frame = null;

  const cancelPendingFrame = () => {
    if (frame !== null) {
      cancelAnimationFrame(frame);
      frame = null;
    }
  };

  try {
    const response = await postChat(query);

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
            // One textContent write per frame instead of a markdown parse per
            // token. The full parse happens once, below.
            if (frame === null) {
              frame = requestAnimationFrame(() => {
                frame = null;
                if (accumulated) content.textContent = accumulated;
                scrollToBottom();
              });
            }
          }
          if (event.citations) citations = event.citations;
        }
        if (streamError) break;
      }
    }

    // The stream is finished, so nothing may write to this node but the final
    // render. Without this the queued frame lands afterwards and strips the
    // answer back to plain text.
    cancelPendingFrame();

    if (streamError) {
      content.replaceChildren(makeEl('p', 'fault', streamError));
    } else if (!accumulated) {
      content.replaceChildren(makeEl('p', 'muted', 'The model returned an empty answer.'));
    } else {
      content.innerHTML = renderMarkdown(accumulated);
    }

    if (citations && citations.length) appendCitations(card.turn, citations);
    await loadSessions();
  } catch (err) {
    cancelPendingFrame();
    content.replaceChildren(makeEl('p', 'fault', err.message));
  } finally {
    cancelPendingFrame();
    card.turn.setAttribute('aria-busy', 'false');
    setBusy(false);
    $('send-btn').disabled = false;
    input.focus();
  }
}

// Auto-grow, capped generously here so the CSS max-height governs. The
// textarea scrolls, so nothing is clipped when the cap is reached.
$('query').addEventListener('input', (event) => {
  const el = event.target;
  el.style.height = 'auto';
  el.style.height = `${el.scrollHeight}px`;
});

$('query').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    $('composer').requestSubmit();
  }
});

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------
/* The inline script in the head sets data-theme before first paint. This only
   has to keep the control's label in step with it, so a screen reader is told
   which theme it would switch to rather than which one is active. */
function currentTheme() {
  return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const next = theme === 'dark' ? 'light' : 'dark';
  $('theme-toggle').setAttribute('aria-label', `Switch to ${next} theme`);
}

$('theme-toggle').addEventListener('click', () => {
  const next = currentTheme() === 'dark' ? 'light' : 'dark';
  try {
    localStorage.setItem(THEME_KEY, next);
  } catch (_) { /* storage blocked; the choice lasts this page only */ }
  applyTheme(next);
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function showBootFault(message) {
  $('app').classList.add('hidden');
  $('boot').classList.remove('hidden');
  setText($('boot-note'), 'The local session could not be started');
  setText($('boot-error'), message);
  $('boot-retry').classList.remove('hidden');
}

/* Everything that happens once a session exists. Shared by the fresh-boot path
   and the stored-token path, which were otherwise the same five lines twice.

   The profile call is made for its side effect and the result deliberately
   not kept: a rejection here is how a stale token is detected. */
async function enterApp() {
  await api('/api/auth/profile');
  $('boot').classList.add('hidden');
  $('app').classList.remove('hidden');
  await Promise.all([loadSessions(), loadDocuments(), loadStatus()]);
  await createSession();
}

async function start() {
  $('boot-retry').classList.add('hidden');
  setText($('boot-error'), '');
  setText($('boot-note'), 'Starting the local session');
  try {
    await localSession();
    await enterApp();
  } catch (err) {
    showBootFault(err.message);
  }
}

$('boot-retry').addEventListener('click', start);

(async function init() {
  applyTheme(currentTheme());
  if (state.token) {
    // A stored token may still be valid. Try it before asking for a new one.
    try {
      await enterApp();
      return;
    } catch (_) {
      state.token = '';
      localStorage.removeItem(TOKEN_KEY);
    }
  }
  await start();
})();