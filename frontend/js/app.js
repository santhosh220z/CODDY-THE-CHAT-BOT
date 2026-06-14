// CODDY Chatbot Frontend Application Logic

// Application State
let token = localStorage.getItem('jwt_token') || '';
let currentUser = null;
let currentSessionId = '';
let documents = [];
let ttsEnabled = false;
let speechRecognition = null;
let isRecording = false;

// UI Elements
const authOverlay = document.getElementById('authOverlay');
const appView = document.getElementById('appView');
const loginForm = document.getElementById('loginForm');
const registerForm = document.getElementById('registerForm');
const showRegister = document.getElementById('showRegister');
const showLogin = document.getElementById('showLogin');

const userRoleBadge = document.getElementById('userRoleBadge');
const userEmailDisplay = document.getElementById('userEmailDisplay');
const modelStatusIndicator = document.getElementById('modelStatusIndicator');
const activeSessionTitle = document.getElementById('activeSessionTitle');

const sessionsList = document.getElementById('sessionsList');
const documentsList = document.getElementById('documentsList');
const uploadStatusPanel = document.getElementById('uploadStatusPanel');
const uploadStatusText = document.getElementById('uploadStatusText');
const sidebarUploadInput = document.getElementById('sidebarUploadInput');

const chatHistoryContainer = document.getElementById('chatHistoryContainer');
const welcomeLanding = document.getElementById('welcomeLanding');
const messagesWrapper = document.getElementById('messagesWrapper');
const queryInput = document.getElementById('queryInput');
const chatInputForm = document.getElementById('chatInputForm');
const typingIndicator = document.getElementById('typingIndicator');

const micBtn = document.getElementById('micBtn');
const micIcon = document.getElementById('micIcon');
const ttsToggleBtn = document.getElementById('ttsToggleBtn');
const ttsIcon = document.getElementById('ttsIcon');

// Modals
const settingsModal = document.getElementById('settingsModal');
const settingsBtn = document.getElementById('settingsBtn');
const settingsForm = document.getElementById('settingsForm');

const adminModal = document.getElementById('adminModal');
const adminBtn = document.getElementById('adminBtn');

const citationModal = document.getElementById('citationModal');
const citationDocTitle = document.getElementById('citationDocTitle');
const citationDocContent = document.getElementById('citationDocContent');
const citationPageIndicator = document.getElementById('citationPageIndicator');

// -------------------------------------------------------------
// INITIALIZATION
// -------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    initApp();
    setupEventListeners();
    setupSpeechRecognition();
});

function initApp() {
    if (token) {
        // Attempt to fetch profile
        fetchProfile();
    } else {
        showAuthScreen();
    }
}

function showAuthScreen() {
    authOverlay.classList.remove('hidden');
    appView.classList.add('hidden');
}

function hideAuthScreen() {
    authOverlay.classList.add('hidden');
    appView.classList.remove('hidden');
}

// -------------------------------------------------------------
// EVENT LISTENERS
// -------------------------------------------------------------
function setupEventListeners() {
    // Auth Forms Toggles
    showRegister.addEventListener('click', (e) => {
        e.preventDefault();
        loginForm.classList.add('hidden');
        registerForm.classList.remove('hidden');
    });

    showLogin.addEventListener('click', (e) => {
        e.preventDefault();
        registerForm.classList.add('hidden');
        loginForm.classList.remove('hidden');
    });

    // Form Submissions
    loginForm.addEventListener('submit', handleLogin);
    registerForm.addEventListener('submit', handleRegister);
    chatInputForm.addEventListener('submit', handleSendMessage);

    // Sidebar upload
    sidebarUploadInput.addEventListener('change', handleFileUpload);

    // New Session
    document.getElementById('newChatBtn').addEventListener('click', createNewSession);

    // Clear Sessions
    document.getElementById('clearHistoryBtn').addEventListener('click', handleClearSessions);

    // Sign Out
    document.getElementById('logoutBtn').addEventListener('click', handleLogout);

    // Textarea autosize
    queryInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
    
    // Shift+Enter handles newlines, single Enter submits form
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatInputForm.dispatchEvent(new Event('submit'));
        }
    });

    // Modals bindings
    settingsBtn.addEventListener('click', () => {
        openSettingsModal();
    });
    adminBtn.addEventListener('click', () => {
        openAdminModal();
    });

    // Close buttons for all modals
    document.querySelectorAll('.closeModalBtn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            settingsModal.classList.add('hidden');
            citationModal.classList.add('hidden');
            adminModal.classList.add('hidden');
        });
    });

    // Save Settings Form
    settingsForm.addEventListener('submit', handleSaveSettings);

    // TTS Toggle
    ttsToggleBtn.addEventListener('click', () => {
        ttsEnabled = !ttsEnabled;
        if (ttsEnabled) {
            ttsIcon.className = "fa-solid fa-volume-high text-base text-accentPurp";
            ttsToggleBtn.title = "Read Responses Aloud (Enabled)";
        } else {
            ttsIcon.className = "fa-solid fa-volume-xmark text-base";
            ttsToggleBtn.title = "Read Responses Aloud";
            window.speechSynthesis.cancel();
        }
    });
}

// -------------------------------------------------------------
// AUTHENTICATION OPERATIONS
// -------------------------------------------------------------
async function handleLogin(e) {
    e.preventDefault();
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;

    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Invalid credentials');
        }

        const data = await response.json();
        token = data.access_token;
        localStorage.setItem('jwt_token', token);
        
        await fetchProfile();
        
    } catch (err) {
        alert('Login failed: ' + err.message);
    }
}

async function handleRegister(e) {
    e.preventDefault();
    const email = document.getElementById('registerEmail').value;
    const password = document.getElementById('registerPassword').value;
    const role = document.getElementById('registerRole').value;

    try {
        const response = await fetch('/api/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password, role })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Registration failed');
        }

        alert('Registration successful! Please sign in.');
        registerForm.classList.add('hidden');
        loginForm.classList.remove('hidden');
        
    } catch (err) {
        alert('Registration failed: ' + err.message);
    }
}

async function fetchProfile() {
    try {
        const response = await fetch('/api/auth/profile', {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (!response.ok) {
            throw new Error('Token expired or invalid');
        }

        currentUser = await response.json();
        
        // Populate view metadata
        userEmailDisplay.textContent = currentUser.email;
        userRoleBadge.textContent = currentUser.role;
        
        if (currentUser.role === 'admin') {
            adminBtn.classList.remove('hidden');
        } else {
            adminBtn.classList.add('hidden');
        }

        hideAuthScreen();
        
        // Initial data loading
        await loadSessions();
        await loadDocuments();
        await fetchSystemStats(); // populate settings default variables
        
    } catch (err) {
        console.error(err);
        handleLogout();
    }
}

function handleLogout() {
    token = '';
    currentUser = null;
    currentSessionId = '';
    localStorage.removeItem('jwt_token');
    showAuthScreen();
    window.speechSynthesis.cancel();
}

// -------------------------------------------------------------
// CHAT SESSION OPERATIONS
// -------------------------------------------------------------
async function loadSessions() {
    try {
        const response = await fetch('/api/chat/sessions', {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        const data = await response.json();
        
        sessionsList.innerHTML = '';
        if (data.length === 0) {
            sessionsList.innerHTML = '<div class="text-xs text-gray-500 italic py-2 text-center">No active chats</div>';
            return;
        }

        data.forEach(s => {
            const li = document.createElement('li');
            li.className = `group flex items-center justify-between px-3 py-2 rounded-lg text-sm transition cursor-pointer ${s.session_id === currentSessionId ? 'bg-accentPurp/20 text-accentPurp border border-accentPurp/30 font-medium' : 'text-gray-300 hover:bg-white/5'}`;
            li.setAttribute('data-id', s.session_id);
            
            li.innerHTML = `
                <div class="flex items-center space-x-2 truncate w-48 select-none">
                    <i class="fa-regular fa-message text-xs opacity-70"></i>
                    <span class="truncate pr-1 session-title">${s.title}</span>
                </div>
                <div class="flex items-center space-x-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button class="rename-session hover:text-accentCyan" title="Rename"><i class="fa-solid fa-pen text-[10px]"></i></button>
                    <button class="delete-session hover:text-red-400" title="Delete"><i class="fa-solid fa-xmark text-[10px]"></i></button>
                </div>
            `;
            
            // Selection event
            li.querySelector('div:first-child').addEventListener('click', () => {
                selectSession(s.session_id);
            });
            
            // Rename event
            li.querySelector('.rename-session').addEventListener('click', (e) => {
                e.stopPropagation();
                const newTitle = prompt("Enter new title for conversation:", s.title);
                if (newTitle && newTitle.trim()) {
                    renameSession(s.session_id, newTitle.trim());
                }
            });
            
            // Delete event
            li.querySelector('.delete-session').addEventListener('click', (e) => {
                e.stopPropagation();
                if (confirm(`Delete conversation "${s.title}"?`)) {
                    deleteSession(s.session_id);
                }
            });

            sessionsList.appendChild(li);
        });
    } catch (err) {
        console.error("Failed to load sessions:", err);
    }
}

async function selectSession(sessionId) {
    currentSessionId = sessionId;
    // Highlight sidebar item
    document.querySelectorAll('#sessionsList li').forEach(li => {
        if (li.getAttribute('data-id') === sessionId) {
            li.className = 'group flex items-center justify-between px-3 py-2 rounded-lg text-sm bg-accentPurp/20 text-accentPurp border border-accentPurp/30 font-medium cursor-pointer';
        } else {
            li.className = 'group flex items-center justify-between px-3 py-2 rounded-lg text-sm text-gray-300 hover:bg-white/5 cursor-pointer';
        }
    });

    // Set title and load messages
    const session = document.querySelector(`#sessionsList li[data-id="${sessionId}"] .session-title`);
    if (session) {
        activeSessionTitle.textContent = session.textContent;
    }
    
    welcomeLanding.classList.add('hidden');
    messagesWrapper.innerHTML = '';
    
    await loadSessionHistory(sessionId);
}

async function createNewSession() {
    try {
        const response = await fetch('/api/chat/session', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ title: "New Conversation" })
        });
        const data = await response.json();
        currentSessionId = data.session_id;
        
        await loadSessions();
        selectSession(data.session_id);
    } catch (err) {
        console.error("Failed to create session:", err);
    }
}

async function renameSession(sessionId, newTitle) {
    try {
        await fetch(`/api/chat/sessions/${sessionId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ title: newTitle })
        });
        await loadSessions();
        if (sessionId === currentSessionId) {
            activeSessionTitle.textContent = newTitle;
        }
    } catch (err) {
        console.error("Failed to rename session:", err);
    }
}

async function deleteSession(sessionId) {
    try {
        await fetch(`/api/chat/sessions/${sessionId}`, {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${token}` }
        });
        
        if (sessionId === currentSessionId) {
            currentSessionId = '';
            activeSessionTitle.textContent = "Select or Create a Chat";
            welcomeLanding.classList.remove('hidden');
            messagesWrapper.innerHTML = '';
        }
        await loadSessions();
    } catch (err) {
        console.error("Failed to delete session:", err);
    }
}

async function handleClearSessions() {
    if (confirm("Are you sure you want to clear all chat sessions? This action is permanent.")) {
        try {
            await fetch('/api/chat/sessions', {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${token}` }
            });
            currentSessionId = '';
            activeSessionTitle.textContent = "Select or Create a Chat";
            welcomeLanding.classList.remove('hidden');
            messagesWrapper.innerHTML = '';
            await loadSessions();
        } catch (err) {
            console.error("Failed to clear sessions:", err);
        }
    }
}

async function loadSessionHistory(sessionId) {
    try {
        const response = await fetch(`/api/chat/history/${sessionId}`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        const messages = await response.json();
        
        messagesWrapper.innerHTML = '';
        messages.forEach(m => {
            appendMessage(m.sender, m.content, m.sources);
        });
        scrollToBottom();
    } catch (err) {
        console.error("Failed to load history:", err);
    }
}

// Helper to fill suggested prompt card click into textbox
window.fillSuggestion = function(text) {
    queryInput.value = text;
    queryInput.style.height = 'auto';
    queryInput.style.height = (queryInput.scrollHeight) + 'px';
    queryInput.focus();
};

// -------------------------------------------------------------
// CHAT MESSAGE OPERATIONS (STREAMING RAG INTERFACES)
// -------------------------------------------------------------
async function handleSendMessage(e) {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    // Create session on fly if not exists
    if (!currentSessionId) {
        await createNewSession();
    }

    // Append user message immediately
    appendMessage('user', query);
    queryInput.value = '';
    queryInput.style.height = 'auto';
    scrollToBottom();

    // Prepare streaming response block
    typingIndicator.classList.remove('hidden');
    scrollToBottom();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                query: query,
                session_id: currentSessionId
            })
        });

        typingIndicator.classList.add('hidden');

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to stream response');
        }

        // Initialize streaming UI elements
        const assistantMsgEl = appendMessage('assistant', '');
        const textContentEl = assistantMsgEl.querySelector('.msg-content');
        
        // Read stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let done = false;
        let accumulatedText = "";
        let citations = [];

        while (!done) {
            const { value, done: streamDone } = await reader.read();
            done = streamDone;
            if (value) {
                const chunk = decoder.decode(value);
                // SSE chunks are separated by double newline. Split them
                const lines = chunk.split('\n\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const payloadStr = line.substring(6);
                        if (!payloadStr) continue;
                        
                        try {
                            const payload = JSON.parse(payloadStr);
                            
                            if (payload.error) {
                                textContentEl.innerHTML = `<span class="text-red-400">Error: ${payload.error}</span>`;
                                break;
                            }
                            
                            if (payload.token) {
                                accumulatedText += payload.token;
                                textContentEl.innerHTML = formatMarkdown(accumulatedText);
                                scrollToBottom();
                            }
                            
                            if (payload.citations) {
                                citations = payload.citations;
                            }
                        } catch (err) {
                            console.error("SSE parse error", err, "Line:", line);
                        }
                    }
                }
            }
        }

        // Apply citations to assistant element at completion
        if (citations && citations.length > 0) {
            appendCitationsList(assistantMsgEl, citations);
        }
        
        // Refresh session titles sidebar to update from default title if needed
        await loadSessions();

        // Read out response if voice output (TTS) is enabled
        if (ttsEnabled && accumulatedText) {
            // Strip out <think> tags from voice read out
            const voiceText = accumulatedText.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
            speakResponse(voiceText);
        }

    } catch (err) {
        typingIndicator.classList.add('hidden');
        appendMessage('assistant', `Failed to retrieve details. Make sure Ollama or Hugging Face provider is running. (Details: ${err.message})`);
        scrollToBottom();
    }
}

// Simple Custom Markdown/Think formatter
function formatMarkdown(text) {
    if (!text) return "";
    
    let html = text;

    // Formatter for DeepSeek reasoning <think> ... </think> block
    html = html.replace(/<think>([\s\S]*?)<\/think>/g, (match, p1) => {
        return `
            <div class="think-block rounded-xl border border-accentCyan/15 mb-3 overflow-hidden shadow-sm">
                <button onclick="this.nextElementSibling.classList.toggle('hidden'); this.querySelector('i').classList.toggle('fa-chevron-down'); this.querySelector('i').classList.toggle('fa-chevron-right')" class="w-full flex items-center justify-between px-3 py-1.5 bg-accentCyan/5 text-xs text-accentCyan hover:bg-accentCyan/10 transition">
                    <span class="flex items-center space-x-1.5 font-medium"><i class="fa-solid fa-brain text-[10px] animate-pulse"></i> <span>Reasoning Thought Process</span></span>
                    <i class="fa-solid fa-chevron-down text-[10px] opacity-70"></i>
                </button>
                <div class="p-3 text-xs text-gray-400 font-light italic bg-darkCard/30 leading-relaxed whitespace-pre-wrap">${p1.trim()}</div>
            </div>
        `;
    });

    // Fallback: If thinking block is opened but not closed yet during streaming
    if (html.includes('<think>')) {
        const parts = html.split('<think>');
        const thinkingContent = parts[1] || "";
        html = parts[0] + `
            <div class="think-block rounded-xl border border-accentCyan/15 mb-3 overflow-hidden shadow-sm">
                <div class="w-full flex items-center px-3 py-1.5 bg-accentCyan/5 text-xs text-accentCyan">
                    <span class="flex items-center space-x-1.5 font-medium"><i class="fa-solid fa-circle-notch animate-spin text-[10px]"></i> <span>Thinking...</span></span>
                </div>
                <div class="p-3 text-xs text-gray-400 font-light italic bg-darkCard/30 leading-relaxed whitespace-pre-wrap">${thinkingContent.trim()}</div>
            </div>
        `;
    }

    // Paragraph format mapping (preserves block code structure)
    // Code blocks formatter (e.g. ```python ... ```)
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
        return `
            <div class="relative my-3 group border border-white/5 rounded-xl overflow-hidden shadow-md">
                <div class="flex items-center justify-between px-4 py-2 bg-darkCard/80 border-b border-white/5 text-xs text-gray-400">
                    <span class="uppercase font-semibold tracking-wider text-[10px]">${lang || 'code'}</span>
                    <button onclick="copyCode(this)" class="hover:text-accentCyan flex items-center space-x-1 font-medium">
                        <i class="fa-regular fa-clipboard"></i>
                        <span>Copy</span>
                    </button>
                </div>
                <pre class="p-4 bg-[#0d0e15] overflow-x-auto text-xs text-gray-200 leading-normal font-mono select-text"><code>${escapeHtml(code.trim())}</code></pre>
            </div>
        `;
    });

    // Inline code (e.g. `code`)
    html = html.replace(/`([^`\n]+)`/g, '<code class="bg-white/10 px-1.5 py-0.5 rounded text-accentCyan text-xs font-mono font-medium">$1</code>');

    // Bold text (e.g. **bold**)
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong class="font-bold text-white">$1</strong>');

    // Lists items
    html = html.replace(/^\s*[-*]\s+(.+)$/gm, '<li class="ml-4 list-disc text-sm text-gray-200 py-0.5">$1</li>');
    html = html.replace(/^\s*\d+\.\s+(.+)$/gm, '<li class="ml-4 list-decimal text-sm text-gray-200 py-0.5">$1</li>');

    // Split text paragraphs by double newlines outside of structured blocks and apply spacing
    // Since simple html tags work, we return final string.
    return html;
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, function(m) { return map[m]; });
}

window.copyCode = function(button) {
    const codeContainer = button.closest('.relative').querySelector('pre code');
    navigator.clipboard.writeText(codeContainer.textContent).then(() => {
        const textSpan = button.querySelector('span');
        const icon = button.querySelector('i');
        
        textSpan.textContent = "Copied!";
        icon.className = "fa-solid fa-check text-emerald-400";
        
        setTimeout(() => {
            textSpan.textContent = "Copy";
            icon.className = "fa-regular fa-clipboard";
        }, 2000);
    });
};

function appendMessage(sender, content, sources = null) {
    const msgCard = document.createElement('div');
    msgCard.className = `flex items-start space-x-4 max-w-4xl ${sender === 'user' ? 'self-end bg-accentPurp/10 border border-accentPurp/15 ml-12 rounded-2xl rounded-tr-sm p-4' : 'self-start w-full border border-white/5 bg-darkCard/30 rounded-2xl rounded-tl-sm p-4'}`;
    
    // Icon
    const avatar = document.createElement('div');
    avatar.className = `w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${sender === 'user' ? 'bg-accentPurp/20 text-accentPurp' : 'bg-accentCyan/20 text-accentCyan'}`;
    avatar.innerHTML = sender === 'user' ? '<i class="fa-solid fa-user text-sm"></i>' : '<i class="fa-solid fa-robot text-sm"></i>';
    
    const body = document.createElement('div');
    body.className = "flex-grow overflow-hidden";
    
    // Username header
    const header = document.createElement('div');
    header.className = "flex items-center space-x-2 text-xs font-semibold text-gray-400 mb-1.5";
    header.innerHTML = `<span>${sender === 'user' ? 'You' : 'CODDY'}</span><span class="text-[9px] opacity-50">${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>`;
    
    const textDiv = document.createElement('div');
    textDiv.className = "msg-content prose text-sm font-light leading-relaxed";
    textDiv.innerHTML = sender === 'user' ? escapeHtml(content) : formatMarkdown(content);
    
    body.appendChild(header);
    body.appendChild(textDiv);
    
    msgCard.appendChild(avatar);
    msgCard.appendChild(body);
    
    // Add actions (copy/regenerate) to assistant response
    if (sender === 'assistant') {
        const actions = document.createElement('div');
        actions.className = "flex items-center space-x-3 mt-3 pt-2.5 border-t border-white/5 text-xs text-gray-500";
        actions.innerHTML = `
            <button onclick="copyMsgText(this)" class="hover:text-accentCyan flex items-center space-x-1"><i class="fa-regular fa-copy"></i> <span>Copy response</span></button>
            <button onclick="regenerateLastMsg()" class="hover:text-accentPurp flex items-center space-x-1"><i class="fa-solid fa-arrows-rotate"></i> <span>Regenerate</span></button>
        `;
        body.appendChild(actions);
    }
    
    messagesWrapper.appendChild(msgCard);
    
    if (sources && sources.length > 0) {
        appendCitationsList(msgCard, sources);
    }
    
    return msgCard;
}

window.copyMsgText = function(button) {
    const content = button.closest('.msg-content, div').querySelector('.msg-content').textContent;
    navigator.clipboard.writeText(content).then(() => {
        const textSpan = button.querySelector('span');
        textSpan.textContent = "Copied!";
        setTimeout(() => {
            textSpan.textContent = "Copy response";
        }, 2000);
    });
};

window.regenerateLastMsg = function() {
    // Find last user message, populate to textbox, and trigger send
    const userMsgs = Array.from(document.querySelectorAll('.self-end'));
    if (userMsgs.length > 0) {
        const lastUserText = userMsgs[userMsgs.length - 1].querySelector('.msg-content').textContent;
        queryInput.value = lastUserText;
        chatInputForm.dispatchEvent(new Event('submit'));
    }
};

function appendCitationsList(messageCardEl, citations) {
    const listContainer = document.createElement('div');
    listContainer.className = "mt-4 pt-3 border-t border-white/5 text-xs";
    
    let citationsHtml = `<div class="font-semibold text-accentCyan mb-2 flex items-center space-x-1.5"><i class="fa-solid fa-circle-nodes text-[10px]"></i> <span>Sources Utilized:</span></div><div class="flex flex-wrap gap-2">`;
    
    citations.forEach((cit, index) => {
        // Embed the details inside window variables or data attributes
        const citId = `citation_${Date.now()}_${index}`;
        window[citId] = cit; // cache it globally to pull in popup modal
        
        citationsHtml += `
            <button onclick="showCitationPreview('${citId}')" class="px-2.5 py-1 rounded bg-white/5 hover:bg-accentCyan/10 border border-white/10 hover:border-accentCyan/20 text-[11px] text-gray-300 hover:text-accentCyan transition duration-200 flex items-center space-x-1 max-w-[220px] truncate">
                <i class="fa-solid fa-file-pdf text-[10px] opacity-75"></i>
                <span class="truncate font-medium">${cit.file_name}</span>
                <span class="text-[9px] text-gray-500">(Page ${cit.page_number})</span>
            </button>
        `;
    });
    
    citationsHtml += `</div>`;
    listContainer.innerHTML = citationsHtml;
    
    // Insert before actions bar
    const bodyEl = messageCardEl.querySelector('div:nth-child(2)');
    const actionsBar = bodyEl.querySelector('div:last-child');
    if (actionsBar && actionsBar !== bodyEl.querySelector('.msg-content')) {
        bodyEl.insertBefore(listContainer, actionsBar);
    } else {
        bodyEl.appendChild(listContainer);
    }
}

window.showCitationPreview = function(cacheId) {
    const data = window[cacheId];
    if (!data) return;
    
    citationDocTitle.textContent = data.file_name;
    citationDocContent.textContent = data.text_preview;
    citationPageIndicator.textContent = `Document Segment Context • Page ${data.page_number}`;
    
    citationModal.classList.remove('hidden');
};

function scrollToBottom() {
    chatHistoryContainer.scrollTop = chatHistoryContainer.scrollHeight;
}

// -------------------------------------------------------------
// KNOWLEDGE BASE / DOCUMENT ACTIONS
// -------------------------------------------------------------
async function loadDocuments() {
    try {
        const response = await fetch('/api/documents', {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        documents = await response.json();
        
        documentsList.innerHTML = '';
        if (documents.length === 0) {
            documentsList.innerHTML = '<div class="text-xs text-gray-500 italic py-2 text-center">No documents uploaded</div>';
            return;
        }

        documents.forEach(doc => {
            const sizeKB = (doc.size / 1024).toFixed(1);
            const li = document.createElement('li');
            li.className = "group flex items-center justify-between px-3 py-2 bg-white/5 border border-white/5 hover:border-white/10 rounded-xl text-xs transition duration-200";
            
            li.innerHTML = `
                <div class="flex items-center space-x-2 truncate w-44">
                    <i class="fa-solid fa-file-contract text-accentCyan text-xs flex-shrink-0"></i>
                    <div class="truncate">
                        <p class="font-medium text-gray-200 truncate" title="${doc.name}">${doc.name}</p>
                        <p class="text-[9px] text-gray-500 font-light">${sizeKB} KB • Type: ${doc.type.toUpperCase()}</p>
                    </div>
                </div>
                <div class="flex items-center space-x-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button class="reindex-doc hover:text-accentPurp" title="Re-index"><i class="fa-solid fa-arrows-rotate text-[10px]"></i></button>
                    <button class="delete-doc hover:text-red-400" title="Delete"><i class="fa-solid fa-trash text-[10px]"></i></button>
                </div>
            `;
            
            // Re-index binder
            li.querySelector('.reindex-doc').addEventListener('click', (e) => {
                e.stopPropagation();
                if (confirm(`Re-index document "${doc.name}"? This recreates embedding vectors.`)) {
                    reindexDocument(doc.id);
                }
            });
            
            // Delete binder
            li.querySelector('.delete-doc').addEventListener('click', (e) => {
                e.stopPropagation();
                if (confirm(`Delete document "${doc.name}" from knowledge base?`)) {
                    deleteDocument(doc.id);
                }
            });

            documentsList.appendChild(li);
        });
    } catch (err) {
        console.error("Failed to load documents:", err);
    }
}

async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Toggle progress panel
    uploadStatusPanel.classList.remove('hidden');
    uploadStatusText.textContent = `Indexing ${file.name}...`;
    
    try {
        const response = await fetch('/api/documents/upload', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${token}` },
            body: formData
        });
        
        uploadStatusPanel.classList.add('hidden');
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Upload and indexing failed.');
        }
        
        const res = await response.json();
        alert(`Successfully indexed "${res.filename}" into ${res.chunks_count} chunks.`);
        
        await loadDocuments();
    } catch (err) {
        uploadStatusPanel.classList.add('hidden');
        alert('Upload failed: ' + err.message);
    } finally {
        sidebarUploadInput.value = ''; // clear input
    }
}

async function deleteDocument(docId) {
    try {
        const response = await fetch(`/api/documents/${docId}`, {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${token}` }
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Deletion failed');
        }
        
        alert("Document deleted successfully.");
        await loadDocuments();
    } catch (err) {
        alert("Delete failed: " + err.message);
    }
}

async function reindexDocument(docId) {
    try {
        const response = await fetch(`/api/documents/${docId}/reindex`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${token}` }
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Reindexing failed');
        }
        
        const res = await response.json();
        alert(`Document re-indexed into ${res.chunks_count} chunks.`);
        await loadDocuments();
    } catch (err) {
        alert("Re-index failed: " + err.message);
    }
}

// -------------------------------------------------------------
// SETTINGS CONFIGURATIONS OPERATIONS
// -------------------------------------------------------------
let cachedStats = null;

async function fetchSystemStats() {
    try {
        const response = await fetch('/api/admin/stats', {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        if (response.ok) {
            cachedStats = await response.json();
            
            // Render settings form defaults from stats
            document.getElementById('settingLlmProvider').value = cachedStats.configuration.active_llm.toLowerCase().split(' - ')[0];
            document.getElementById('settingLlmModel').value = cachedStats.configuration.active_llm.split(' - ')[1];
            document.getElementById('settingEmbeddingModel').value = cachedStats.configuration.active_embedding;
            document.getElementById('settingVectorStore').value = cachedStats.configuration.vector_store.toLowerCase();
            
            // Update models UI text
            modelStatusIndicator.innerHTML = `
                <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                <span>Active Model • ${cachedStats.configuration.active_llm}</span>
            `;
        }
    } catch (err) {
        console.error("Failed to load initial system configurations:", err);
    }
}

function openSettingsModal() {
    // Populate form variables from settings cached state or standard inputs
    document.getElementById('settingTopK').value = localStorage.getItem('top_k') || '5';
    document.getElementById('settingHybridSearch').checked = (localStorage.getItem('hybrid_search') !== 'false');
    document.getElementById('settingBm25Weight').value = localStorage.getItem('bm25_weight') || '0.3';
    document.getElementById('settingChunkSize').value = localStorage.getItem('chunk_size') || '1000';
    document.getElementById('settingChunkOverlap').value = localStorage.getItem('chunk_overlap') || '200';
    
    // Toggle Hugging Face credential boxes depending on selected LLM
    const provider = document.getElementById('settingLlmProvider');
    const hfWrap = document.getElementById('hfCredentialsWrapper');
    
    const handleLlmChange = () => {
        if (provider.value === 'huggingface') {
            hfWrap.classList.remove('hidden');
        } else {
            hfWrap.classList.add('hidden');
        }
    };
    provider.addEventListener('change', handleLlmChange);
    handleLlmChange();

    settingsModal.classList.remove('hidden');
}

async function handleSaveSettings(e) {
    e.preventDefault();
    
    const topK = document.getElementById('settingTopK').value;
    const hybrid = document.getElementById('settingHybridSearch').checked;
    const weight = document.getElementById('settingBm25Weight').value;
    const size = document.getElementById('settingChunkSize').value;
    const overlap = document.getElementById('settingChunkOverlap').value;
    
    // Store configurations locally
    localStorage.setItem('top_k', topK);
    localStorage.setItem('hybrid_search', hybrid);
    localStorage.setItem('bm25_weight', weight);
    localStorage.setItem('chunk_size', size);
    localStorage.setItem('chunk_overlap', overlap);
    
    // Save provider choices
    const provider = document.getElementById('settingLlmProvider').value;
    const model = document.getElementById('settingLlmModel').value;
    const embedding = document.getElementById('settingEmbeddingModel').value;
    const store = document.getElementById('settingVectorStore').value;
    const hfToken = document.getElementById('settingHfToken').value;

    alert("Settings saved successfully. Changes will apply to new documents and chats.");
    settingsModal.classList.add('hidden');
    
    // Update local indicators
    modelStatusIndicator.innerHTML = `
        <span class="w-1.5 h-1.5 rounded-full bg-accentCyan"></span>
        <span>Active Model • ${provider.toUpperCase()} - ${model}</span>
    `;
}

// -------------------------------------------------------------
// ADMINISTRATIVE OPERATIONS
// -------------------------------------------------------------
async function openAdminModal() {
    try {
        const response = await fetch('/api/admin/stats', {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to retrieve metrics');
        }
        
        const data = await response.json();
        
        // Populate items
        document.getElementById('adminStatUsers').textContent = data.stats.total_users;
        document.getElementById('adminStatDocs').textContent = data.stats.total_documents;
        document.getElementById('adminStatSessions').textContent = data.stats.chat_sessions;
        document.getElementById('adminStatMessages').textContent = data.stats.total_messages;
        
        document.getElementById('adminStorageUploads').textContent = data.storage.uploads_mb + " MB";
        document.getElementById('adminStorageVector').textContent = data.storage.vector_db_mb + " MB";
        document.getElementById('adminStorageTotal').textContent = data.storage.total_used_mb + " MB";
        
        // Compute bar percentages relative to maximum 100MB representation
        const maxRep = 100.0;
        const uploadPct = Math.min((data.storage.uploads_mb / maxRep) * 100, 100);
        const vectorPct = Math.min((data.storage.vector_db_mb / maxRep) * 100, 100);
        
        document.getElementById('adminStorageUploadsBar').style.width = uploadPct + "%";
        document.getElementById('adminStorageVectorBar').style.width = vectorPct + "%";
        
        document.getElementById('adminConfStore').textContent = data.configuration.vector_store;
        document.getElementById('adminConfEmbedding').textContent = data.configuration.active_embedding;
        document.getElementById('adminConfLlm').textContent = data.configuration.active_llm;
        
        const healthEl = document.getElementById('adminSystemHealth');
        if (data.system_health === 'Healthy') {
            healthEl.innerHTML = '<i class="fa-solid fa-circle text-[8px] text-emerald-400 animate-pulse"></i> <span>Healthy</span>';
            healthEl.className = 'font-bold text-emerald-400 flex items-center space-x-1';
        } else {
            healthEl.innerHTML = '<i class="fa-solid fa-circle text-[8px] text-red-500 animate-pulse"></i> <span>Unhealthy</span>';
            healthEl.className = 'font-bold text-red-500 flex items-center space-x-1';
        }

        adminModal.classList.remove('hidden');
    } catch (err) {
        alert("Failed to load administration stats: " + err.message);
    }
}

// -------------------------------------------------------------
// VOICE API INTEGRATION
// -------------------------------------------------------------
function setupSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
        speechRecognition = new SpeechRecognition();
        speechRecognition.continuous = false;
        speechRecognition.interimResults = false;
        speechRecognition.lang = 'en-US';
        
        speechRecognition.onstart = () => {
            isRecording = true;
            micIcon.className = "fa-solid fa-microphone-lines text-base text-accentCyan animate-pulse";
            micBtn.title = "Listening...";
        };
        
        speechRecognition.onend = () => {
            isRecording = false;
            micIcon.className = "fa-solid fa-microphone text-base";
            micBtn.title = "Voice Input";
        };
        
        speechRecognition.onerror = (e) => {
            console.error("Speech recognition error:", e);
            isRecording = false;
            micIcon.className = "fa-solid fa-microphone text-base";
        };
        
        speechRecognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            queryInput.value = transcript;
            queryInput.style.height = 'auto';
            queryInput.style.height = (queryInput.scrollHeight) + 'px';
            queryInput.focus();
        };
        
        micBtn.addEventListener('click', () => {
            if (isRecording) {
                speechRecognition.stop();
            } else {
                speechRecognition.start();
            }
        });
    } else {
        micBtn.style.display = 'none'; // hide microphone if browser doesn't support Web Speech API
        console.warn("Browser does not support SpeechRecognition.");
    }
}

function speakResponse(text) {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel(); // cancel existing readings
        
        // Clean markdown remnants for reading
        const cleanedText = text
            .replace(/```[\s\S]*?```/g, "") // skip code blocks
            .replace(/`([^`]+)`/g, "$1")
            .replace(/\*\*([^*]+)\*\*/g, "$1")
            .replace(/[*#_-]/g, "");
            
        const utterance = new SpeechSynthesisUtterance(cleanedText);
        utterance.lang = 'en-US';
        utterance.rate = 1.0;
        
        window.speechSynthesis.speak(utterance);
    }
}
