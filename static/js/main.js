// ============================================================
//  INDOOR OBJECT FINDER — main.js  (voice utility)
// ============================================================

function speakMessage(text) {
  if (!('speechSynthesis' in window)) return;
  const u = new SpeechSynthesisUtterance(text);
  const voices = window.speechSynthesis.getVoices();
  const preferred = voices.find(v => /female|zira|susan|aria|samantha|google us english/i.test(v.name))
    || voices.find(v => /english/i.test(v.name)) || voices[0];
  if (preferred) u.voice = preferred;
  u.rate = 1; u.pitch = 1.1; u.volume = 1;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(u);
}

window.speakMessage = speakMessage;

if ('speechSynthesis' in window) {
  window.speechSynthesis.getVoices();
  window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
}

// Voice button (search.html uses this)
const voiceBtn = document.getElementById('voice-btn');
if (voiceBtn) {
  voiceBtn.addEventListener('click', () => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
      alert('Voice input not supported. Please use Chrome.'); return;
    }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const rec = new SR();
    rec.lang = 'en-US'; rec.interimResults = false; rec.maxAlternatives = 1;
    const orig = voiceBtn.innerHTML;
    voiceBtn.innerHTML = '🔴 Listening…';
    voiceBtn.classList.add('is-listening');
    voiceBtn.disabled = true;
    rec.start();

    rec.onresult = async (event) => {
      const text = event.results[0][0].transcript;
      const input = document.getElementById('search-input');
      if (input) input.value = text;
      try {
        const res  = await fetch('/voice_search', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text}) });
        const data = await res.json();
        if (window.showRichResult) {
          const logsRes = await fetch('/api/logs');
          const logs = await logsRes.json();
          window.showRichResult(data.message || 'Done.', text, logs);
          window._filterData = logs;
          if (window.renderFiltered) window.renderFiltered();
        }
        speakMessage(data.message || '');
      } catch(e) {} finally { reset(); }
    };

    rec.onerror = () => { reset(); };
    rec.onend   = reset;

    function reset() {
      voiceBtn.innerHTML = orig;
      voiceBtn.classList.remove('is-listening');
      voiceBtn.disabled = false;
    }
  });
}