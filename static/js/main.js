document.addEventListener('DOMContentLoaded', () => {
    const activityLog = document.getElementById('activityLog');
    const searchInput = document.getElementById('objectSearch');
    const searchResults = document.getElementById('searchResults');
    const startVoiceBtn = document.getElementById('startVoice');

    // Fetch latest logs every 2 seconds
    function fetchLogs() {
        fetch('/api/logs')
            .then(response => response.json())
            .then(data => {
                activityLog.innerHTML = '';
                data.forEach(log => {
                    const item = document.createElement('div');
                    item.className = 'log-item';
                    item.innerHTML = `
                        <div class="info">
                            <div class="name">${log.object_name} (ID: ${log.object_id})</div>
                            <div class="zone"><i class="fas fa-location-dot"></i> ${log.zone}</div>
                        </div>
                        <div class="time">${log.timestamp.split(' ')[1]}</div>
                    `;
                    activityLog.appendChild(item);
                });
            });
    }

    setInterval(fetchLogs, 2000);
    fetchLogs();

    // Search functionality
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const query = searchInput.value.trim();
            if (query) {
                searchObject(query);
            }
        }
    });

    const voiceToggle = document.getElementById('voiceAssistantToggle');
    let voiceEnabled = localStorage.getItem('voiceEnabled') === 'true';
    voiceToggle.checked = voiceEnabled;

    voiceToggle.addEventListener('change', () => {
        voiceEnabled = voiceToggle.checked;
        localStorage.setItem('voiceEnabled', voiceEnabled);
        if (voiceEnabled) {
            speak("Voice assistant enabled");
        }
    });

    function speak(text) {
        if (!voiceEnabled && !text.includes("enabled")) return;
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
    }

    function searchObject(query) {
        // Extract object name if it's a full sentence (e.g. "where is the bottle")
        let name = query.toLowerCase();
        const keywords = ["bottle", "phone", "cell phone", "person", "chair", "bed", "remote", "laptop"];

        // Simple extraction logic
        for (const word of keywords) {
            if (name.includes(word)) {
                name = word;
                break;
            }
        }

        searchResults.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Searching...';
        fetch(`/api/search?name=${name}`)
            .then(response => response.json())
            .then(data => {
                if (data.found) {
                    const responseMsg = `${name} was ${data.message}`;
                    searchResults.innerHTML = `<span style="color: #7c4dff;"><i class="fas fa-check-circle"></i> ${data.message}</span>`;
                    if (voiceEnabled) speak(responseMsg);
                } else {
                    searchResults.innerHTML = `<span style="color: #ff5252;"><i class="fas fa-times-circle"></i> ${data.message}</span>`;
                    if (voiceEnabled) speak(`I couldn't find the ${name} in the recent logs.`);
                }
            })
            .catch(err => {
                searchResults.innerHTML = `<span style="color: #ff5252;">Error searching for object</span>`;
            });
    }

    // Voice Input Implementation
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.lang = 'en-US';
        recognition.interimResults = false;

        recognition.onstart = () => {
            searchResults.innerHTML = '<span class="voice-status">LISTENING...</span>';
            startVoiceBtn.classList.add('listening-pulse');
        };

        recognition.onspeechend = () => {
            recognition.stop();
            startVoiceBtn.classList.remove('listening-pulse');
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            searchInput.value = transcript;
            searchObject(transcript);
        };

        recognition.onerror = (event) => {
            searchResults.innerHTML = `<span style="color: #ff5252;">Voice Error: ${event.error}</span>`;
            startVoiceBtn.classList.remove('listening-pulse');
        };

        startVoiceBtn.addEventListener('click', () => {
            recognition.start();
        });
    } else {
        startVoiceBtn.style.display = 'none';
        console.log("Speech recognition not supported in this browser.");
    }
});
