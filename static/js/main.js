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

    function searchObject(name) {
        searchResults.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Searching...';
        fetch(`/api/search?name=${name}`)
            .then(response => response.json())
            .then(data => {
                if (data.found) {
                    searchResults.innerHTML = `<span style="color: #7c4dff;"><i class="fas fa-check-circle"></i> ${data.message}</span>`;
                } else {
                    searchResults.innerHTML = `<span style="color: #ff5252;"><i class="fas fa-times-circle"></i> ${data.message}</span>`;
                }
            });
    }

    // Simple Voice Feedback Simulation
    startVoiceBtn.addEventListener('click', () => {
        searchResults.innerHTML = '<span class="voice-status">LISTENING...</span>';

        // This is a prototype: we'll simulate listening for 2 seconds
        setTimeout(() => {
            const mockVoiceQuery = "bottle";
            searchInput.value = mockVoiceQuery;
            searchObject(mockVoiceQuery);
        }, 2000);
    });
});
