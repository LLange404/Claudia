document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const chatOutput = document.getElementById('chat-output');
    
    const API_URL = 'http://localhost:8000/ask';

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    function appendMessage(text, sender, sources = [], scrollToStart = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'claudia-message');
        
        // Konvertiere Markdown-Links zu HTML-Links
        const formattedText = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
        messageDiv.innerHTML = formattedText;

        if (sources.length > 0) {
            const sourcesContainer = document.createElement('div');
            sourcesContainer.classList.add('sources-container');
            sourcesContainer.innerHTML = '<strong>Quellen:</strong>';

            sources.forEach(source => {
                const sourceCard = document.createElement('div');
                sourceCard.classList.add('source-card');
                sourceCard.innerHTML = `
                    <p><strong>AZ:</strong> ${source.file_number || 'N/A'}</p>
                    <p><strong>Gericht:</strong> ${source.court_name || 'N/A'}</p>
                    <p><i>"${source.snippet}"</i></p>
                `;
                sourcesContainer.appendChild(sourceCard);
            });
            messageDiv.appendChild(sourcesContainer);
        }

        chatOutput.appendChild(messageDiv);
        
        // Timeout stellt sicher, dass das DOM fertig gerendert ist (inkl. Quellen)
        setTimeout(() => {
            if (scrollToStart) {
                // Präzise Berechnung: 
                // Aktueller Scroll-Stand + Position der Nachricht im Fenster - oberer Rand des Chat-Fensters
                const targetY = messageDiv.getBoundingClientRect().top + chatOutput.scrollTop - chatOutput.getBoundingClientRect().top - 10;
                
                chatOutput.scrollTo({
                    top: targetY,
                    behavior: 'smooth'
                });
            } else {
                // Bei User-Nachrichten einfach nach ganz unten
                chatOutput.scrollTop = chatOutput.scrollHeight;
            }
        }, 100);
        
        return messageDiv;
    }

    async function sendMessage() {
        const query = userInput.value.trim();
        if (!query) return;

        appendMessage(query, 'user');
        userInput.value = '';

        const loadingMessage = appendMessage('Claudia denkt nach...', 'claudia-message loading');

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: query }),
            });

            if (loadingMessage && chatOutput.contains(loadingMessage)) {
                chatOutput.removeChild(loadingMessage);
            }

            if (!response.ok) {
                const errorData = await response.json();
                appendMessage(`Fehler: ${errorData.detail || 'Etwas ist schiefgelaufen.'}`, 'claudia-message');
                return;
            }

            const data = await response.json();
            // Hier nutzen wir scrollToStart = true für Claudias Antwort
            appendMessage(data.answer, 'claudia-message', data.sources, true);

        } catch (error) {
            if (loadingMessage && chatOutput.contains(loadingMessage)) {
                chatOutput.removeChild(loadingMessage);
            }
            appendMessage('Verbindung zum Backend fehlgeschlagen. Läuft der Server?', 'claudia-message');
            console.error('Fetch Error:', error);
        }
    }
});
