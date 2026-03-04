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

    function appendMessage(text, sender, sources = []) {
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
        chatOutput.scrollTop = chatOutput.scrollHeight;
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

            chatOutput.removeChild(loadingMessage);

            if (!response.ok) {
                const errorData = await response.json();
                appendMessage(`Fehler: ${errorData.detail || 'Etwas ist schiefgelaufen.'}`, 'claudia-message');
                return;
            }

            const data = await response.json();
            appendMessage(data.answer, 'claudia-message', data.sources);

        } catch (error) {
            chatOutput.removeChild(loadingMessage);
            appendMessage('Verbindung zum Backend fehlgeschlagen. Läuft der Server?', 'claudia-message');
            console.error('Fetch Error:', error);
        }
    }
});
