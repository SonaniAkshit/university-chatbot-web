function sendMessage() {
    const userInput = document.getElementById('userInput').value.trim();
    if (!userInput) return;

    // Display user message
    const chatBody = document.getElementById('chatBody');
    const userMessage = document.createElement('div');
    userMessage.className = 'message user';
    userMessage.textContent = userInput;
    chatBody.appendChild(userMessage);
    chatBody.scrollTop = chatBody.scrollHeight;

    // Show typing indicator
    const typingIndicator = document.getElementById('typingIndicator');
    typingIndicator.style.display = 'flex';

    // Clear input
    document.getElementById('userInput').value = '';

    // Send to backend
    fetch('/chatbot/webhook', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userInput })
    })
    .then(response => response.json())
    .then(data => {
        // Hide typing indicator
        typingIndicator.style.display = 'none';

        // Display bot message
        const botMessage = document.createElement('div');
        botMessage.className = 'message bot fade-in';
        botMessage.textContent = data.response || 'Sorry, I didn’t understand that.';
        chatBody.appendChild(botMessage);
        chatBody.scrollTop = chatBody.scrollHeight;
    })
    .catch(error => {
        console.error('Error:', error);
        typingIndicator.style.display = 'none';
        const botMessage = document.createElement('div');
        botMessage.className = 'message bot fade-in';
        botMessage.textContent = 'Error connecting to the server.';
        chatBody.appendChild(botMessage);
        chatBody.scrollTop = chatBody.scrollHeight;
    });
}

// Allow sending message with Enter key
document.getElementById('userInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') sendMessage();
});