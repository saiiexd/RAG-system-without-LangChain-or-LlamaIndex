const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileName = document.getElementById('file-name');
const uploadBtn = document.getElementById('upload-btn');
const uploadStatus = document.getElementById('upload-status');
const chatBox = document.getElementById('chat-box');
const queryInput = document.getElementById('query-input');
const sendBtn = document.getElementById('send-btn');

let selectedFile = null;

// File Selection Logic
dropZone.onclick = () => fileInput.click();

fileInput.onchange = (e) => {
    handleFiles(e.target.files);
};

dropZone.ondragover = (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
};

dropZone.ondragleave = () => {
    dropZone.classList.remove('dragover');
};

dropZone.ondrop = (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
};

function handleFiles(files) {
    if (files.length > 0) {
        selectedFile = files[0];
        fileName.innerText = selectedFile.name;
        uploadBtn.disabled = false;
    }
}

// Upload Logic
uploadBtn.onclick = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    uploadBtn.disabled = true;
    uploadStatus.innerText = "Processing document...";
    uploadStatus.style.color = "var(--text-main)";

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            uploadStatus.innerText = "✓ Document processed successfully!";
            uploadStatus.style.color = "green";
            queryInput.disabled = false;
            sendBtn.disabled = false;
            addMessage("Assistant", "The document is ready! You can ask questions now.");
        } else {
            const errorText = data.detail || "Unknown error";
            uploadStatus.innerText = "Error: " + errorText;
            uploadStatus.style.color = "red";
            uploadBtn.disabled = false;
            addMessage("Assistant", "I couldn't process the document. Error: " + errorText);
        }
    } catch (error) {
        console.error("Upload error:", error);
        console.log("Current Page URL:", window.location.href);
        const isFileProtocol = window.location.protocol === 'file:';

        if (isFileProtocol) {
            uploadStatus.innerText = "Error: Please open http://localhost:8000 in your browser, NOT the file directly.";
            addMessage("Assistant", "It looks like you opened the HTML file directly. Please navigate to http://localhost:8000 in your browser address bar.");
        } else {
            uploadStatus.innerText = "Connection error. Make sure the server is running on port 8000.";
            addMessage("Assistant", "Network error: Could not reach the backend. Check if the server is running.");
        }
        uploadStatus.style.color = "red";
        uploadBtn.disabled = false;
    }
};

// Chat Logic
async function askQuestion() {
    const question = queryInput.value.trim();
    if (!question) return;

    addMessage("User", question);
    queryInput.value = "";
    queryInput.disabled = true;
    sendBtn.disabled = true;

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        const data = await response.json();

        if (response.ok) {
            addMessage("Assistant", data.answer);
        } else {
            const errorMsg = data.detail || "Server error";
            addMessage("Assistant", "Error: " + errorMsg);
        }

    } catch (error) {
        console.error("Chat error:", error);
        addMessage("Assistant", "Error: Could not reach the server. Please check your network connection.");
    } finally {
        queryInput.disabled = false;
        sendBtn.disabled = false;
        queryInput.focus();
    }
}

sendBtn.onclick = askQuestion;
queryInput.onkeypress = (e) => {
    if (e.key === 'Enter') askQuestion();
};

function addMessage(sender, text) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', sender.toLowerCase());
    msgDiv.innerText = text;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}


