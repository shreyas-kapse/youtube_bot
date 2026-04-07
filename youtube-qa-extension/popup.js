document.addEventListener("DOMContentLoaded", async () => {

    const chat = document.getElementById("chat");
    const input = document.getElementById("query");
    const btn = document.getElementById("askBtn");

    function addMessage(text, sender) {
        const div = document.createElement("div");
        div.className = `msg ${sender}`;
        div.innerText = text;

        chat.appendChild(div);
        chat.scrollTop = chat.scrollHeight;
    }

    function addSegments(segments) {
        segments.forEach(seg => {
            const div = document.createElement("div");
            div.className = "msg bot";

            div.innerHTML = `
                <p>${seg.sentence}</p>
                <small>⏱ ${seg.timestamp}</small><br/>
                <button class="jump" data-url="${seg.url}">Jump</button>
            `;

            chat.appendChild(div);
        });

        chat.scrollTop = chat.scrollHeight;

        document.querySelectorAll(".jump").forEach(btn => {
            btn.onclick = () => {
                chrome.tabs.update({ url: btn.dataset.url });
            };
        });
    }

    async function getCurrentVideoId() {
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        const url = tabs[0].url;

        if (!url.includes("youtube.com/watch")) return null;

        return new URL(url).searchParams.get("v");
    }

    async function saveMessage(videoId, message) {
        const key = `chat_${videoId}`;
        const data = await chrome.storage.local.get(key);
        const history = data[key] || [];

        history.push(message);

        await chrome.storage.local.set({ [key]: history });
    }

    async function loadChat(videoId) {
        const key = `chat_${videoId}`;
        const data = await chrome.storage.local.get(key);
        const history = data[key] || [];

        chat.innerHTML = "";

        history.forEach(msg => {
            if (msg.type === "user") {
                addMessage(msg.text, "user");
            } else if (msg.type === "bot") {
                addSegments(msg.segments);
            }
        });
    }

    const videoId = await getCurrentVideoId();

    if (videoId) {
        await loadChat(videoId);
        try {
            await fetch(`http://localhost:8000/process?video_id=${videoId}`, {
                method: "GET"
            });
            console.log("Process API called");
        } catch (err) {
            console.error("Process API failed", err);
        }
    }

    btn.addEventListener("click", async () => {
        const query = input.value.trim();
        if (!query) return;

        const videoId = await getCurrentVideoId();

        if (!videoId) {
            addMessage("Not a YouTube video", "bot");
            return;
        }

        addMessage(query, "user");

        await saveMessage(videoId, {
            type: "user",
            text: query
        });

        input.value = "";

        addMessage("Thinking...", "bot");

        try {
            const response = await fetch(
                `http://127.0.0.1:8000/ask?query=${encodeURIComponent(query)}&video_id=${videoId}`
            );

            let data = await response.json();

            if (typeof data === "string") {
                data = JSON.parse(data);
            }

            chat.removeChild(chat.lastChild);

            if (data.segments && data.segments.length > 0) {
                addSegments(data.segments);

                await saveMessage(videoId, {
                    type: "bot",
                    segments: data.segments
                });

            } else {
                addMessage("No results found", "bot");
            }

        } catch (err) {
            chat.removeChild(chat.lastChild);
            addMessage("Error fetching response", "bot");
            console.error(err);
        }
    });

});