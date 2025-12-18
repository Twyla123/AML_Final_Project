const API_BASE = "https://interrogatively-untame-terrence.ngrok-free.dev";

// 1. SELECT UI ELEMENTS
const elQuery = document.getElementById("query");
const elAsk = document.getElementById("askBtn");
const elClear = document.getElementById("clearBtn");
const elAns = document.getElementById("answer");
const elStatus = document.getElementById("status");
const elBadge = document.getElementById("badge");
const elApiDisplay = document.getElementById("apiBase");

// 2. INITIALIZATION
if (elApiDisplay) elApiDisplay.textContent = API_BASE;
console.log("🚀 Terminal connected to:", API_BASE);

/**
 * Updates the status message bar
 */
function setStatus(text, isError = false) {
    if (!elStatus) return;
    elStatus.textContent = text;
    elStatus.classList.toggle("hidden", !text);
    elStatus.classList.toggle("error", isError);
}

/**
 * Updates the badge in the Answer panel with colors
 */
function setBadge(text, type = "default") {
    if (!elBadge) return;
    if (!text) {
        elBadge.classList.add("hidden");
        return;
    }
    
    elBadge.classList.remove("hidden");
    elBadge.textContent = text.toUpperCase();

    const colors = {
        loading: { bg: "#FEF0C7", text: "#93370D" },
        success: { bg: "#ECFDF3", text: "#027A48" },
        error:   { bg: "#FEF3F2", text: "#B42318" },
        default: { bg: "#F2F4F7", text: "#344054" }
    };

    const theme = colors[type] || colors.default;
    elBadge.style.backgroundColor = theme.bg;
    elBadge.style.color = theme.text;
}

/**
 * Main function to communicate with Flask backend
 */
async function ask(query) {
    if (!query) return;

    // UI Reset & Loading State
    elAsk.disabled = true;
    elAns.innerHTML = '<div class="loading-state">Synthesizing Agent Intelligence...</div>';
    
    // REVISION: Reset scroll to top when starting a new query
    elAns.scrollTop = 0; 

    setStatus("Coordinating multiple financial agents...");
    setBadge("Loading", "loading");

    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { 
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "true" 
            },
            body: JSON.stringify({ input: query }),
        });

        if (!res.ok) throw new Error(`Server returned ${res.status}`);

        const data = await res.json();

        // 3. RENDER MARKDOWN OUTPUT
        if (typeof marked !== 'undefined') {
            elAns.innerHTML = marked.parse(data.result || "Agent returned no data.");
        } else {
            elAns.textContent = data.result;
        }

        // REVISION: Ensure view starts at the top of the new analysis
        elAns.scrollTop = 0; 

        setStatus(""); 
        setBadge("OK", "success");

    } catch (e) {
        console.error("Fetch Error:", e);
        setStatus(`Network Error: Ensure your Colab server is running.`, true);
        setBadge("Error", "error");
        elAns.innerHTML = `<div style="color: #fda29b">Failed to reach the AI agents. Check your ngrok URL.</div>`;
    } finally {
        elAsk.disabled = false;
    }
}

// 4. EVENT LISTENERS
elAsk?.addEventListener("click", () => {
    ask(elQuery.value.trim());
});

elQuery?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        ask(elQuery.value.trim());
    }
});

elClear?.addEventListener("click", () => {
    elQuery.value = "";
    elAns.innerHTML = "Ready.";
    elAns.scrollTop = 0; // Reset scroll on clear
    setStatus("");
    setBadge("");
});

document.querySelectorAll(".chip").forEach((btn) => {
    btn.addEventListener("click", () => {
        const q = btn.getAttribute("data-q");
        if (elQuery) elQuery.value = q;
        ask(q);
    });
});