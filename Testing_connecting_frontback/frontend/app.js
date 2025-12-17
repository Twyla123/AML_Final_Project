const API_BASE = "http://127.0.0.1:8000";

const elQuery = document.getElementById("query");
const elAsk = document.getElementById("askBtn");
const elClear = document.getElementById("clearBtn");
const elAns = document.getElementById("answer");
const elStatus = document.getElementById("status");
const elBadge = document.getElementById("badge");
document.getElementById("apiBase").textContent = API_BASE;

function setStatus(text, isError = false) {
  elStatus.textContent = text;
  elStatus.classList.toggle("hidden", !text);
  elStatus.classList.toggle("error", !!isError);
}

function setBadge(text) {
  if (!text) {
    elBadge.classList.add("hidden");
    elBadge.textContent = "";
    return;
  }
  elBadge.classList.remove("hidden");
  elBadge.textContent = text;
}

async function ask(query) {
  setStatus("Running agents…", false);
  setBadge("Loading");
  elAns.textContent = "";

  elAsk.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    const data = await res.json();

    if (!res.ok) {
      setStatus(data?.detail ? JSON.stringify(data.detail) : "Request failed.", true);
      setBadge("Error");
      elAns.textContent = "";
      return;
    }

    setStatus("", false);
    setBadge("OK");
    elAns.textContent = data.answer ?? "(empty)";
  } catch (e) {
    setStatus(`Network error: ${e?.message || e}`, true);
    setBadge("Error");
    elAns.textContent = "";
  } finally {
    elAsk.disabled = false;
  }
}

elAsk.addEventListener("click", () => {
  const q = elQuery.value.trim();
  if (!q) return;
  ask(q);
});

elQuery.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    const q = elQuery.value.trim();
    if (!q) return;
    ask(q);
  }
});

elClear.addEventListener("click", () => {
  elQuery.value = "";
  elAns.textContent = "Ready.";
  setStatus("", false);
  setBadge("");
});

document.querySelectorAll(".chip").forEach((btn) => {
  btn.addEventListener("click", () => {
    const q = btn.getAttribute("data-q");
    elQuery.value = q;
    ask(q);
  });
});