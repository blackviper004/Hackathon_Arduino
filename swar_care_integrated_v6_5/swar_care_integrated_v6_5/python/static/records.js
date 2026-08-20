// records.js — Saved Records Explorer (Tab 2)
(function () {
  let limit = 3;

  const csvListEl = document.getElementById("csv-file-list");
  const wavListEl = document.getElementById("wav-file-list");
  const zipWrap = document.getElementById("zip-download-wrap");
  const btnZip = document.getElementById("btn-download-zip");
  const btnShowMore = document.getElementById("btn-show-more");
  const btnResetView = document.getElementById("btn-reset-view");
  const btnClearAll = document.getElementById("btn-clear-all");

  function escapeAttr(s) {
    return (s || "").replace(/&/g, "&amp;").replace(/"/g, "&quot;");
  }
  function escapeHtml(s) {
    return (s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function fileCardHtml(file, kind) {
    const icon = kind === "csv" ? "📄" : "🎧";
    const dlLabel = kind === "csv" ? "📥 CSV" : "📥 WAV";
    const replay = kind === "wav"
      ? `<details class="expander replay-wrap">
           <summary>▶️ Replay</summary>
           <audio controls preload="none" src="/api/recordings/audio/${encodeURIComponent(file.name)}"></audio>
         </details>`
      : "";
    return `
    <div class="file-card" data-name="${escapeAttr(file.name)}" data-kind="${kind}">
      <div class="file-name">${icon} <strong>${escapeHtml(file.name)}</strong></div>
      <input type="text" class="rename-input" value="${escapeAttr(file.name)}" placeholder="Rename download:" />
      <div class="file-card-actions">
        <button class="btn btn-dl" type="button">${dlLabel}</button>
        <button class="btn btn-rename" type="button" title="Rename file on disk">✏️</button>
        <button class="btn btn-delete" type="button" title="Delete file">🗑️</button>
      </div>
      ${replay}
    </div>`;
  }

  function wireCard(cardEl, kind) {
    const name = cardEl.dataset.name;
    const ext = kind === "csv" ? ".csv" : ".wav";
    const input = cardEl.querySelector(".rename-input");

    function targetName() {
      let v = input.value.trim() || name;
      if (!v.endsWith(ext)) v += ext;
      return v;
    }

    cardEl.querySelector(".btn-dl").addEventListener("click", () => {
      const t = targetName();
      const url = `/api/recordings/download/${encodeURIComponent(name)}?as_name=${encodeURIComponent(t)}`;
      const a = document.createElement("a");
      a.href = url; a.download = t;
      document.body.appendChild(a); a.click(); a.remove();
    });

    cardEl.querySelector(".btn-rename").addEventListener("click", async () => {
      const t = targetName();
      if (t === name) return;
      try {
        const res = await fetch("/api/recordings/rename", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ old_name: name, new_name: t }),
        });
        if (!res.ok) throw new Error();
        window.SwarCare.toast(`✏️ Renamed to ${t}!`);
        refresh();
      } catch (e) {
        window.SwarCare.toast("⚠️ Rename failed");
      }
    });

    cardEl.querySelector(".btn-delete").addEventListener("click", async () => {
      try {
        const res = await fetch(`/api/recordings/${encodeURIComponent(name)}`, { method: "DELETE" });
        if (!res.ok) throw new Error();
        window.SwarCare.toast(`🗑️ Removed ${name}!`);
        refresh();
      } catch (e) {
        window.SwarCare.toast("⚠️ Delete failed");
      }
    });
  }

  function renderList(container, files, kind) {
    if (!files.length) {
      container.innerHTML = `<p class="empty-note">No ${kind.toUpperCase()} records available.</p>`;
      return;
    }
    container.innerHTML = files.map((f) => fileCardHtml(f, kind)).join("");
    container.querySelectorAll(".file-card").forEach((card) => wireCard(card, kind));
  }

  async function refresh() {
    try {
      const res = await fetch(`/api/recordings?limit=${limit}`);
      if (!res.ok) return;
      const data = await res.json();

      renderList(csvListEl, data.csv_files, "csv");
      renderList(wavListEl, data.wav_files, "wav");

      const hasAny = data.csv_total > 0 || data.wav_total > 0;
      zipWrap.classList.toggle("hidden", !hasAny);

      const totalMax = Math.max(data.csv_total, data.wav_total);
      if (limit < totalMax) {
        btnShowMore.disabled = false;
        btnShowMore.textContent = "🔽 Show More";
      } else {
        btnShowMore.disabled = true;
        btnShowMore.textContent = "✨ All Displayed";
      }
      btnResetView.disabled = limit <= 3;
    } catch (e) {
      // leave last rendered state on transient network errors
    }
  }

  btnShowMore.addEventListener("click", () => { limit += 5; refresh(); });
  btnResetView.addEventListener("click", () => { limit = 3; refresh(); });

  btnClearAll.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/recordings/clear_all", { method: "POST" });
      if (!res.ok) throw new Error();
      limit = 3;
      window.SwarCare.toast("💥 Storage directory wiped clean!");
      refresh();
    } catch (e) {
      window.SwarCare.toast("⚠️ Clear failed");
    }
  });

  btnZip.addEventListener("click", () => {
    const a = document.createElement("a");
    a.href = "/api/recordings/zip";
    document.body.appendChild(a); a.click(); a.remove();
  });

  window.SwarCareRecords = { refresh };
  refresh();
})();
