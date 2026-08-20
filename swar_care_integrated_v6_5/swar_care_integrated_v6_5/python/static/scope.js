// scope.js — Audio Live Monitor (USB Microphone, 24 FPS oscilloscope canvas)
(function () {
  const canvas = document.getElementById("audio-canvas");
  const ctx = canvas.getContext("2d");

  function styleVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function updateCanvasDimensions() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const containerWidth = rect.width || 360;
    const containerHeight = rect.height || 205;
    canvas.width = containerWidth * dpr;
    canvas.height = containerHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w: containerWidth, h: containerHeight };
  }
  updateCanvasDimensions();
  window.addEventListener("resize", updateCanvasDimensions);

  const visibleWindowSec = 4.0;
  let history = [];
  let serverTimeSec = 0.0;
  let lastFetchMs = performance.now();
  let displayTimeSec = 0.0;
  let state = "STOPPED";

  const targetFPS = 24;
  const frameIntervalMs = 1000 / targetFPS;
  let lastRenderMs = performance.now();

  document.addEventListener("swarcare:telemetry", (evt) => {
    const audio = evt.detail.audio;
    if (audio.samples && audio.samples.length > 0) {
      history = audio.samples;
    }
    serverTimeSec = audio.elapsed_s || 0.0;
    state = audio.state || "STOPPED";
    lastFetchMs = performance.now();
  });

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    if (w < 5 || h < 5) return;

    const nowMs = performance.now();

    if (state === "RECORDING") {
      const targetTimeSec = serverTimeSec + (nowMs - lastFetchMs) / 1000.0;
      if (displayTimeSec === 0 || Math.abs(displayTimeSec - targetTimeSec) > 1.0) {
        displayTimeSec = targetTimeSec;
      } else {
        displayTimeSec += (targetTimeSec - displayTimeSec) * 0.25;
      }
    } else {
      displayTimeSec = serverTimeSec;
    }

    const gridColor = styleVar("--scope-grid");
    const borderBoxColor = styleVar("--border-subtle");
    const axisTextColor = styleVar("--scope-axis");
    const activeWaveColor = styleVar("--scope-wave");
    const pausedWaveColor = styleVar("--accent-amber");
    const stoppedWaveColor = styleVar("--text-secondary");
    const markerColor = styleVar("--term-marker-color");
    const labelTextColor = styleVar("--text-primary");

    const isMobile = w < 480;
    const padLeft = isMobile ? 56 : 68;
    const padBottom = isMobile ? 28 : 32;
    const padTop = 18;
    const padRight = isMobile ? 12 : 20;
    const plotW = Math.max(10, w - padLeft - padRight);
    const plotH = Math.max(10, h - padTop - padBottom);
    const midY = padTop + plotH / 2;

    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padTop + (plotH / 4) * i;
      ctx.beginPath(); ctx.moveTo(padLeft, y); ctx.lineTo(w - padRight, y); ctx.stroke();
    }

    ctx.strokeStyle = borderBoxColor;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = axisTextColor;
    ctx.beginPath(); ctx.moveTo(padLeft, midY); ctx.lineTo(w - padRight, midY); ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = axisTextColor;
    ctx.font = isMobile ? "8.5px monospace" : "10px monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText("+1.0", padLeft - 6, padTop);
    ctx.fillText("0.0", padLeft - 6, midY);
    ctx.fillText("-1.0", padLeft - 6, padTop + plotH);

    const N = history.length;
    if (N > 0 && state !== "STOPPED") {
      ctx.fillStyle = state === "RECORDING" ? activeWaveColor : pausedWaveColor;
      const barWidth = isMobile ? 1.5 : 2.0;
      for (let col = 0; col < plotW; col += barWidth) {
        const x = padLeft + col;
        let sampleIdx = Math.floor((col / plotW) * N);
        if (sampleIdx >= N) sampleIdx = N - 1;
        const amplitude = Math.abs(history[sampleIdx]);
        const barHeight = Math.max(2, amplitude * (plotH * 0.88));
        ctx.fillRect(x, midY - barHeight / 2, barWidth, barHeight);
      }
    } else if (N > 0 && state === "STOPPED") {
      ctx.fillStyle = stoppedWaveColor;
      const barWidth = isMobile ? 1.5 : 2.0;
      for (let col = 0; col < plotW; col += barWidth) {
        const x = padLeft + col;
        let sampleIdx = Math.floor((col / plotW) * N);
        if (sampleIdx >= N) sampleIdx = N - 1;
        const amplitude = Math.abs(history[sampleIdx]);
        const barHeight = Math.max(2, amplitude * (plotH * 0.88));
        ctx.fillRect(x, midY - barHeight / 2, barWidth, barHeight);
      }
    } else {
      ctx.fillStyle = axisTextColor;
      ctx.font = isMobile ? "10px monospace" : "11.5px monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("--- AUDIO MONITOR IDLE ---", padLeft + plotW / 2, midY);
    }

    const pixelsPerSec = plotW / visibleWindowSec;
    const start2sMarker = Math.floor(Math.max(0, displayTimeSec - visibleWindowSec) / 2.0) * 2.0;
    const end2sMarker = displayTimeSec + 2.0;

    for (let tMark = start2sMarker; tMark <= end2sMarker; tMark += 2.0) {
      const markOffset = displayTimeSec - tMark;
      const markX = (w - padRight) - markOffset * pixelsPerSec;
      if (markX >= padLeft && markX <= w - padRight) {
        ctx.save();
        ctx.setLineDash([2, 2]);
        ctx.strokeStyle = markerColor;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(markX, padTop); ctx.lineTo(markX, padTop + plotH); ctx.stroke();
        ctx.restore();

        ctx.fillStyle = markerColor;
        ctx.font = isMobile ? "bold 8px monospace" : "bold 9.5px monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText(tMark.toFixed(1) + "s", markX, padTop - 2);

        ctx.fillStyle = axisTextColor;
        ctx.textBaseline = "top";
        ctx.fillText(tMark.toFixed(1) + "s", markX, padTop + plotH + 3);
      }
    }

    ctx.fillStyle = axisTextColor;
    ctx.font = isMobile ? "bold 9.5px monospace" : "bold 11px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText("Time", padLeft + plotW / 2, h - 12);

    ctx.save();
    ctx.translate(isMobile ? 12 : 14, padTop + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = labelTextColor;
    ctx.font = isMobile ? "bold 9px monospace" : "bold 11px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Normalised Amplitude", 0, 0);
    ctx.restore();
  }

  function anim(nowMs) {
    requestAnimationFrame(anim);
    const elapsed = nowMs - lastRenderMs;
    if (elapsed >= frameIntervalMs) {
      lastRenderMs = nowMs - (elapsed % frameIntervalMs);
      draw();
    }
  }
  requestAnimationFrame(anim);
})();
