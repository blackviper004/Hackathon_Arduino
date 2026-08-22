// theme.js — dark/light toggle (mirrors main.py's st.session_state.theme_mode)
(function () {
  const root = document.documentElement;
  const btn = document.getElementById("theme-toggle-btn");

  function apply(mode) {
    root.setAttribute("data-theme", mode);
    btn.textContent = mode === "dark" ? "☀️ Light Mode" : "🌙 Dark Mode";
    try { localStorage.setItem("swarcare_theme", mode); } catch (e) {}
  }

  let saved = "dark";
  try { saved = localStorage.getItem("swarcare_theme") || "dark"; } catch (e) {}
  apply(saved);

  btn.addEventListener("click", () => {
    const current = root.getAttribute("data-theme") === "dark" ? "dark" : "light";
    apply(current === "dark" ? "light" : "dark");
  });
})();
