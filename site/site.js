/* Preserve previously published section links after splitting the documentation. */
if (/\/(?:index\.html)?$/.test(location.pathname)) {
  const legacyPages = { '#question': 'execution-mode-behavior-correlation.html', '#results': 'execution-mode-behavior-correlation.html', '#related': 'execution-mode-behavior-correlation.html', '#source-reproduction': 'execution-mode-behavior-correlation.html' };
  const destination = legacyPages[location.hash];
  if (destination) location.replace(destination + location.search + location.hash);
}

/* Lightweight navigation state; all content remains usable without JavaScript. */
const links = [...document.querySelectorAll(".rail nav a")];
if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        links.forEach((link) => link.classList.toggle("active", link.hash === "#" + entry.target.id));
      }
    }
  }, {rootMargin: "-15% 0px -72% 0px"});
  document.querySelectorAll("main section[id]").forEach((section) => observer.observe(section));


}

for (const button of document.querySelectorAll("[data-copy]")) {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copy);
    const text = target?.textContent?.trim().replace(/^\$\s*/, "");
    if (!text) return;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const field = document.createElement("textarea");
        field.value = text;
        field.setAttribute("readonly", "");
        field.style.position = "fixed";
        field.style.opacity = "0";
        document.body.append(field);
        field.select();
        document.execCommand("copy");
        field.remove();
      }
      const original = button.textContent;
      const english = () => document.documentElement.lang === "en";
      const toast = document.getElementById("copy-toast");
      if (toast && button.dataset.copy !== "ai-brief-markdown") {
        toast.textContent = english()
          ? (button.dataset.copySuccessEn || "Copied")
          : (button.dataset.copySuccess || "已复制");
        toast.hidden = false;
        window.clearTimeout(toast.hideTimer);
        toast.hideTimer = window.setTimeout(() => { toast.hidden = true; }, 2400);
      }
      button.textContent = button.dataset.copy === "ai-brief-markdown"
        ? (english() ? button.dataset.copySuccessEn : button.dataset.copySuccess)
        : (english() ? "Copied" : "已复制");
      window.setTimeout(() => { button.textContent = original; }, 1400);
    } catch {
      const english = document.documentElement.lang === "en";
      const toast = document.getElementById("copy-toast");
      if (toast && button.dataset.copy !== "ai-brief-markdown") {
        toast.textContent = english ? "Copy failed — select the text manually" : "复制失败，请手动选择文本";
        toast.hidden = false;
        window.clearTimeout(toast.hideTimer);
        toast.hideTimer = window.setTimeout(() => { toast.hidden = true; }, 3000);
      }
      button.textContent = english ? "Select and copy" : "请手动复制";
      window.setTimeout(() => { button.textContent = button.dataset.en || (english ? "Copy" : "复制"); }, 1600);
    }
  });
}

const commandLineToast = document.getElementById("copy-toast");
function showCommandLineToast(message, duration = 2400) {
  if (!commandLineToast) return;
  commandLineToast.textContent = message;
  commandLineToast.hidden = false;
  window.clearTimeout(commandLineToast.hideTimer);
  commandLineToast.hideTimer = window.setTimeout(() => { commandLineToast.hidden = true; }, duration);
}

async function copyCommandLine(line) {
  const text = line.textContent?.trim().replace(/^\$\s*/, "");
  if (!text) return;
  const english = document.documentElement.lang === "en";
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const field = document.createElement("textarea");
      field.value = text;
      field.setAttribute("readonly", "");
      field.style.position = "fixed";
      field.style.opacity = "0";
      document.body.append(field);
      field.select();
      let copied = false;
      try {
        copied = document.execCommand("copy");
      } finally {
        field.remove();
      }
      if (!copied) throw new Error("Clipboard copy failed");
    }
    showCommandLineToast(english ? "Command line copied" : "已复制此行命令");
  } catch {
    showCommandLineToast(
      english ? "Copy failed. Select the command manually." : "复制失败，请手动选择命令",
      3000,
    );
  }
}

for (const line of document.querySelectorAll(".docs-command-window .code-line")) {
  line.setAttribute("role", "button");
  line.tabIndex = 0;
  const updateCopyHint = () => {
    line.title = document.documentElement.lang === "en"
      ? "Click to copy this line"
      : "点击复制这一行";
  };
  updateCopyHint();
  document.addEventListener("pagelanguagechange", updateCopyHint);
  line.addEventListener("click", () => copyCommandLine(line));
  line.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    copyCommandLine(line);
  });
}

// Reveal the runnable source workflow when its link is followed from the API preview.
for (const link of document.querySelectorAll('a[href="#source-reproduction"]')) {
  link.addEventListener('click', () => {
    const details = document.getElementById('source-reproduction')?.closest('details');
    if (details) details.open = true;
  });
}

if (location.hash === "#source-reproduction") {
  document.getElementById("source-reproduction")?.closest("details")?.setAttribute("open", "");
}
