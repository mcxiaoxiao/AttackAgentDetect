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

  const evidence = document.querySelector(".hero-evidence");
  if (evidence && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    const traceObserver = new IntersectionObserver((entries, observer) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        evidence.classList.add("is-visible");
        observer.disconnect();
      }
    }, {threshold: 0.25});
    traceObserver.observe(evidence);
  }
}

for (const button of document.querySelectorAll("[data-copy]")) {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copy);
    const text = target?.innerText?.trim();
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
      button.textContent = "已复制";
      window.setTimeout(() => { button.textContent = original; }, 1400);
    } catch {
      button.textContent = "请手动复制";
      window.setTimeout(() => { button.textContent = "复制"; }, 1600);
    }
  });
}
