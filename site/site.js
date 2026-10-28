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
