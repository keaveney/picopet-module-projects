// File previews cannot run the search worker reliably. Explain how to enable it.
if (location.protocol === "file:") {
  const search = document.querySelector('[data-md-component="search"]');
  if (search) {
    const hint = document.createElement('p');
    hint.style.cssText='padding:1rem;color:#173d42;background:white;font-size:.8rem';
    hint.textContent='For search, serve this site locally: python -m http.server --directory site 8000. Then open http://localhost:8000.';
    search.prepend(hint);
  }
}
