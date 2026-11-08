(() => {
  'use strict';
  const status = document.getElementById('copy-status');
  const timers = new WeakMap();
  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const field = document.createElement('textarea');
    field.value = text;
    field.setAttribute('readonly', '');
    field.style.cssText = 'position:fixed;left:-9999px;top:0;';
    document.body.appendChild(field);
    field.select();
    const copied = document.execCommand('copy');
    field.remove();
    if (!copied) throw new Error('Clipboard unavailable');
  }
  for (const button of document.querySelectorAll('[data-copy]')) {
    button.addEventListener('click', async () => {
      const block = document.getElementById(button.dataset.copy);
      if (!block) return;
      try {
        await copyText(block.innerText);
        button.textContent = 'Copied';
        status.textContent = 'Copied to clipboard.';
      } catch (_) {
        button.textContent = 'Select text';
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(block);
        selection.removeAllRanges();
        selection.addRange(range);
        status.textContent = 'Clipboard unavailable. Text selected; copy it with your keyboard.';
      }
      button.focus({ preventScroll: true });
      clearTimeout(timers.get(button));
      timers.set(button, setTimeout(() => { button.textContent = 'Copy'; }, 2500));
    });
  }
  const links = [...document.querySelectorAll('nav a[href^="#"]')];
  const sections = links.map(link => document.querySelector(link.getAttribute('href'))).filter(Boolean);
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      const entering = entries.filter(entry => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
      if (!entering) return;
      for (const link of links) {
        if (link.getAttribute('href') === '#' + entering.target.id) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      }
    }, { rootMargin: '-115px 0px -65% 0px', threshold: 0 });
    sections.forEach(section => observer.observe(section));
  }
})();
