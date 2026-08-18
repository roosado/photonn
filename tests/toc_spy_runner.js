/*
 * Node runner for the contents card's active-section mark.
 *
 * The card highlights the section being read. That behaviour cannot be checked in
 * the driven browser: the tab there is always hidden, and while the page really
 * does scroll (window.scrollY changes), *no scroll event is ever delivered* and no
 * IntersectionObserver callback fires either -- measured, not assumed. So the one
 * path that matters most is exactly the one a screenshot cannot reach.
 *
 * It is therefore exercised here against the *shipped* page: the script is pulled
 * out of site/tolerance.html rather than retyped, so this tests the bytes a reader
 * downloads. Scroll position is under this file's control. Prints one JSON object.
 * Driven by tests/test_toc_spy.py.
 */
const fs = require("fs");
const path = require("path");

const PAGE = path.join(__dirname, "..", "site", "tolerance.html");

/** The contents-card IIFE, lifted out of the built page by its own marker. */
function spySource(html) {
  const start = html.indexOf("(function(){\n  var links=document.querySelectorAll('.toc-list a[href^=\"#\"]');");
  if (start < 0) throw new Error("contents-card script not found in the built page");
  const end = html.indexOf("\n})();", start);
  if (end < 0) throw new Error("contents-card script is not terminated");
  return html.slice(start, end + "\n})();".length);
}

/** Ids the card links to, in the order the card lists them. */
function cardIds(html) {
  const nav = /<nav class="toc[^"]*" aria-label="On this page">[\s\S]*?<\/nav>/.exec(html);
  if (!nav) throw new Error("contents card not found in the built page");
  return [...nav[0].matchAll(/href="#([^"]+)"/g)].map((m) => m[1]);
}

/**
 * A stand-in for the parts of the DOM the script touches. Headings are laid out
 * SPACING px apart in document order; `scrollTo` moves the whole column, exactly
 * as a real scroll moves every getBoundingClientRect().top together.
 */
function makeEnv(ids, spacing) {
  let scroll = 0;
  const timers = [];
  const listeners = {};

  const headings = ids.map((id, i) => ({
    id,
    _top: 200 + i * spacing,
    getBoundingClientRect() { return { top: this._top - scroll }; },
  }));

  const anchors = ids.map((id) => ({
    _href: "#" + id,
    attrs: {},
    getAttribute(k) { return k === "href" ? this._href : this.attrs[k]; },
    setAttribute(k, v) { this.attrs[k] = v; },
    removeAttribute(k) { delete this.attrs[k]; },
    addEventListener() {},
  }));

  const list = anchors.slice();
  list.forEach = Array.prototype.forEach.bind(anchors);

  const env = {
    document: {
      querySelectorAll(sel) {
        if (sel === '.toc-list a[href^="#"]') return list;
        throw new Error("unexpected selector " + sel);
      },
      getElementById(id) { return headings.find((h) => h.id === id) || null; },
    },
    setTimeout(fn, ms) { timers.push({ at: ms, fn }); return timers.length; },
    addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },
    // Scroll the column, fire the event, then let the throttle window elapse.
    scrollTo(y) {
      scroll = y;
      (listeners.scroll || []).forEach((fn) => fn());
      const due = timers.splice(0, timers.length);
      due.forEach((t) => t.fn());
    },
    marked() {
      const on = anchors.filter((a) => a.attrs["aria-current"]);
      return { count: on.length, id: on.length ? on[0]._href.slice(1) : null,
               value: on.length ? on[0].attrs["aria-current"] : null };
    },
    listeners,
  };
  return env;
}

// The built pages are written in text mode on Windows, so they carry CRLF; the
// script is matched by its own source lines, which do not.
const html = fs.readFileSync(PAGE, "utf8").replace(/\r\n/g, "\n");
const ids = cardIds(html);
const SPACING = 900;
const env = makeEnv(ids, SPACING);

// The script closes over bare `addEventListener` / `setTimeout` / `document`, the
// way it does on a real page, so hand it exactly those.
new Function("document", "setTimeout", "addEventListener", spySource(html))(
  env.document, env.setTimeout, env.addEventListener,
);

const onLoad = env.marked();

// Walk down the page, stopping just past each heading in turn.
const walk = ids.map((id, i) => {
  env.scrollTo(200 + i * SPACING - 40 + 60);   // heading now 20 px above the 90 px line
  const m = env.marked();
  return { expected: id, got: m.id, count: m.count, value: m.value };
});

// A jump straight to the last heading, the way clicking the card behaves.
env.scrollTo(200 + (ids.length - 1) * SPACING - 78);
const afterJump = env.marked();

// And back to the very top.
env.scrollTo(0);
const backToTop = env.marked();

console.log(JSON.stringify({
  ids,
  onLoad,
  walk,
  afterJump,
  backToTop,
  listens: Object.keys(env.listeners).sort(),
}, null, 2));
