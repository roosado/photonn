/*
 * mount_queue.js -- when the widgets on a page are allowed to start.
 *
 * Every widget on this site mounts itself, and used to do so from one shared
 * DOMContentLoaded handler. On the optics page that meant the browser parsed two
 * megabytes of inline script and then, in a single uninterrupted task, decoded a
 * 56-mask model, ran it, ran it again for the 3D stage, and built ~60 offscreen
 * canvases -- before the first pixel of prose was on screen. On a laptop that is
 * an unpleasant half second. On a phone it is long enough that the tab is judged
 * unresponsive and reloaded, which is why the page could appear to never load.
 *
 * Nothing here makes a widget cheaper or less interactive. It changes *when* the
 * cost is paid:
 *
 *   1. Nothing runs until the first paint has happened, so the page is readable
 *      immediately and the reader has something to look at while the machines
 *      warm up.
 *   2. Widgets start one at a time, in document order, with a yield between them.
 *      The main thread is never held for more than one widget's mount, so the
 *      page keeps responding to scrolls and taps throughout.
 *   3. A widget far below the fold waits until the reader is approaching it. It
 *      still starts by itself -- no tap, no button, nothing to discover -- just
 *      not while it is a screen and a half away.
 *
 * There is deliberately no small-screen fallback and no "tap to activate": a
 * phone gets the same working machines a laptop does, only sequenced so the
 * device can survive delivering them.
 *
 * Usage:  window.PhotonnMount('containerId', function (el) { ... }, opts);
 * opts (all optional):
 *   defer  -- wait until the element is near the viewport before mounting.
 *
 * Standalone demo pages do not include this file, so every generated mount falls
 * back to a plain DOMContentLoaded when window.PhotonnMount is absent.
 */
(function () {
  "use strict";

  const STYLE_ID = "photonn-mount-style";
  const CSS = `
.pm-pending{min-height:120px;display:flex;align-items:center;justify-content:center;}
.pm-pending::after{content:"Warming up\\2026";font:0.9rem/1.5 ui-sans-serif,system-ui,sans-serif;
  color:#7a8698;opacity:0;animation:pm-fade .4s ease-out .25s forwards;}
@keyframes pm-fade{to{opacity:1;}}
@media (prefers-reduced-motion:reduce){.pm-pending::after{animation:none;opacity:1;}}
`;

  function injectStyle() {
    if (typeof document === "undefined" || document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  //: How far ahead of the viewport a deferred widget starts warming up. Generous
  //: on purpose -- a widget that is still blank when it scrolls into view has
  //: saved load time by spending the reader's attention, which is a bad trade.
  const NEAR_MARGIN = "600px";

  const queue = [];
  let draining = false;
  let armed = false;      // afterFirstPaint has been scheduled
  let painted = false;    // ...and has fired; until then nothing may run

  function idle(fn) {
    if (typeof window.requestIdleCallback === "function") {
      window.requestIdleCallback(fn, { timeout: 400 });
    } else {
      setTimeout(fn, 0);
    }
  }

  //: Longest we will wait for a paint that should already have happened. Only a
  //: pathologically busy main thread reaches this; it exists so that no failure
  //: mode of the paint signal can leave a page stuck showing "warming up".
  const PAINT_TIMEOUT_MS = 2000;

  /**
   * Run `fn` once the page has painted -- or once it is clear no paint is coming.
   *
   * A hidden tab never paints and never fires requestAnimationFrame. That is not
   * an edge case: opening a link in a background tab, or restoring a session,
   * lands here. Waiting for a paint that will not arrive would leave every widget
   * queued forever, so a hidden document skips the wait entirely and starts its
   * widgets straight away -- by the time the reader switches to the tab, the page
   * is ready rather than warming up.
   */
  function afterFirstPaint(fn) {
    let done = false;
    function go() {
      if (done) return;
      done = true;
      fn();
    }
    if (typeof window.requestAnimationFrame !== "function"
        || (typeof document !== "undefined" && document.visibilityState === "hidden")) {
      return setTimeout(go, 0);
    }
    // rAF runs *before* the paint it belongs to; the timeout inside it lands
    // after that paint has been committed.
    window.requestAnimationFrame(function () { setTimeout(go, 0); });
    setTimeout(go, PAINT_TIMEOUT_MS);
  }

  function run(job) {
    try {
      job.fn(job.el);
    } catch (e) {
      // One broken widget must not strand the rest of the queue.
      if (window.console) window.console.error("photonn: mounting #" + job.id + " failed", e);
    }
    job.el.classList.remove("pm-pending");
  }

  function drain() {
    if (draining || !painted) return;
    const job = queue.shift();
    if (!job) return;
    draining = true;
    idle(function () {
      run(job);
      draining = false;
      drain();
    });
  }

  function enqueue(job) {
    queue.push(job);
    drain();
  }

  /** Hold a job until its element is within NEAR_MARGIN of the viewport. */
  function whenNear(job) {
    if (typeof window.IntersectionObserver !== "function") return enqueue(job);
    const io = new window.IntersectionObserver(function (entries) {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        io.disconnect();
        enqueue(job);
      }
    }, { rootMargin: NEAR_MARGIN });
    io.observe(job.el);
  }

  function schedule(id, fn, opts) {
    const el = document.getElementById(id);
    if (!el) return;
    injectStyle();
    el.classList.add("pm-pending");
    const job = { id: id, el: el, fn: fn };
    if (opts && opts.defer) whenNear(job);
    else enqueue(job);
  }

  function mount(id, fn, opts) {
    if (document.readyState === "loading") {
      window.addEventListener("DOMContentLoaded", function () { schedule(id, fn, opts); });
    } else {
      schedule(id, fn, opts);
    }
    if (!armed) {
      armed = true;
      afterFirstPaint(function () { painted = true; drain(); });
    }
  }

  if (typeof window !== "undefined") window.PhotonnMount = mount;
  if (typeof module !== "undefined") module.exports = { mount: mount, NEAR_MARGIN: NEAR_MARGIN };
})();
