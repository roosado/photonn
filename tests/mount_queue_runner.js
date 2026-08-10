/*
 * Node runner for the widget mount scheduler (apps/web/mount_queue.js).
 *
 * The scheduler decides when every widget on a page is allowed to start, which
 * makes it the one piece of front-end code that can strand a whole page: a gate
 * that never opens leaves the reader looking at "warming up" forever. That
 * failure is invisible to the rest of the suite -- the widgets themselves are
 * fine, they are simply never called -- and it cannot be checked in the driven
 * browser either, because that tab is always hidden and never paints.
 *
 * So the scheduler is exercised here against a hand-built stand-in for the parts
 * of the DOM it touches, with time and paint under this file's control. Prints
 * one JSON object per scenario. Driven by tests/test_mount_queue.py.
 */
const path = require("path");

const SRC = path.join(__dirname, "..", "apps", "web", "mount_queue.js");

/**
 * A minimal window/document good enough for mount_queue.js, with the paint
 * signal, the idle queue and IntersectionObserver all driven by hand.
 */
function makeEnv(opts) {
  const o = opts || {};
  const timers = [];          // {at, fn}
  let now = 0;
  let rafs = [];
  const observers = [];

  const el = (id) => ({
    id: id,
    classes: new Set(),
    classList: {
      add(c) { el._all[id].classes.add(c); },
      remove(c) { el._all[id].classes.delete(c); },
      contains(c) { return el._all[id].classes.has(c); },
    },
  });
  el._all = {};
  for (const id of o.ids || []) el._all[id] = el(id);

  const doc = {
    readyState: o.readyState || "complete",
    visibilityState: o.hidden ? "hidden" : "visible",
    getElementById: (id) => el._all[id] || null,
    createElement: () => ({ set textContent(v) {}, id: "" }),
    head: { appendChild() {} },
    _styles: {},
    addEventListener(type, fn) { (doc._ev = doc._ev || {})[type] = fn; },
  };
  // injectStyle guards on getElementById(STYLE_ID); the style element is never
  // registered here, so the guard stays false and injection is attempted once.

  const win = {
    document: doc,
    console: { error() {} },
    setTimeout(fn, ms) { timers.push({ at: now + (ms || 0), fn: fn }); return timers.length; },
    clearTimeout() {},
    addEventListener(type, fn) { (win._ev = win._ev || {})[type] = fn; },
    requestAnimationFrame: o.noRaf ? undefined : function (fn) { rafs.push(fn); },
    requestIdleCallback: o.noIdle ? undefined : function (fn) { timers.push({ at: now, fn: fn }); },
    IntersectionObserver: o.noIo ? undefined : function (cb, opt) {
      const self = { cb: cb, opt: opt, targets: [], disconnected: false };
      self.observe = (t) => self.targets.push(t);
      self.disconnect = () => { self.disconnected = true; };
      observers.push(self);
      return self;
    },
  };
  win.window = win;

  return {
    win, doc, observers,
    /** Advance virtual time, running due timers (and any they schedule). */
    tick(ms) {
      now += (ms || 0);
      for (let guard = 0; guard < 10000; guard++) {
        const i = timers.findIndex((t) => t.at <= now);
        if (i < 0) return;
        const t = timers.splice(i, 1)[0];
        t.fn();
      }
      throw new Error("timer storm: scheduler kept queueing work");
    },
    /** Fire the pending animation frames, as a real paint would. */
    paint() {
      const due = rafs;
      rafs = [];
      for (const fn of due) fn();
    },
    /** Report an element as having scrolled near the viewport. */
    intersect(id) {
      for (const ob of observers) {
        for (const t of ob.targets) {
          if (t.id === id && !ob.disconnected) ob.cb([{ isIntersecting: true, target: t }]);
        }
      }
    },
  };
}

/** Load a fresh copy of the scheduler into `env`. */
function load(env) {
  const src = require("fs").readFileSync(SRC, "utf8");
  // The file is an IIFE closing over `window`, `document` and `module`. `setTimeout`
  // is passed as a parameter as well: the scheduler calls it unqualified, which in
  // a browser is window.setTimeout but under Node would reach the real event loop
  // and take the clock out of this file's hands.
  const fn = new Function("window", "document", "module", "setTimeout", src);
  const mod = { exports: {} };
  fn(env.win, env.doc, mod, env.win.setTimeout);
  return env.win.PhotonnMount;
}

const out = {};

// 1. Nothing runs before the first paint; then jobs run one at a time, in order.
{
  const env = makeEnv({ ids: ["a", "b", "c"] });
  const mount = load(env);
  const order = [];
  mount("a", () => order.push("a"));
  mount("b", () => order.push("b"));
  mount("c", () => order.push("c"));
  env.tick(0);
  const beforePaint = order.slice();
  env.paint();
  env.tick(0);
  const afterOneDrain = order.slice();
  env.tick(50);
  out.ordering = {
    beforePaint: beforePaint,
    afterFirstDrain: afterOneDrain,
    final: order,
    pendingClassCleared: !env.doc.getElementById("a").classList.contains("pm-pending"),
  };
}

// 2. A hidden document never paints, so it must not wait for one.
{
  const env = makeEnv({ ids: ["a"], hidden: true });
  const mount = load(env);
  const order = [];
  mount("a", () => order.push("a"));
  env.tick(10);                     // no paint() call at all
  out.hiddenRunsAnyway = order;
}

// 3. No rAF at all (very old or exotic host): still runs.
{
  const env = makeEnv({ ids: ["a"], noRaf: true });
  const mount = load(env);
  const order = [];
  mount("a", () => order.push("a"));
  env.tick(10);
  out.noRafRunsAnyway = order;
}

// 4. A paint that never arrives must not strand the page forever.
{
  const env = makeEnv({ ids: ["a"] });
  const mount = load(env);
  const order = [];
  mount("a", () => order.push("a"));
  env.tick(500);
  const at500 = order.slice();
  env.tick(2500);
  out.paintTimeout = { at500: at500, at3000: order.slice() };
}

// 5. A deferred job waits for its element to come near, then runs.
{
  const env = makeEnv({ ids: ["near", "far"] });
  const mount = load(env);
  const order = [];
  mount("near", () => order.push("near"));
  mount("far", () => order.push("far"), { defer: true });
  env.paint();
  env.tick(10);
  const beforeScroll = order.slice();
  env.intersect("far");
  env.tick(10);
  out.deferred = {
    beforeScroll: beforeScroll,
    afterScroll: order.slice(),
    rootMargin: env.observers.length ? env.observers[0].opt.rootMargin : null,
  };
}

// 6. Without IntersectionObserver, a deferred job must still run.
{
  const env = makeEnv({ ids: ["far"], noIo: true });
  const mount = load(env);
  const order = [];
  mount("far", () => order.push("far"), { defer: true });
  env.paint();
  env.tick(10);
  out.deferredWithoutIo = order;
}

// 7. One widget throwing must not strand the ones behind it.
{
  const env = makeEnv({ ids: ["bad", "good"] });
  const mount = load(env);
  const order = [];
  mount("bad", () => { order.push("bad"); throw new Error("boom"); });
  mount("good", () => order.push("good"));
  env.paint();
  env.tick(10);
  out.survivesThrow = order;
}

// 8. An id that is not on the page is skipped, not queued forever.
{
  const env = makeEnv({ ids: ["real"] });
  const mount = load(env);
  const order = [];
  mount("missing", () => order.push("missing"));
  mount("real", () => order.push("real"));
  env.paint();
  env.tick(10);
  out.missingIdSkipped = order;
}

process.stdout.write(JSON.stringify(out));
