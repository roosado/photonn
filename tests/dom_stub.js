/*
 * The DOM stand-in the widget runners mount against.
 *
 * There is no jsdom here, and the driven Chrome tab cannot stand in for one --
 * that tab is always hidden, so it delivers no observer callbacks and screenshots
 * as a blank field. What the widgets in apps/web/ actually touch is small and
 * boring: create an element, set a class, append a child, measure a width. So the
 * runners hand-build that much.
 *
 * The two parts that genuinely differ per widget are parameters, not copies:
 *
 *   layoutWidth(node)  the layout model. errors.js needs a mini-flexbox that
 *                      parses its own stylesheet; interfere.js needs one capped
 *                      pane. This is the part each runner is really testing.
 *   ctxStub()          the canvas 2D methods that widget calls. Deliberately not
 *                      a union of everything: a widget reaching for a method its
 *                      runner did not declare should throw, not silently no-op.
 *
 * Everything else -- the element factory, the document, the window -- was
 * duplicated line for line between tests/error_widget_runner.js and
 * tests/interference_runner.js before this file existed.
 */

/**
 * A window/document pair backed by stub elements.
 *
 * @param {object} opts
 * @param {number} opts.dpr          window.devicePixelRatio for this run.
 * @param {function} opts.layoutWidth  node -> the CSS width it lays out at.
 * @param {function} opts.ctxStub      () -> a 2D context stub.
 * @returns {{win: object, doc: object, makeEl: function}}
 */
function makeEnv(opts) {
  const styles = {};

  function makeEl(tag) {
    const node = {
      tagName: String(tag).toUpperCase(),
      className: "",
      id: "",
      innerHTML: "",
      textContent: "",
      children: [],
      parentNode: null,
      style: {},
      appendChild(c) { c.parentNode = node; node.children.push(c); return c; },
      setAttribute() {},
      addEventListener() {},
      getBoundingClientRect() { return { width: opts.layoutWidth(node), height: 0 }; },
    };
    if (node.tagName === "CANVAS") {
      node.width = 300; node.height = 150;
      node.getContext = () => opts.ctxStub();
    }
    return node;
  }

  const doc = {
    getElementById: (id) => styles[id] || null,
    createElement: (tag) => makeEl(tag),
    head: { appendChild(s) { if (s.id) styles[s.id] = s; } },
  };
  const win = {
    document: doc,
    devicePixelRatio: opts.dpr,
    addEventListener() {},
    // No ResizeObserver on purpose: the fallback path must work too.
  };
  return { win, doc, makeEl };
}

/**
 * Evaluate a widget source against one of these environments.
 *
 * The widgets are IIFEs that publish onto `window` and, when running under Node,
 * onto `module.exports`; `new Function` gives each one its own `window` and
 * `document` without a global. `extras` binds any further name the source expects
 * (errors.js wants `atob`).
 */
function loadWidget(src, env, extras) {
  const names = Object.keys(extras || {});
  const mod = { exports: {} };
  const fn = new Function("window", "document", "module", ...names, src);
  fn(env.win, env.doc, mod, ...names.map((k) => extras[k]));
  return mod.exports;
}

module.exports = { makeEnv, loadWidget };
