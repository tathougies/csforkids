import { basicSetup } from 'codemirror';
import { EditorState, StateField, StateEffect } from '@codemirror/state';
import { EditorView, Decoration, keymap, showTooltip } from '@codemirror/view';
import { defaultKeymap } from '@codemirror/commands';
import { python } from '@codemirror/lang-python';
import { linter, setDiagnostics } from "@codemirror/lint";

const diagnosticsField = StateField.define({
  create: () => [],
  update: (value, tr) => tr.annotation(setDiagnostics) ?? value,
});

const pythonSyntaxLint = [
  diagnosticsField,
  linter(view => view.state.field(diagnosticsField))
];


const errorLineDeco = Decoration.line({
  attributes: { class: "cm-error-line" }
});

const setErrorLine = StateEffect.define();
const setErrorTooltip = StateEffect.define()
const errorLineField = StateField.define({
  create() {
    return Decoration.none;
  },
  update(decos, tr) {
    const newErrors = tr.effects.filter((e) => e.is(setErrorLine));
    if ( newErrors.length > 0 )
      return Decoration.set(newErrors.map((e) => errorLineDeco.range(e.value)));
    else if ( tr.docChanged )
      return decos.map(tr.changes);
    else
      return decos;
  },
  provide: f => EditorView.decorations.from(f)
});

const decisionLineDeco = Decoration.line({
  attributes: { class: "duckbot-decision-line" }
});
const finalLineDeco = Decoration.line({
  attributes: { class: "duckbot-final-line" }
});
const executedLineDeco = Decoration.line({
  attributes: { class: "duckbot-executed-line" }
})
const setExecutedLine = StateEffect.define();
const executionLineField = StateField.define({
  create() {
    return Decoration.none;
  },
  update(decos, tr) {
    const newErrors = tr.effects.filter((e) => e.is(setExecutedLine));
    if ( newErrors.length > 0 ) {
      return Decoration.set(newErrors.map((e) => {
        switch (e.value.type) {
        case 'EXECUTED':
          return executedLineDeco.range(e.value.range);
        case 'DECISION':
          return decisionLineDeco.range(e.value.range);
        default:
        case 'FINAL':
          return finalLineDeco.range(e.value.range);
        }
      }));
    } else if ( tr.docChanged )
      return decos.map(tr.changes);
    else
      return decos;
  },
  provide: f => EditorView.decorations.from(f)
});

const errorTooltipField = StateField.define({
  create() {
    return [];
  },

  update(decos, tr) {
    return tr.effects.filter((e) => e.is(setErrorTooltip)).map((e) => {
      const message = e.value.msg;
      return {
        pos: e.value.from,
        to: e.value.to,
        above: true,
        strictSide: true,
        arrow: true,
        create: () => {
          const dom = document.createElement("div");
          dom.className = "cm-error-tooltip";
          dom.textContent = message;
          return { dom };
        }
      }
    });
  },

  provide(f) {
    return showTooltip.computeN([f], state => state.field(f))
  }
});

window.newCodeMirror = function newCodeMirror(parent) {
  // Define the editor state with extensions
  let startState = EditorState.create({
    doc: '',
    extensions: [
      basicSetup,
      keymap.of(defaultKeymap),
      python(), // Enable JavaScript language support and syntax highlighting
      pythonSyntaxLint,
      errorLineField,
      errorTooltipField,
      executionLineField
    ],
  });

  // Create the editor view and append it to the document body
  let view = new EditorView({
    state: startState,
    parent,
  });

  return view;
}

function pythonLocToRange(view, err) {
    const doc = view.state.doc;

    // Guard: line numbers outside document
    const lineNo = Math.min(Math.max(err.lineno ?? 1, 1), doc.lines);
    const line = doc.line(lineNo);

    // Python offset is usually 1-based; clamp into [0, line.length]
    const startCol0 = Math.min(Math.max((err.offset ?? 1) - 1, 0), line.length);
    let from = line.from + startCol0;

    // If Python provides an end range, use it; otherwise mark 1 char (or to line end)
    let to = Math.min(from + 1, line.to);

    if (err.end_lineno != null && err.end_offset != null) {
      const endLineNo = Math.min(Math.max(err.end_lineno, 1), doc.lines);
      const endLine = doc.line(endLineNo);
      const endCol0 = Math.min(Math.max(err.end_offset - 1, 0), endLine.length);
      to = endLine.from + endCol0;

      // Ensure non-empty range
      if (to <= from) to = Math.min(from + 1, endLine.to);
    }

    // Final clamp
    from = Math.min(Math.max(from, 0), doc.length);
    to = Math.min(Math.max(to, from), doc.length);

    return { from, to };
};

window.editorUtil = {
  pythonLocToRange,
  clearSyntaxErrors: (view) => {
    view.dispatch(setDiagnostics(view.state, []));
  },
  setThoughtPath: (view, executed, decisions, finals) => {
    const doc = view.state.doc;
    const executedEffects = executed.map((e) => setExecutedLine.of({ type: 'EXECUTED', range: doc.line(e).from}));
    const decisionEffects = decisions.map((e) => setExecutedLine.of({ type: 'DECISION', range: doc.line(e).from}));
    const finalEffects = finals.map((e) => setExecutedLine.of({ type: 'FINAL', range: doc.line(e).from}));
    const effects = [ ...executedEffects, ...decisionEffects, ...finalEffects ];
    effects.sort((a, b) => {
      if ( a.value.range < b.value.range ) return -1;
      else if ( a.value.range == b.value.range ) return 0;
      else return 1;
    });
    view.dispatch({effects});
  },
  showSyntaxError: (view, pyErr) => {
    const { from, to } = pythonLocToRange(view, pyErr);
    const line = view.state.doc.lineAt(from);

    const diag = [{
      from,
      to,
      severity: "error",
      message: pyErr.msg || "SyntaxError",
      // Optional: show the raw error line/trace etc
      // actions: [{ name: "Fix", apply(view, from, to) { ... } }]
    }];

    view.dispatch({
      annotations: setDiagnostics(view.state, diag),
      effects: [
        setErrorLine.of(line.from),
        setErrorTooltip.of({ from, to, msg: pyErr.msg || "SyntaxEror"})
      ]
    });
  }
};
