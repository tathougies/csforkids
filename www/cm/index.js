import { basicSetup } from 'codemirror';
import { EditorState, StateField } from '@codemirror/state';
import { EditorView, Decoration, keymap } from '@codemirror/view';
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

const decisionLineDeco = Decoration.line({
  attributes: { class: "duckbot-decision-line" }
});
const finalLineDeco = Decoration.line({
  attributes: { class: "duckbot-final-line" }
});

const errorLineDeco = Decoration.line({
  attributes: { class: "cm-error-line" }
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
//      EditorView.lineNumbers(), // Add line numbers
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
  showSyntaxError: (view, pyErr) => {
  const { from, to } = pythonLocToRange(view, pyErr);

    const diag = [{
      from,
      to,
      severity: "error",
      message: pyErr.msg || "SyntaxError",
      // Optional: show the raw error line/trace etc
      // actions: [{ name: "Fix", apply(view, from, to) { ... } }]
    }];

    view.dispatch(setDiagnostics(view.state, diag));
  }
};
