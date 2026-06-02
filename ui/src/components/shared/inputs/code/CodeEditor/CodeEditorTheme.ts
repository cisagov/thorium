import { EditorView, Decoration, type DecorationSet, ViewPlugin, type ViewUpdate } from '@codemirror/view';
import { HighlightStyle, syntaxHighlighting, syntaxTree } from '@codemirror/language';
import { tags } from '@lezer/highlight';

export const thoriumEditorTheme = EditorView.theme({
  '&': {
    backgroundColor: 'var(--thorium-secondary-panel-bg)',
    color: 'var(--thorium-text)',
    border: '1px solid var(--thorium-panel-border)',
    borderRadius: '4px',
    fontSize: '13px',
    fontFamily: 'monospace',
  },
  '&.cm-focused': {
    outline: '2px solid var(--thorium-info-secondary-bg)',
    outlineOffset: '-1px',
  },
  '.cm-content': {
    caretColor: 'var(--thorium-text)',
    padding: '8px 0',
  },
  '.cm-gutters': {
    backgroundColor: 'var(--thorium-panel-bg)',
    color: 'var(--thorium-secondary-text)',
    borderRight: '1px solid var(--thorium-panel-border)',
    minWidth: '40px',
  },
  '.cm-activeLineGutter': {
    backgroundColor: 'var(--thorium-highlight-panel-bg)',
  },
  '.cm-activeLine': {
    backgroundColor: 'var(--thorium-highlight-panel-bg)',
  },
  '.cm-cursor, .cm-dropCursor': {
    borderLeftColor: 'var(--thorium-text)',
  },
  '.cm-selectionBackground': {
    backgroundColor: 'var(--thorium-highlight-panel-bg) !important',
  },
  '&.cm-focused .cm-selectionBackground': {
    backgroundColor: 'var(--thorium-info-secondary-bg) !important',
    opacity: '0.3',
  },
  '.cm-panels': {
    backgroundColor: 'var(--thorium-panel-bg)',
    color: 'var(--thorium-text)',
  },
  '.cm-tooltip': {
    backgroundColor: 'var(--thorium-panel-bg)',
    color: 'var(--thorium-text)',
    border: '1px solid var(--thorium-panel-border)',
    borderRadius: '6px',
    overflow: 'hidden',
  },
  '.cm-tooltip-lint': {
    maxWidth: '500px',
  },
  '.cm-diagnostic': {
    whiteSpace: 'pre-wrap',
    wordWrap: 'break-word',
  },
  '.cm-tooltip-autocomplete': {
    '& > ul > li[aria-selected]': {
      backgroundColor: 'var(--thorium-highlight-panel-bg)',
    },
  },
  '.cm-searchMatch': {
    backgroundColor: 'var(--thorium-warning-bg)',
    opacity: '0.3',
  },
  '.cm-searchMatch.cm-searchMatch-selected': {
    backgroundColor: 'var(--thorium-info-secondary-bg)',
    opacity: '0.4',
  },
  '.cm-diagnostic-error': {
    borderLeft: '4px solid var(--thorium-danger-bg)',
    borderBottom: '2px solid var(--thorium-danger-bg)',
  },
  '.cm-diagnostic-warning': {
    borderLeft: '4px solid var(--thorium-warning-bg)',
    borderBottom: '2px solid var(--thorium-warning-bg)',
  },
  '.cm-diagnostic-info': {
    borderLeft: '4px solid var(--thorium-info-secondary-bg)',
    borderBottom: '2px solid var(--thorium-info-secondary-bg)',
  },
  '.cm-lintRange': {
    backgroundImage: 'none',
    textDecoration: 'underline wavy',
    textDecorationThickness: '2px',
    textUnderlineOffset: '3px',
    textDecorationSkipInk: 'none',
  },
  '.cm-lintRange-error': {
    textDecorationColor: 'var(--thorium-danger-bg)',
  },
  '.cm-lintRange-warning': {
    textDecorationColor: 'var(--thorium-warning-bg)',
  },
  '.cm-lintRange-info': {
    textDecorationColor: 'var(--thorium-info-secondary-bg)',
  },
  '.cm-lintRange-error.cm-lint-has-warning, .cm-lintRange-error.cm-lint-has-info': {
    textDecorationColor: 'var(--thorium-lint-overlap)',
  },
  '.cm-lintRange-warning.cm-lint-has-info': {
    textDecorationColor: 'var(--thorium-lint-overlap)',
  },
  '.cm-lint-marker-error': {
    content: '"!"',
  },
  '.cm-lint-marker-warning': {
    content: '"?"',
  },
  '.cm-line-strikethrough': {
    textDecoration: 'line-through',
    opacity: '0.5',
  },
});

const thoriumHighlightStyle = HighlightStyle.define([
  { tag: tags.keyword, color: 'var(--thorium-link-text)' },
  { tag: tags.string, color: 'var(--thorium-ok-bg)' },
  { tag: tags.number, color: 'var(--thorium-warning-bg)' },
  { tag: tags.bool, color: 'var(--thorium-warning-bg)' },
  { tag: tags.null, color: 'var(--thorium-secondary-text)' },
  { tag: tags.comment, color: 'var(--thorium-secondary-text)', fontStyle: 'italic' },
  { tag: tags.propertyName, color: 'var(--thorium-link-text-alt)' },
  { tag: tags.punctuation, color: 'var(--thorium-secondary-text)' },
  { tag: tags.meta, color: 'var(--thorium-highlight-text)' },
  { tag: tags.atom, color: 'var(--thorium-warning-bg)' },
  { tag: tags.definition(tags.variableName), color: 'var(--thorium-link-text-alt)' },
  { tag: tags.typeName, color: 'var(--thorium-link-text)' },
  { tag: tags.content, color: 'var(--thorium-ok-bg)' },
  { tag: tags.separator, color: 'var(--thorium-secondary-text)' },
]);

export const thoriumHighlighting = syntaxHighlighting(thoriumHighlightStyle);

// YAML parser treats all unquoted values as Literal/tags.content.
// This plugin post-processes to apply number/bool/null styling.
const YAML_NUM_RE = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$|^0x[0-9a-fA-F]+$|^0o[0-7]+$|^[+-]?(\.inf|\.Inf|\.INF)$|^\.nan|\.NaN|\.NAN$/;
const YAML_BOOL_RE = /^(true|false|True|False|TRUE|FALSE|yes|no|Yes|No|YES|NO|on|off|On|Off|ON|OFF)$/;
const YAML_NULL_RE = /^(null|Null|NULL|~)$/;

const yamlNumberDeco = Decoration.mark({ class: 'cm-yaml-number' });
const yamlBoolDeco = Decoration.mark({ class: 'cm-yaml-bool' });
const yamlNullDeco = Decoration.mark({ class: 'cm-yaml-null' });

function buildYamlValueDecorations(view: EditorView): DecorationSet {
  const ranges: { from: number; to: number; deco: Decoration }[] = [];
  const tree = syntaxTree(view.state);

  tree.iterate({
    enter(node) {
      if (node.name !== 'Literal') return;

      const parent = node.node.parent;
      if (parent?.name === 'Key') return;

      const text = view.state.doc.sliceString(node.from, node.to);
      if (YAML_NULL_RE.test(text)) {
        ranges.push({ from: node.from, to: node.to, deco: yamlNullDeco });
      } else if (YAML_BOOL_RE.test(text)) {
        ranges.push({ from: node.from, to: node.to, deco: yamlBoolDeco });
      } else if (YAML_NUM_RE.test(text)) {
        ranges.push({ from: node.from, to: node.to, deco: yamlNumberDeco });
      }
    },
  });

  if (ranges.length === 0) return Decoration.none;
  ranges.sort((a, b) => a.from - b.from);
  return Decoration.set(ranges.map((r) => r.deco.range(r.from, r.to)));
}

export const yamlValueHighlighter = ViewPlugin.fromClass(
  class {
    decorations: DecorationSet;
    constructor(view: EditorView) {
      this.decorations = buildYamlValueDecorations(view);
    }
    update(update: ViewUpdate) {
      if (update.docChanged || update.viewportChanged || syntaxTree(update.state) !== syntaxTree(update.startState)) {
        this.decorations = buildYamlValueDecorations(update.view);
      }
    }
  },
  { decorations: (v) => v.decorations },
);

export const yamlValueTheme = EditorView.theme({
  '.cm-yaml-number': { color: 'var(--thorium-warning-bg)' },
  '.cm-yaml-bool': { color: 'var(--thorium-warning-bg)' },
  '.cm-yaml-null': { color: 'var(--thorium-secondary-text)' },
});
