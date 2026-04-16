import { EditorView, Decoration, type DecorationSet, WidgetType } from '@codemirror/view';
import { EditorState, StateField, StateEffect } from '@codemirror/state';
import { FormatType } from '@utilities/rules/types';

export interface PreviewProposal {
  field: string;
  value: string;
  format: FormatType;
  cursorLine?: number;
  isList?: boolean;
}

export const addPreview = StateEffect.define<PreviewProposal>();
export const clearPreview = StateEffect.define<void>();
export const acceptPreview = StateEffect.define<void>();
const updateInsertText = StateEffect.define<string>();

function lineOffset(lines: string[], lineIdx: number): number {
  let offset = 0;
  for (let k = 0; k < lineIdx; k++) {
    offset += lines[k].length + 1;
  }
  return offset;
}

function buildYaraInsertText(
  field: string,
  value: string,
  docText: string,
  lines: string[],
  cursorLine?: number,
): { text: string; pos: number; inline?: boolean } {
  if (field.startsWith('section.')) {
    const sectionName = field.slice('section.'.length);
    const sectionOrder = ['meta', 'strings', 'condition'];
    const sectionIdx = sectionOrder.indexOf(sectionName);

    const scaffolds: Record<string, string> = {
      meta: '    meta:\n        description = ""\n        author = ""\n',
      strings: '    strings:\n        $s1 = ""\n',
      condition: '    condition:\n        true\n',
    };

    let insertLineIdx = -1;
    for (let s = sectionIdx + 1; s < sectionOrder.length; s++) {
      const re = new RegExp(`^\\s*${sectionOrder[s]}\\s*:`);
      for (let i = 0; i < lines.length; i++) {
        if (re.test(lines[i])) {
          insertLineIdx = i;
          break;
        }
      }
      if (insertLineIdx >= 0) break;
    }

    if (insertLineIdx < 0) {
      for (let i = lines.length - 1; i >= 0; i--) {
        if (lines[i].trim() === '}') {
          insertLineIdx = i;
          break;
        }
      }
    }

    if (insertLineIdx >= 0) {
      const hasBlankBefore = insertLineIdx > 0 && lines[insertLineIdx - 1].trim() === '';
      const scaffold = scaffolds[sectionName] ?? `    ${sectionName}:\n`;
      const insertText = (hasBlankBefore ? '' : '\n') + scaffold + '\n';
      const pos = Math.min(lineOffset(lines, insertLineIdx), docText.length);
      return { text: insertText, pos };
    }
    const insertText = '\n' + (scaffolds[sectionName] ?? `    ${sectionName}:\n`);
    return { text: insertText, pos: docText.length };
  }

  if (field === 'meta' || field.startsWith('meta.')) {
    const metaKey = field === 'meta' ? value : field.split('.').slice(1).join('.');
    const metaVal = field === 'meta' ? `"<value>"` : `"${value}"`;
    let metaIdx = -1;
    let insertAfterIdx = -1;
    for (let i = 0; i < lines.length; i++) {
      if (/^\s*meta\s*:/.test(lines[i])) {
        metaIdx = i;
        insertAfterIdx = i;
        for (let j = i + 1; j < lines.length; j++) {
          if (/^\s+\S/.test(lines[j]) && !/^\s*(strings|condition)\s*:/.test(lines[j])) {
            insertAfterIdx = j;
          } else {
            break;
          }
        }
        break;
      }
    }
    if (metaIdx >= 0) {
      const indent = lines[insertAfterIdx].match(/^(\s*)/)?.[1] ?? '        ';
      const lineEnd = Math.min(lineOffset(lines, insertAfterIdx) + lines[insertAfterIdx].length, docText.length);
      return { text: `\n${indent}${metaKey} = ${metaVal}`, pos: lineEnd };
    }
    return { text: `\n        ${metaKey} = ${metaVal}`, pos: docText.length };
  }

  if (field === 'import') {
    return { text: `import "${value}"\n`, pos: 0 };
  }

  if (field === 'strings.modifiers' && cursorLine != null) {
    const lineIdx = Math.min(cursorLine - 1, lines.length - 1);
    const lineEnd = Math.min(lineOffset(lines, lineIdx + 1) - 1, docText.length);
    return { text: ` ${value}`, pos: lineEnd, inline: true };
  }

  return { text: `${value}\n`, pos: docText.length };
}

function buildYamlInsertText(
  field: string,
  value: string,
  docText: string,
  lines: string[],
  isList?: boolean,
): { text: string; pos: number } {
  const parts = field.split('.');
  if (parts.length === 1) {
    const trailing = docText.endsWith('\n') ? '' : '\n';
    if (isList) {
      return {
        text: `${trailing}${field}:\n    - '${value}'\n`,
        pos: docText.length,
      };
    }
    return {
      text: `${trailing}${field}: '${value}'\n`,
      pos: docText.length,
    };
  }

  const parentKey = parts[0];
  const childKey = parts.slice(1).join('.');

  let parentLineIdx = -1;
  let insertAfterIdx = -1;
  for (let i = 0; i < lines.length; i++) {
    const trimmed = lines[i].trimStart();
    if (trimmed.startsWith(`${parentKey}:`) || trimmed.startsWith(`${parentKey} :`)) {
      parentLineIdx = i;
      insertAfterIdx = i;
      for (let j = i + 1; j < lines.length; j++) {
        if (lines[j].match(/^\s/) && !lines[j].match(/^\S/)) {
          insertAfterIdx = j;
        } else if (lines[j].trim() === '') {
          insertAfterIdx = j;
        } else {
          break;
        }
      }
      break;
    }
  }

  if (parentLineIdx >= 0) {
    const pos = Math.min(lineOffset(lines, insertAfterIdx + 1), docText.length);
    const needsNewline = insertAfterIdx < lines.length - 1 || !docText.endsWith('\n');
    return {
      text: `    ${childKey}: '${value}'${needsNewline ? '\n' : ''}`,
      pos,
    };
  }

  const trailing = docText.endsWith('\n') ? '' : '\n';
  return {
    text: `${trailing}${parentKey}:\n    ${childKey}: '${value}'\n`,
    pos: docText.length,
  };
}

function buildInsertText(
  field: string,
  value: string,
  docText: string,
  format: FormatType,
  cursorLine?: number,
  isList?: boolean,
): { text: string; pos: number; inline?: boolean } {
  const lines = docText.split('\n');

  switch (format) {
    case FormatType.JSON:
      return { text: `"${field}": "${value}"`, pos: docText.length };
    case FormatType.YARA:
      return buildYaraInsertText(field, value, docText, lines, cursorLine);
    default:
      return buildYamlInsertText(field, value, docText, lines, isList);
  }
}

const setViewRef = StateEffect.define<EditorView>();

const EDITABLE_RE = /(=\s*"|:\s*'|-\s*')([^"']*)("|')/;

function findEditableRange(text: string): { before: string; value: string; after: string; quote: string } | null {
  const trimmed = text.trim();
  const m = trimmed.match(EDITABLE_RE);
  if (!m) return null;
  const matchStart = m.index!;
  const before = trimmed.slice(0, matchStart + m[1].length);
  const value = m[2];
  const after = trimmed.slice(matchStart + m[1].length + m[2].length);
  return { before, value, after, quote: m[3] };
}

function makeBtn(label: string, title: string, bgVar: string, onClick: () => void): HTMLButtonElement {
  const btn = document.createElement('button');
  btn.textContent = label;
  btn.title = title;
  btn.style.cssText = [
    `background-color: var(${bgVar})`,
    'color: var(--thorium-button-text)',
    'border: none',
    'border-radius: 4px',
    'padding: 2px 10px',
    'cursor: pointer',
    'font-size: 11px',
    'font-weight: 600',
    'transition: filter 0.15s',
  ].join(';');
  btn.addEventListener('mouseenter', () => {
    btn.style.filter = 'brightness(1.15)';
  });
  btn.addEventListener('mouseleave', () => {
    btn.style.filter = '';
  });
  btn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    onClick();
  });
  return btn;
}

class PreviewWidget extends WidgetType {
  constructor(
    readonly insertText: string,
    readonly viewRef: { current: EditorView | null },
    readonly inline: boolean = false,
  ) {
    super();
  }

  eq(other: PreviewWidget): boolean {
    return this.insertText === other.insertText && this.inline === other.inline;
  }

  toDOM(): HTMLElement {
    if (this.inline) return this.toInlineDOM();

    const firstLine = this.insertText.split('\n').find((l) => l.trim().length > 0) ?? this.insertText;
    const leadingSpaces = firstLine.match(/^(\s*)/)?.[1].length ?? 0;
    const marginLeft = `${leadingSpaces}ch`;

    const wrapper = document.createElement('div');
    wrapper.className = 'cm-suggestion-preview';
    wrapper.style.cssText = [
      'display: flex',
      'flex-wrap: wrap',
      'align-items: flex-end',
      'gap: 8px',
      'padding: 4px 10px',
      `margin: 0 4px 0 ${marginLeft}`,
      'background-color: var(--thorium-panel-bg)',
      'border: 1px solid var(--thorium-panel-border)',
      'border-left: 3px solid var(--thorium-info-secondary-bg)',
      'border-radius: 6px',
      'font-family: monospace',
      'font-size: 13px',
      'box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15)',
    ].join(';');

    const editable = findEditableRange(this.insertText);
    const contentContainer = document.createElement('span');
    contentContainer.style.cssText = ['flex: 1 1 auto', 'min-width: 0'].join(';');

    if (editable) {
      const labelStyle = ['color: var(--thorium-info-secondary-bg)', 'font-style: italic', 'white-space: pre', 'line-height: 20px'].join(
        ';',
      );

      const trimmedText = this.insertText.trim();
      const editMatch = trimmedText.match(EDITABLE_RE);
      const editLineIdx = editMatch ? trimmedText.slice(0, editMatch.index!).split('\n').length - 1 : 0;
      const allLines = trimmedText.split('\n');

      const headerLines = allLines.slice(0, editLineIdx);
      if (headerLines.length > 0) {
        const headerSpan = document.createElement('div');
        headerSpan.style.cssText = [
          'color: var(--thorium-info-secondary-bg)',
          'font-style: italic',
          'white-space: pre',
          'line-height: 20px',
        ].join(';');
        headerSpan.textContent = headerLines.join('\n');
        contentContainer.appendChild(headerSpan);
      }

      const editableRow = document.createElement('div');
      editableRow.style.cssText = ['display: flex', 'align-items: baseline', 'gap: 0'].join(';');

      const prefixSpan = document.createElement('span');
      prefixSpan.style.cssText = labelStyle;
      const editableLine = allLines[editLineIdx] ?? '';
      const lineMatch = editableLine.match(EDITABLE_RE);
      prefixSpan.textContent = lineMatch ? editableLine.slice(0, lineMatch.index! + lineMatch[1].length) : editable.before;

      const input = document.createElement('input');
      input.type = 'text';
      input.value = editable.value === '<value>' ? '' : editable.value;
      input.placeholder = 'value';
      input.style.cssText = [
        'background: var(--thorium-highlight-panel-bg)',
        'color: var(--thorium-text)',
        'border: 1px solid var(--thorium-panel-border)',
        'border-radius: 3px',
        'padding: 0 4px',
        'font-family: monospace',
        'font-size: 13px',
        'line-height: 18px',
        'height: 20px',
        'box-sizing: border-box',
        'min-width: 80px',
        'max-width: 300px',
        'outline: none',
        'vertical-align: baseline',
      ].join(';');

      const suffixSpan = document.createElement('span');
      suffixSpan.style.cssText = labelStyle;
      suffixSpan.textContent = editable.after;

      const rebuildText = () => {
        const indent = this.insertText.match(/^(\s*)/)?.[1] ?? '';
        const rest = this.insertText.trimStart();
        const m = rest.match(EDITABLE_RE);
        if (m) {
          const newRest = rest.slice(0, m.index! + m[1].length) + input.value + rest.slice(m.index! + m[1].length + m[2].length);
          return indent + newRest;
        }
        return this.insertText;
      };

      input.addEventListener('input', () => {
        this.viewRef.current?.dispatch({ effects: updateInsertText.of(rebuildText()) });
      });

      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          e.stopPropagation();
          this.viewRef.current?.dispatch({ effects: updateInsertText.of(rebuildText()) });
          setTimeout(() => this.viewRef.current?.dispatch({ effects: acceptPreview.of() }), 0);
        }
        if (e.key === 'Escape') {
          e.preventDefault();
          e.stopPropagation();
          this.viewRef.current?.dispatch({ effects: clearPreview.of() });
        }
      });

      editableRow.appendChild(prefixSpan);
      editableRow.appendChild(input);
      editableRow.appendChild(suffixSpan);
      contentContainer.appendChild(editableRow);

      const footerLines = allLines.slice(editLineIdx + 1).filter((l) => l.trim().length > 0);
      if (footerLines.length > 0) {
        const footerSpan = document.createElement('div');
        footerSpan.style.cssText = [
          'color: var(--thorium-info-secondary-bg)',
          'font-style: italic',
          'white-space: pre',
          'line-height: 20px',
          'opacity: 0.7',
        ].join(';');
        footerSpan.textContent = footerLines.join('\n');
        contentContainer.appendChild(footerSpan);
      }

      setTimeout(() => input.focus(), 0);
    } else {
      const textSpan = document.createElement('span');
      textSpan.style.cssText = [
        'color: var(--thorium-info-secondary-bg)',
        'font-style: italic',
        'white-space: pre-wrap',
        'word-break: break-word',
      ].join(';');
      textSpan.textContent = this.insertText.trim();
      contentContainer.appendChild(textSpan);
    }

    const btnContainer = document.createElement('span');
    btnContainer.style.cssText = ['display: flex', 'gap: 6px', 'flex-shrink: 0'].join(';');

    const acceptBtn = makeBtn('Accept', 'Insert this field (Enter)', '--thorium-ok-bg', () => {
      this.viewRef.current?.dispatch({ effects: acceptPreview.of() });
    });
    const dismissBtn = makeBtn('Dismiss', 'Cancel (Escape)', '--thorium-info-bg', () => {
      this.viewRef.current?.dispatch({ effects: clearPreview.of() });
    });

    btnContainer.appendChild(acceptBtn);
    btnContainer.appendChild(dismissBtn);
    wrapper.appendChild(contentContainer);
    wrapper.appendChild(btnContainer);
    return wrapper;
  }

  private toInlineDOM(): HTMLElement {
    const wrapper = document.createElement('span');
    wrapper.className = 'cm-suggestion-preview';
    wrapper.style.cssText = [
      'display: inline-flex',
      'align-items: center',
      'gap: 6px',
      'padding: 1px 8px',
      'margin-left: 8px',
      'background-color: var(--thorium-panel-bg)',
      'border: 1px solid var(--thorium-panel-border)',
      'border-left: 3px solid var(--thorium-info-secondary-bg)',
      'border-radius: 4px',
      'font-family: monospace',
      'font-size: 13px',
      'vertical-align: baseline',
    ].join(';');

    const textSpan = document.createElement('span');
    textSpan.style.cssText = ['color: var(--thorium-info-secondary-bg)', 'font-style: italic', 'white-space: pre'].join(';');
    textSpan.textContent = this.insertText.trim();
    wrapper.appendChild(textSpan);

    const btnContainer = document.createElement('span');
    btnContainer.style.cssText = ['display: inline-flex', 'gap: 4px', 'flex-shrink: 0'].join(';');

    const acceptBtn = makeBtn('Accept', 'Insert (Enter)', '--thorium-ok-bg', () => {
      this.viewRef.current?.dispatch({ effects: acceptPreview.of() });
    });
    const dismissBtn = makeBtn('Dismiss', 'Cancel (Escape)', '--thorium-info-bg', () => {
      this.viewRef.current?.dispatch({ effects: clearPreview.of() });
    });

    btnContainer.appendChild(acceptBtn);
    btnContainer.appendChild(dismissBtn);
    wrapper.appendChild(btnContainer);
    return wrapper;
  }

  ignoreEvent(): boolean {
    return true;
  }
}

interface ProposalState {
  proposal: PreviewProposal | null;
  insertText: string;
  insertPos: number;
  inline: boolean;
}

export const previewState = StateField.define<ProposalState>({
  create() {
    return { proposal: null, insertText: '', insertPos: 0, inline: false };
  },

  update(state, tr) {
    const empty: ProposalState = { proposal: null, insertText: '', insertPos: 0, inline: false };
    for (const effect of tr.effects) {
      if (effect.is(addPreview)) {
        const docText = tr.state.doc.toString();
        const result = buildInsertText(
          effect.value.field,
          effect.value.value,
          docText,
          effect.value.format,
          effect.value.cursorLine,
          effect.value.isList,
        );
        return { proposal: effect.value, insertText: result.text, insertPos: result.pos, inline: result.inline ?? false };
      }
      if (effect.is(clearPreview)) return empty;
      if (effect.is(acceptPreview)) return empty;
      if (effect.is(updateInsertText)) {
        return { ...state, insertText: effect.value };
      }
    }
    if (tr.docChanged && state.proposal) return empty;
    return state;
  },
});

const previewDecoField = StateField.define<{ decos: DecorationSet; viewRef: { current: EditorView | null } }>({
  create() {
    return { decos: Decoration.none, viewRef: { current: null } };
  },

  update(state, tr) {
    for (const effect of tr.effects) {
      if (effect.is(setViewRef)) {
        return { ...state, viewRef: { current: effect.value } };
      }
    }

    const prev = tr.startState.field(previewState);
    const curr = tr.state.field(previewState);
    if (prev === curr) return state;

    if (!curr.proposal) {
      return { ...state, decos: Decoration.none };
    }

    const iPos = Math.min(curr.insertPos, tr.state.doc.length);
    const widget = new PreviewWidget(curr.insertText, state.viewRef, curr.inline);

    if (curr.inline) {
      const widgetDeco = Decoration.widget({ widget, side: 1, block: false }).range(iPos);
      return { ...state, decos: Decoration.set([widgetDeco]) };
    }

    const doc = tr.state.doc;
    const line = doc.lineAt(iPos);
    const side = iPos === doc.length || iPos !== line.from ? 1 : -1;
    const widgetDeco = Decoration.widget({ widget, side, block: true }).range(iPos);
    return {
      ...state,
      decos: Decoration.set([widgetDeco]),
    };
  },

  provide(field) {
    return EditorView.decorations.from(field, (val) => val.decos);
  },
});

export function handleAcceptEffect(view: EditorView): boolean {
  const state = view.state.field(previewState);
  if (!state.proposal) return false;

  view.dispatch({ effects: acceptPreview.of() });
  return true;
}

export function createPreviewExtensions() {
  const acceptFilter = EditorState.transactionFilter.of((tr) => {
    for (const effect of tr.effects) {
      if (effect.is(acceptPreview)) {
        const state = tr.startState.field(previewState);
        if (state.proposal) {
          return [
            tr,
            {
              changes: { from: state.insertPos, insert: state.insertText },
            },
          ];
        }
      }
    }
    return tr;
  });

  const keyHandler = EditorView.domEventHandlers({
    keydown(event, view) {
      const state = view.state.field(previewState);
      if (!state.proposal) return false;

      const target = event.target as HTMLElement;
      if (target.closest('.cm-suggestion-preview')) return false;

      if (event.key === 'Enter') {
        event.preventDefault();
        view.dispatch({ effects: acceptPreview.of() });
        return true;
      }
      if (event.key === 'Escape') {
        event.preventDefault();
        view.dispatch({ effects: clearPreview.of() });
        return true;
      }
      return false;
    },
  });

  const viewRefPlugin = EditorView.updateListener.of((update) => {
    const field = update.state.field(previewDecoField);
    if (!field.viewRef.current) {
      update.view.dispatch({ effects: setViewRef.of(update.view) });
    }
  });

  return [previewState, previewDecoField, acceptFilter, keyHandler, viewRefPlugin];
}
