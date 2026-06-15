import React, { useRef, useEffect, useCallback, useState, useMemo } from 'react';
import { EditorView, lineNumbers, keymap } from '@codemirror/view';
import { EditorState } from '@codemirror/state';
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
import { bracketMatching, indentOnInput, foldGutter, foldKeymap, StreamLanguage } from '@codemirror/language';
import { linter, type Diagnostic as CmDiagnostic } from '@codemirror/lint';
import { yaml as yamlLang } from '@codemirror/lang-yaml';
import { json as jsonLang } from '@codemirror/lang-json';
import { cpp as cppLang } from '@codemirror/lang-cpp';
import { gas } from '@codemirror/legacy-modes/mode/gas';
import { parseDocument } from 'yaml';
import { yaraLanguage } from './yara-language';
import styled from 'styled-components';

import { thoriumEditorTheme, thoriumHighlighting, yamlValueHighlighter, yamlValueTheme } from './CodeEditorTheme';
import SuggestionPanel from './SuggestionPanel';
import { createPreviewExtensions, addPreview } from './SuggestionPreview';
import { type RuleChecker, type FieldSchema, FormatType, type Suggestion, Severity } from '@utilities/rules/types';

// spec: ./CodeEditor.spec.md

// When `height` is '100%' the editor fills its parent (flex column) instead of taking a fixed
// pixel/vh height — used by the file-preview renderers so the editor sizes to the overlay window
// and owns the single scroll for its content.
const Root = styled.div<{ $fill: boolean }>`
  ${(props) =>
    props.$fill &&
    `
    flex: 1 1 auto;
    min-height: 0;
    height: 100%;
    display: flex;
    flex-direction: column;
  `}
`;

const EditorContainer = styled.div<{ $height: string; $fill: boolean }>`
  width: 100%;
  ${(props) =>
    props.$fill &&
    `
    flex: 1 1 auto;
    min-height: 0;
  `}

  .cm-editor {
    height: ${(props) => (props.$fill ? '100%' : props.$height)};
    overflow: auto;
  }
`;

// keep the suggestion panel at its natural height beneath the (flex-filling) editor
const SuggestionSlot = styled.div`
  flex: 0 0 auto;
`;

function getLanguageExtension(format: FormatType) {
  switch (format) {
    case FormatType.YAML:
      return yamlLang();
    case FormatType.JSON:
      return jsonLang();
    case FormatType.YARA:
      return yaraLanguage();
    case FormatType.Decomp:
      // decompiled output is C-like; reuse the prebuilt C/C++ grammar for highlighting
      return cppLang();
    case FormatType.Disassembly:
      // disassembly is assembly text; the gas (GNU assembler) legacy mode highlights it well
      return StreamLanguage.define(gas);
    default:
      return [];
  }
}

function parseText(text: string, format: FormatType): unknown {
  if (!text.trim()) return null;
  try {
    switch (format) {
      case FormatType.YAML: {
        const doc = parseDocument(text);
        return doc.errors.length === 0 ? doc.toJS() : null;
      }
      case FormatType.JSON:
        return JSON.parse(text);
      case FormatType.YARA:
        return null;
      default:
        return null;
    }
  } catch {
    return null;
  }
}

function filterSuggestionsByCursor(suggestions: Suggestion[], cursorLine: number, format: FormatType): Suggestion[] {
  if (format !== FormatType.YARA) return suggestions;
  return suggestions.filter((s) => {
    if (s.lineEnd == null) return true;
    return cursorLine >= s.line && cursorLine <= s.lineEnd;
  });
}

export interface CodeEditorProps {
  value: string;
  onChange: (text: string, parsed: unknown) => void;
  checker?: RuleChecker;
  format: FormatType;
  height?: string;
  disabled?: boolean;
}

const CodeEditor: React.FC<CodeEditorProps> = ({ value, onChange, checker, format, height = '300px', disabled = false }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<EditorView | null>(null);
  const onChangeRef = useRef(onChange);
  const checkerRef = useRef(checker);
  const [allSuggestions, setAllSuggestions] = useState<Suggestion[]>([]);
  const [cursorLine, setCursorLine] = useState(1);

  onChangeRef.current = onChange;
  checkerRef.current = checker;

  const formatRef = useRef(format);
  formatRef.current = format;

  const cursorLineRef = useRef(cursorLine);
  cursorLineRef.current = cursorLine;

  const filteredSuggestions = useMemo(
    () => filterSuggestionsByCursor(allSuggestions, cursorLine, format),
    [allSuggestions, cursorLine, format],
  );

  const createLinter = useCallback(() => {
    return linter(
      (view) => {
        const text = view.state.doc.toString();
        const currentChecker = checkerRef.current;
        if (!currentChecker) {
          setAllSuggestions([]);
          return [];
        }

        currentChecker.format = formatRef.current;
        const result = currentChecker.check(text);
        setAllSuggestions(result.suggestions);

        return result.diagnostics.map((d): CmDiagnostic => {
          const fromLine = Math.max(1, Math.min(d.line, view.state.doc.lines));
          const lineInfo = view.state.doc.line(fromLine);
          const fromCol = Math.max(0, (d.column ?? 1) - 1);
          const from = Math.min(lineInfo.from + fromCol, lineInfo.to);

          let to = from;
          if (d.endLine) {
            const endLineNum = Math.max(1, Math.min(d.endLine, view.state.doc.lines));
            const endLineInfo = view.state.doc.line(endLineNum);
            to = d.endColumn ? Math.min(endLineInfo.from + d.endColumn - 1, endLineInfo.to) : endLineInfo.to;
          } else if (d.endColumn) {
            to = Math.min(lineInfo.from + d.endColumn - 1, lineInfo.to);
          } else {
            to = lineInfo.to;
          }

          if (to <= from) {
            to = Math.min(from + 1, lineInfo.to);
          }

          if (to - from <= 1) {
            const rest = view.state.doc.sliceString(from, lineInfo.to);
            const wordMatch = rest.match(/^[\w-]+/);
            if (wordMatch && wordMatch[0].length > 1) {
              to = from + wordMatch[0].length;
            }
          }

          const severity =
            d.severity === Severity.Error ? Severity.Error : d.severity === Severity.Warning ? Severity.Warning : Severity.Info;
          return {
            from,
            to,
            severity,
            message: d.message,
            markClass: `cm-lint-has-${severity}`,
          };
        });
      },
      { delay: 300 },
    );
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;

    const extensions = [
      lineNumbers(),
      history(),
      indentOnInput(),
      bracketMatching(),
      foldGutter(),
      keymap.of([...defaultKeymap, ...historyKeymap, ...foldKeymap]),
      thoriumEditorTheme,
      thoriumHighlighting,
      getLanguageExtension(format),
      ...(format === FormatType.YAML ? [yamlValueHighlighter, yamlValueTheme] : []),
      createLinter(),
      EditorView.updateListener.of((update) => {
        if (update.docChanged) {
          const text = update.state.doc.toString();
          const parsed = parseText(text, format);
          onChangeRef.current(text, parsed);
        }
        if (update.selectionSet || update.docChanged) {
          const pos = update.state.selection.main.head;
          const line = update.state.doc.lineAt(pos).number;
          setCursorLine(line);
        }
      }),
      ...createPreviewExtensions(),
      EditorView.editable.of(!disabled),
      EditorState.readOnly.of(disabled),
    ];

    const state = EditorState.create({
      doc: value,
      extensions,
    });

    const view = new EditorView({
      state,
      parent: containerRef.current,
    });

    viewRef.current = view;
    (containerRef.current as HTMLElement & { _cmView?: EditorView })._cmView = view;

    return () => {
      view.destroy();
      viewRef.current = null;
      if (containerRef.current) {
        delete (containerRef.current as HTMLElement & { _cmView?: EditorView })._cmView;
      }
    };
  }, [format, disabled]);

  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    const currentText = view.state.doc.toString();
    if (currentText !== value) {
      view.dispatch({
        changes: { from: 0, to: currentText.length, insert: value },
      });
    }
  }, [value]);

  const handleValueClick = useCallback(
    (
      field: string,
      clickedValue: string,
      isList?: boolean,
      isMapEntry?: boolean,
      isRemoval?: boolean,
      schema?: FieldSchema,
      isReplace?: boolean,
    ) => {
      const view = viewRef.current;
      if (!view) return;
      view.dispatch({
        effects: addPreview.of({
          field,
          value: clickedValue,
          format,
          cursorLine: cursorLineRef.current,
          isList,
          isMapEntry,
          isRemoval,
          isReplace,
          schema,
        }),
      });
      view.focus();
    },
    [format],
  );

  const fill = height === '100%';
  return (
    <Root $fill={fill}>
      <EditorContainer ref={containerRef} $height={height} $fill={fill} />
      <SuggestionSlot>
        <SuggestionPanel suggestions={filteredSuggestions} onValueClick={handleValueClick} />
      </SuggestionSlot>
    </Root>
  );
};

export default CodeEditor;
