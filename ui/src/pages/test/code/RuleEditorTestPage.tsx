import React, { useState, useEffect } from 'react';
import CodeEditor from '@components/shared/inputs/code/CodeEditor/CodeEditor';
import Page from '@components/pages/Page';
import { addPreview, clearPreview, acceptPreview } from '@components/shared/inputs/code/CodeEditor/SuggestionPreview';
import type { RuleChecker, FormatType } from '@utilities/rules/types';

interface RuleEditorTestPageProps {
  title: string;
  sampleRule: string;
  checker: RuleChecker;
  format: FormatType;
  helpersKey: string;
  showParsed?: boolean;
}

const RuleEditorTestPage: React.FC<RuleEditorTestPageProps> = ({ title, sampleRule, checker, format, helpersKey, showParsed = false }) => {
  const [text, setText] = useState(sampleRule);
  const [parsed, setParsed] = useState<unknown>(null);

  useEffect(() => {
    (window as unknown as Record<string, unknown>)[helpersKey] = { addPreview, clearPreview, acceptPreview };
    return () => {
      delete (window as unknown as Record<string, unknown>)[helpersKey];
    };
  }, [helpersKey]);

  return (
    <Page title={title}>
      <h2>{title}</h2>
      <CodeEditor
        value={text}
        onChange={(t, p) => {
          setText(t);
          if (showParsed) setParsed(p);
        }}
        checker={checker}
        format={format}
        height="500px"
      />
      {showParsed && (
        <pre data-testid="parsed-output" style={{ marginTop: 16, fontSize: 11, opacity: 0.7 }}>
          {parsed ? JSON.stringify(parsed, null, 2) : '(no valid parse)'}
        </pre>
      )}
    </Page>
  );
};

export default RuleEditorTestPage;
