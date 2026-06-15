// spec: ../ToolResult.spec.md
import React from 'react';
import { ErrorBoundary } from 'react-error-boundary';
import styled from 'styled-components';

// project imports
import Decomposition from '../displays/Decomposition';
import Disassembly from '../displays/Disassembly';
import Image from '../displays/Image';
import JSON from '../displays/JSON';
import Markdown from '../displays/Markdown';
import String from '../displays/String';
import Tables from '../displays/Tables';
import XML from '../displays/XML';
import Yaml from '../displays/Yaml';
import AvMulti from '../displays/custom/AvMulti';
import TC2 from '../displays/custom/TC2';
import VBA from '../displays/custom/VBA';
import SafeHtml from '../SafeHtml';
import { ToolResultTabProps } from './types';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import RenderErrorAlert from '@components/shared/alerts/RenderErrorAlert';
import { OutputDisplayType, type Value } from '@models/results';

// plain flex container (not a bootstrap Row, whose negative side margins overflowed the clipped
// result body and clipped the left edge of the content)
const ResultContainer = styled.div`
  display: flex;
  justify-content: center;
  width: 100%;
  min-width: 0;
`;

/**
 * True when a result value has no displayable content: null/undefined, an empty/whitespace string,
 * the literal `"{}"`/`"[]"` strings, an empty array, or an object with no keys (e.g. JSON output
 * with no keys).
 */
export function isEmptyResult(value: Value): boolean {
  if (value === null || value === undefined) return true;
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed === '' || trimmed === '{}' || trimmed === '[]';
  }
  if (Array.isArray(value)) return value.length === 0;
  if (typeof value === 'object') return Object.keys(value).length === 0;
  return false;
}

/**
 * The "Result" tab body: dispatches to the appropriate display component based on the result's
 * display type (and, for custom display types, the tool name). Extracted from the old monolithic
 * ToolResult so the result rendering lives alongside the other tabs.
 */
const ResultTab: React.FC<ToolResultTabProps> = ({ result, sha256, tool, type }) => {
  // Every display type renders nothing for empty content, so show a consistent info alert instead.
  // Image is the exception — it can render images from result files even with an empty `result`.
  const hasImageFiles = type === OutputDisplayType.Image && (result.files?.length ?? 0) > 0;
  if (isEmptyResult(result.result) && !hasImageFiles) {
    return <AlertBanner severity={Severity.Info}>This tool produced an empty result.</AlertBanner>;
  }
  return (
    <ErrorBoundary
      onError={(error, info) => {
        console.error('ErrorBoundary caught:', error);
        console.error('React component stack:', info.componentStack);
      }}
      fallback={
        <RenderErrorAlert
          page={false}
          message={
            'Uh Oh! An error occurred while rendering this result, please report it to your Thorium admins.\nNote: This may be caused by an image with a misconfigured display_type. '
          }
        />
      }
    >
      <ResultContainer>
        {type == OutputDisplayType.Custom && (tool == 'symantec' || tool == 'clamav') && (
          <AvMulti result={result} sha256={sha256} tool={tool} />
        )}
        {type == OutputDisplayType.Custom && tool == 'vbaextraction' && <VBA result={result} />}
        {type == OutputDisplayType.Custom && (tool == 'titanium-core2' || tool == 'tc2') && (
          <TC2 result={result} sha256={sha256} tool={tool} />
        )}
        {type == OutputDisplayType.Decomposition && <Decomposition result={result} sha256={sha256} tool={tool} />}
        {type == OutputDisplayType.Disassembly && <Disassembly result={result} sha256={sha256} tool={tool} />}
        {type == OutputDisplayType.Html && <SafeHtml result={result} sha256={sha256} tool={tool} />}
        {type == OutputDisplayType.Image && <Image result={result} sha256={sha256} tool={tool} />}
        {type == OutputDisplayType.Json && <JSON result={result} sha256={sha256} tool={tool} />}
        {type == OutputDisplayType.Markdown && <Markdown result={result} sha256={sha256} tool={tool} />}
        {type == OutputDisplayType.String && <String result={result} sha256={sha256} tool={tool} errors={[]} warnings={[]} />}
        {type == OutputDisplayType.Table && <Tables result={result} sha256={sha256} tool={tool} />}
        {type == OutputDisplayType.Xml && <XML result={result} sha256={sha256} tool={tool} />}
        {type == OutputDisplayType.Yaml && <Yaml result={result} sha256={sha256} tool={tool} />}
      </ResultContainer>
    </ErrorBoundary>
  );
};

export default ResultTab;
