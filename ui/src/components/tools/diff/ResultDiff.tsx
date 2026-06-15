// spec: ../ToolResult.spec.md
import React, { useEffect, useMemo, useState } from 'react';
import { ErrorBoundary } from 'react-error-boundary';
import { DiffMethod } from 'react-diff-viewer-continued';
import styled from 'styled-components';

// project imports
import HexDiffViewer from './HexDiffViewer';
import TextDiff from './TextDiff';
import { diffForFile, diffForValue, resultToYaml } from './diffHelpers';
import { VersionSelect } from '../ToolResult.styled';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import RenderErrorAlert from '@components/shared/alerts/RenderErrorAlert';
import { decodeText, detectRenderKind, RenderKind } from '@components/shared/renderers';
import { OverlayWindow, Placement, PositionType } from '@components/shared/windows';
import { getResultsFile } from '@thorpi/results';
import { useAuth } from '@utilities/auth';
import { versionLabel } from '@utilities/version';
import { OutputDisplayType, type Output } from '@models/results';

// sentinel "compare" value for the structured result (vs. a named result file); the `__`-wrapped
// name is unlikely to collide with a real result-file name
const RESULT_TARGET = '__result__';

const Controls = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  padding: 8px 4px 12px;
  flex: 0 0 auto;
`;

const Control = styled.label`
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.85rem;
  color: var(--thorium-secondary-text);
`;

// scroll owner for the diff window: fills the remaining height below the controls and scrolls its
// content (OverlayBody no longer scrolls; see OverlayBody.tsx)
const DiffBody = styled.div`
  flex: 1 1 auto;
  min-height: 0;
  overflow: auto;
`;

// centered notice with symmetric vertical spacing (the empty/invalid-selection state)
const DiffNotice = styled.div`
  display: flex;
  justify-content: center;
  margin: 16px 0;
`;

export interface ResultDiffProps {
  results: Output[];
  sha256: string;
  tool: string;
  initialIndex: number;
  onClose: () => void;
}

/**
 * Floating diff window for comparing two versions of a tool result.
 *
 * A "what to compare" selector chooses the structured result or any result file; two version
 * selectors choose the base/compare versions. The result and text files diff via
 * `react-diff-viewer-continued`; binary files diff with the side-by-side {@link HexDiffViewer}.
 * Result-file bytes are fetched lazily on demand (directly via `getResultsFile`, not through the
 * shared result-file cache) and held only for the current base/compare selection.
 */
const ResultDiff: React.FC<ResultDiffProps> = ({ results, sha256, tool, initialIndex, onClose }) => {
  const { checkCookie } = useAuth();
  const [baseIndex, setBaseIndex] = useState(initialIndex);
  const [compareIndex, setCompareIndex] = useState(initialIndex === 0 ? Math.min(1, results.length - 1) : 0);
  const [target, setTarget] = useState<string>(RESULT_TARGET);

  const [baseBytes, setBaseBytes] = useState<ArrayBuffer | null>(null);
  const [compareBytes, setCompareBytes] = useState<ArrayBuffer | null>(null);
  const [loading, setLoading] = useState(false);
  // true when a file that IS listed in a version's files array failed to fetch, so an empty buffer
  // (a genuine empty/missing file) can be told apart from a transient network/auth failure
  const [fetchError, setFetchError] = useState(false);

  const baseResult = results[baseIndex];
  const compareResult = results[compareIndex];

  // union of result-file names across the two compared versions
  const fileOptions = useMemo(() => {
    const names = new Set<string>();
    (baseResult?.files ?? []).forEach((f) => names.add(f));
    (compareResult?.files ?? []).forEach((f) => names.add(f));
    return Array.from(names).sort();
  }, [baseResult, compareResult]);

  // fetch both sides' bytes when diffing a file target
  useEffect(() => {
    if (target === RESULT_TARGET) {
      setBaseBytes(null);
      setCompareBytes(null);
      setFetchError(false);
      return;
    }
    let active = true;
    const errorHandler = () => void checkCookie();
    const load = async () => {
      setLoading(true);
      // a file listed in a version's files array is expected to fetch; missing from the array is a
      // legitimate empty side (whole-file add/remove), so only the expected-but-failed case is an error
      const baseExpected = baseResult?.files?.includes(target) ?? false;
      const compareExpected = compareResult?.files?.includes(target) ?? false;
      const [b, c] = await Promise.all([
        baseExpected ? getResultsFile(sha256, tool, baseResult.id, target, errorHandler) : Promise.resolve(null),
        compareExpected ? getResultsFile(sha256, tool, compareResult.id, target, errorHandler) : Promise.resolve(null),
      ]);
      if (!active) return;
      setFetchError((baseExpected && b === null) || (compareExpected && c === null));
      setBaseBytes(b?.data ?? new ArrayBuffer(0));
      setCompareBytes(c?.data ?? new ArrayBuffer(0));
      setLoading(false);
    };
    void load();
    return () => {
      active = false;
    };
  }, [target, baseResult, compareResult, sha256, tool, checkCookie]);

  const oldTitle = baseResult ? versionLabel(baseResult.uploaded, baseResult.tool_version) : 'unknown';
  const newTitle = compareResult ? versionLabel(compareResult.uploaded, compareResult.tool_version) : 'unknown';

  const renderBody = () => {
    if (target === RESULT_TARGET) {
      // YAML-display results diff with the YAML structural differ; everything else by value type
      if (baseResult?.display_type === OutputDisplayType.Yaml || compareResult?.display_type === OutputDisplayType.Yaml) {
        return (
          <TextDiff
            oldValue={resultToYaml(baseResult?.result)}
            newValue={resultToYaml(compareResult?.result)}
            method={DiffMethod.YAML}
            oldTitle={oldTitle}
            newTitle={newTitle}
          />
        );
      }
      const { oldValue, newValue, method } = diffForValue(baseResult?.result, compareResult?.result);
      return <TextDiff oldValue={oldValue} newValue={newValue} method={method} oldTitle={oldTitle} newTitle={newTitle} />;
    }
    if (loading || baseBytes === null || compareBytes === null) {
      return <LoadingSpinner loading={true} />;
    }
    // an expected file that failed to fetch would otherwise render as a misleading whole-file diff
    if (fetchError) {
      return (
        <DiffNotice>
          <AlertBanner severity={Severity.Error}>Failed to load a result file for this comparison; try again.</AlertBanner>
        </DiffNotice>
      );
    }
    // detect on whichever side has content
    const kind = detectRenderKind(target, baseBytes.byteLength ? baseBytes : compareBytes);
    if (kind === RenderKind.Hex) {
      return <HexDiffViewer base={baseBytes} compare={compareBytes} baseTitle={oldTitle} compareTitle={newTitle} />;
    }
    const { oldValue, newValue, method } = diffForFile(target, decodeText(baseBytes), decodeText(compareBytes));
    return <TextDiff oldValue={oldValue} newValue={newValue} method={method} oldTitle={oldTitle} newTitle={newTitle} />;
  };

  return (
    <OverlayWindow
      id={`result-diff-${tool}`}
      show={true}
      title={`Diff: ${tool}`}
      width={1000}
      height={680}
      positioning={PositionType.Fixed}
      placement={Placement.Center}
      onHide={onClose}
    >
      <Controls>
        <Control>
          Compare
          <VersionSelect aria-label="What to compare" value={target} onChange={(e) => setTarget(e.target.value)}>
            <option value={RESULT_TARGET}>Result</option>
            {fileOptions.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </VersionSelect>
        </Control>
        <Control>
          Base
          <VersionSelect aria-label="Base version" value={baseIndex} onChange={(e) => setBaseIndex(Number(e.target.value))}>
            {results.map((r, idx) => (
              <option key={r.id} value={idx}>
                {versionLabel(r.uploaded, r.tool_version)}
              </option>
            ))}
          </VersionSelect>
        </Control>
        <Control>
          Against
          <VersionSelect aria-label="Compare version" value={compareIndex} onChange={(e) => setCompareIndex(Number(e.target.value))}>
            {results.map((r, idx) => (
              <option key={r.id} value={idx}>
                {versionLabel(r.uploaded, r.tool_version)}
              </option>
            ))}
          </VersionSelect>
        </Control>
      </Controls>
      <DiffBody>
        {baseIndex === compareIndex ? (
          <DiffNotice>
            <AlertBanner severity={Severity.Info}>Select two different versions to compare.</AlertBanner>
          </DiffNotice>
        ) : (
          // Contain a diff render failure to this window (with the version selectors still usable)
          // instead of letting it take down the file-details page. resetKeys re-attempts the render
          // when the user changes what/which versions are compared.
          <ErrorBoundary
            resetKeys={[target, baseIndex, compareIndex]}
            fallback={
              <RenderErrorAlert
                page={false}
                message={'Uh Oh! An error occurred while rendering this diff, please report it to your Thorium admins.'}
              />
            }
          >
            {renderBody()}
          </ErrorBoundary>
        )}
      </DiffBody>
    </OverlayWindow>
  );
};

export default ResultDiff;
