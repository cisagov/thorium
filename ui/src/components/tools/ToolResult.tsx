// spec: ./ToolResult.spec.md
import React, { Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { FaAngleDown, FaAngleUp, FaArrowsRotate, FaCodeCompare, FaDownload, FaLink } from 'react-icons/fa6';
import { InView } from 'react-intersection-observer';
import { toast } from 'react-toastify';

// project imports
import {
  CardBody,
  CardHeader,
  ClipViewport,
  ExpandToggle,
  FadeOverlay,
  HeaderControls,
  HeaderTabs,
  ScrollArea,
  TitleGroup,
  TitleLink,
  TitleRow,
  ToggleRow,
  ToolName,
  ToolResultCard,
  ToolVersion,
  VersionSelect,
} from './ToolResult.styled';
import ChildrenTab from './tabs/ChildrenTab';
import EntitiesTab from './tabs/EntitiesTab';
import FilesTab from './tabs/FilesTab';
import ResultTab from './tabs/ResultTab';
import { ToolResultTabKey } from './tabs/types';
import { useChildrenMetadata } from './tabs/useChildrenMetadata';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import { IconButton } from '@components/shared/buttons';
import { ButtonSize } from '@components/shared/buttons/types';
import { OverlayTipTop } from '@components/shared/overlay/tips';
import { Tabs, TabItem } from '@components/shared/tabs';
import { downloadResultsAsZip } from '@utilities/zip';
import { useAuth } from '@utilities/auth';
import { formatImageVersion, versionLabel } from '@utilities/version';
import { OutputDisplayType, type Output } from '@models/results';

// the diff view (and its heavy emotion/diff deps) only loads when the user opens it
const ResultDiff = React.lazy(() => import('./diff/ResultDiff'));

// default (collapsed) height of the result body before the user expands it
const DEFAULT_BODY_HEIGHT = 420;

interface ToolResultProps {
  /** All result versions for this tool, most-recent-first (as returned by the API). */
  results: Output[];
  header: string;
  sha256: string;
  tool: string;
  updateInView: (inView: boolean, tool: string) => void;
  updateURLSection: (section: string, value: string) => void;
}

const ToolResult = ({ results, header, sha256, tool, updateInView, updateURLSection }: ToolResultProps) => {
  const { checkCookie } = useAuth();
  const [activeIndex, setActiveIndex] = useState(0);
  const [activeTab, setActiveTab] = useState<ToolResultTabKey>(ToolResultTabKey.Result);
  const [showDiff, setShowDiff] = useState(false);
  // collapsible body state — capped to a default height until the user expands it
  const [collapsed, setCollapsed] = useState(true);
  const [overflowing, setOverflowing] = useState(false);
  // whether the collapsed body is scrolled to the bottom — hides the fade so the last line is readable
  const [atBottom, setAtBottom] = useState(false);
  const bodyContentRef = useRef<HTMLDivElement>(null);

  const result = results[activeIndex] ?? results[0];
  const type = result?.display_type ?? OutputDisplayType.Json;

  // shared child-metadata state for the Children tab body + the header "fetch" button; auto-fetches
  // small child sets lazily once the tab is opened
  const childrenMeta = useChildrenMetadata(result ?? results[0], activeTab === ToolResultTabKey.Children);

  // keep the selected version in range if a poll refresh returns fewer versions, so the version
  // selector and diff button don't point past the end of the array
  useEffect(() => {
    setActiveIndex((i) => Math.min(i, Math.max(0, results.length - 1)));
  }, [results]);

  // measure the body content and track whether it exceeds the default height (so the
  // show-more/less toggle only appears when there's something to expand)
  useEffect(() => {
    const el = bodyContentRef.current;
    if (!el) return;
    const check = () => setOverflowing(el.scrollHeight > DEFAULT_BODY_HEIGHT + 8);
    check();
    const observer = new ResizeObserver(check);
    observer.observe(el);
    return () => observer.disconnect();
  }, [activeTab, activeIndex, result]);

  // start each tab/version at the default (collapsed) size, scrolled to the top
  useEffect(() => {
    setCollapsed(true);
    setAtBottom(false);
  }, [activeTab, activeIndex]);

  // re-collapsing resets the scroll position, so the fade should reappear
  useEffect(() => {
    if (collapsed) setAtBottom(false);
  }, [collapsed]);

  // hide the fade once the collapsed body is scrolled to the bottom so the last line is fully readable
  const handleBodyScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    setAtBottom(el.scrollTop + el.clientHeight >= el.scrollHeight - 1);
  };

  // build the visible tab set from what the active result actually contains
  const tabs = useMemo<TabItem<ToolResultTabKey>[]>(() => {
    const items: TabItem<ToolResultTabKey>[] = [{ key: ToolResultTabKey.Result, label: 'Result' }];
    const fileCount = result?.files?.length ?? 0;
    if (fileCount > 0) items.push({ key: ToolResultTabKey.Files, label: 'Files', count: fileCount, tip: 'Tool Result Files' });
    const childCount = result?.children ? Object.keys(result.children).length : 0;
    if (childCount > 0) {
      // large child sets fetch on demand via an action button on the tab itself; its tooltip replaces
      // the tab's own tip so the two don't both appear
      const showFetch = childrenMeta.isManual && childrenMeta.status !== 'done';
      items.push({
        key: ToolResultTabKey.Children,
        label: 'Children',
        count: childCount,
        ...(showFetch
          ? {
              action: {
                icon: <FaArrowsRotate />,
                tip: 'Fetch children metadata',
                ariaLabel: 'Fetch children metadata',
                onClick: childrenMeta.fetch,
                disabled: childrenMeta.status === 'loading',
              },
            }
          : { tip: 'Files extracted from the sample' }),
      });
    }
    // count the total number of entities (sum of per-type counts), not the number of types
    const entityCount = result?.entities ? Object.values(result.entities).reduce((sum, n) => sum + n, 0) : 0;
    if (entityCount > 0)
      items.push({ key: ToolResultTabKey.Entities, label: 'Entities', count: entityCount, tip: 'Potential identified entities' });
    return items;
  }, [result, childrenMeta.isManual, childrenMeta.status, childrenMeta.fetch]);

  // if the active tab disappears after switching versions, fall back to Result
  useEffect(() => {
    if (!tabs.some((t) => t.key === activeTab)) {
      setActiveTab(ToolResultTabKey.Result);
    }
  }, [tabs, activeTab]);

  // copy a deep link to this result section to the clipboard (preserves the jump-to behavior)
  const copySectionLink = () => {
    updateURLSection('results', tool);
    void navigator.clipboard.writeText(window.location.href);
    toast(`Copied "${window.location.href}" to clipboard!`);
  };

  const handleDownloadAll = () => {
    if (result) void downloadResultsAsZip(sha256, tool, result, () => void checkCookie());
  };

  if (!result) return null;

  // collapse only applies to content that actually overflows the default height
  const isClipped = collapsed && overflowing;

  return (
    <InView
      as="div"
      id={`results-tab-${tool}`}
      className="navbar-scroll-offset results-content"
      // trip-line band near the top of the viewport (below the 60px navbar) with threshold 0 so
      // a tool is "in view" whenever it crosses the band — works even for tiles taller than the
      // viewport, which a fractional threshold could never satisfy
      rootMargin="-60px 0px -55% 0px"
      threshold={0}
      onChange={(inView) => updateInView(inView, tool)}
    >
      <ToolResultCard>
        <CardHeader>
          <TitleRow>
            <TitleGroup>
              <OverlayTipTop tip="Copy a link to this result">
                <TitleLink onClick={copySectionLink}>
                  <FaLink size={12} />
                  <ToolName>{header}</ToolName>
                </TitleLink>
              </OverlayTipTop>
              {formatImageVersion(result.tool_version) && <ToolVersion>{formatImageVersion(result.tool_version)}</ToolVersion>}
            </TitleGroup>
            {tabs.length > 1 && (
              <HeaderTabs>
                <Tabs tabs={tabs} active={activeTab} onChange={setActiveTab} aria-label={`${header} result sections`} flush />
              </HeaderTabs>
            )}
            <HeaderControls>
              {results.length > 1 && (
                <>
                  <VersionSelect
                    aria-label="Select result version"
                    value={activeIndex}
                    onChange={(e) => setActiveIndex(Number(e.target.value))}
                  >
                    {results.map((r, idx) => (
                      <option key={r.id} value={idx}>
                        {versionLabel(r.uploaded, r.tool_version)}
                      </option>
                    ))}
                  </VersionSelect>
                  <OverlayTipTop tip="Diff results">
                    <IconButton size={ButtonSize.Small} aria-label="Diff results" onClick={() => setShowDiff(true)}>
                      <FaCodeCompare />
                    </IconButton>
                  </OverlayTipTop>
                </>
              )}
              <OverlayTipTop tip="Download all results">
                <IconButton size={ButtonSize.Small} aria-label="Download all results" onClick={handleDownloadAll}>
                  <FaDownload />
                </IconButton>
              </OverlayTipTop>
              {overflowing && (
                <OverlayTipTop tip={collapsed ? 'Expand' : 'Collapse'}>
                  <IconButton
                    size={ButtonSize.Small}
                    aria-label={collapsed ? 'Expand result' : 'Collapse result'}
                    onClick={() => setCollapsed((c) => !c)}
                  >
                    {collapsed ? <FaAngleDown /> : <FaAngleUp />}
                  </IconButton>
                </OverlayTipTop>
              )}
            </HeaderControls>
          </TitleRow>
        </CardHeader>
        <CardBody role="tabpanel" id={`tabpanel-${activeTab}`} aria-labelledby={`tab-${activeTab}`}>
          <ClipViewport>
            <ScrollArea
              data-testid="result-body-scroll"
              $collapsed={isClipped}
              $maxHeight={DEFAULT_BODY_HEIGHT}
              onScroll={handleBodyScroll}
            >
              <div ref={bodyContentRef}>
                {activeTab === ToolResultTabKey.Result && <ResultTab result={result} sha256={sha256} tool={tool} type={type} />}
                {/* key on result.id so per-version state (opened file bytes) resets on a version switch */}
                {activeTab === ToolResultTabKey.Files && (
                  <FilesTab key={result.id} result={result} sha256={sha256} tool={tool} type={type} />
                )}
                {activeTab === ToolResultTabKey.Children && (
                  <ChildrenTab
                    result={result}
                    sha256={sha256}
                    tool={tool}
                    type={type}
                    samples={childrenMeta.samples}
                    status={childrenMeta.status}
                    loaded={childrenMeta.loaded}
                    total={childrenMeta.total}
                  />
                )}
                {/* key on result.id so per-version state (entities by kind, created flags) resets on a version switch */}
                {activeTab === ToolResultTabKey.Entities && (
                  <EntitiesTab key={result.id} result={result} sha256={sha256} tool={tool} type={type} />
                )}
              </div>
            </ScrollArea>
            {isClipped && !atBottom && <FadeOverlay />}
          </ClipViewport>
          {overflowing && (
            <ToggleRow>
              <ExpandToggle onClick={() => setCollapsed((c) => !c)} aria-expanded={!collapsed}>
                {collapsed ? (
                  <>
                    <FaAngleDown /> Show more
                  </>
                ) : (
                  <>
                    <FaAngleUp /> Show less
                  </>
                )}
              </ExpandToggle>
            </ToggleRow>
          )}
        </CardBody>
      </ToolResultCard>
      {showDiff && (
        <Suspense fallback={<LoadingSpinner loading={true} />}>
          <ResultDiff results={results} sha256={sha256} tool={tool} initialIndex={activeIndex} onClose={() => setShowDiff(false)} />
        </Suspense>
      )}
    </InView>
  );
};

export default ToolResult;
