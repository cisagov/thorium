import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useParams, useLocation } from 'react-router-dom';
import styled from 'styled-components';

// project imports
import { Tabs, TabItem } from '@components/shared/tabs';
const AssociationGraph = React.lazy(() => import('@components/associations/graph/AssociationGraph'));
const Results = React.lazy(() => import('@components/pages/files/Results'));
const FileEntities = React.lazy(() => import('@components/pages/files/FileEntities'));
const RunPipelines = React.lazy(() => import('@components/pages/files/reactions/RunPipelines'));
import ReactionStatus from '@components/pages/files/reactions/ReactionStatus';
import Page from '@components/pages/Page';
import Download from '@components/pages/files/Download';
import Comments from '@components/pages/files/Comments';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import { GraphDataProvider } from '@components/associations/data/GraphDataContext';
import { fetchGroups } from '@utilities/fetch';
import { scrollToSection } from '@utilities/interactions';
import { updateURLSection } from '@utilities/url';
import { getFileDetails } from '@thorpi/files';
import type { Sample } from '@models/files';
import type { Group } from '@models/groups';
import type { Output } from '@models/results';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import FileInfo from './FileInfo';

// spec: ../EntityDetails.spec.md

// top-level file detail sections, in display order
const FILE_TABS: TabItem[] = [
  { key: 'results', label: 'Results' },
  { key: 'entities', label: 'Entities' },
  { key: 'associations', label: 'Associations' },
  { key: 'runpipelines', label: 'Create Reactions' },
  { key: 'comments', label: 'Comments' },
  { key: 'reactionstatus', label: 'Reaction Status' },
  { key: 'download', label: 'Download' },
];

// hash-deep-link keys are exactly the file tab keys, so derive them to avoid drift
const ValidTabs = FILE_TABS.map((t) => t.key);

const TabPanels = styled.div`
  width: 100%;
`;

// panels stay mounted (only toggled via display:none) so lazy children and their inView gating are preserved across tab switches
const TabPanel = styled.div<{ $active: boolean }>`
  display: ${({ $active }) => ($active ? 'block' : 'none')};
`;

// File details page: header info card + tabbed sections (results/entities/associations/…)
const FileDetails = () => {
  const { sha256 } = useParams<{ sha256: string }>();
  const [numResults, setNumResults] = useState(0);
  const [results, setResults] = useState<Record<string, Output[]>>({});
  const [details, setDetails] = useState<Partial<Sample>>({});
  const [groupDetails, setGroupDetails] = useState<Record<string, Group>>({});
  const [viewGraph, setViewGraph] = useState(false);
  const [entitiesTabSelected, setEntitiesTabSelected] = useState(false);
  const [reactionsTabSelected, setReactionsTabSelected] = useState(false);
  const [getFileError, setGetFileError] = useState('');
  const [loading, setLoading] = useState(true);
  const [deletionStatus, setDeletionStatus] = useState('');
  const location = useLocation();
  const section =
    location.hash && ValidTabs.includes(location.hash.replace('#', '').split('-')[0]) ? location.hash.replace('#', '').split('-') : [];
  const [allowResultsHashUpdate, setAllowResultsHashUpdate] = useState(false);
  const [activeTab, setActiveTab] = useState<string>(Array.isArray(section) && section.length ? section[0] : 'results');
  const associationInitial = useMemo(() => ({ samples: [sha256!] }), [sha256]);
  // the tab strip, scrolled into view when a toolbar action jumps to a tab
  const tabsRef = useRef<HTMLDivElement>(null);

  // jump to correct tab/subsection on page load
  useEffect(() => {
    const triggerPageScroll = () => {
      switch (section[0]) {
        case 'results':
          setAllowResultsHashUpdate(true);
          if (section.length >= 2) {
            const tool = section.slice(1).toString().replaceAll(',', '-');
            setTimeout(() => scrollToSection(`${section[0]}-tab-${tool}`), 1500);
          }
          break;
        case 'entities':
          setEntitiesTabSelected(true);
          break;
        case 'associations':
          setViewGraph(true);
          break;
        case 'reactionstatus':
          setReactionsTabSelected(true);
          break;
        default:
          setTimeout(() => scrollToSection(`${section[0]}-tab`), 1500);
          break;
      }
    };

    if (Array.isArray(section) && section.length) {
      triggerPageScroll();
    } else {
      setTimeout(() => window.scrollTo(0, 0), 10);
      setAllowResultsHashUpdate(true);
    }
  }, []);

  // fetch file details
  useEffect(() => {
    const fetchFileDetails = async () => {
      const reqDetails = await getFileDetails(sha256!, setGetFileError);
      setLoading(true);
      if (reqDetails) {
        setDetails(reqDetails);
      }
      setLoading(false);
    };
    void fetchFileDetails();
    void fetchGroups(setGroupDetails as (groups: Record<string, Group> | Group[] | string[]) => void, () => {}, true);
  }, [sha256, deletionStatus]);

  // handle tab switching with side effects
  const handleTabChange = (key: string | null) => {
    if (!key) return;
    setActiveTab(key);
    if (key.includes('results')) {
      setAllowResultsHashUpdate(true);
    } else {
      setAllowResultsHashUpdate(false);
    }
    setEntitiesTabSelected(key === 'entities');

    switch (key) {
      case 'reactionstatus':
        setReactionsTabSelected(true);
        updateURLSection(key, '');
        setViewGraph(false);
        break;
      case 'results':
        updateURLSection(key, '');
        setReactionsTabSelected(false);
        setViewGraph(false);
        break;
      case 'associations':
        updateURLSection(key, '');
        setReactionsTabSelected(false);
        setViewGraph(true);
        break;
      default:
        updateURLSection(key, '');
        setReactionsTabSelected(false);
        setViewGraph(false);
        break;
    }
  };

  // open a tab from the under-tags toolbar and scroll the tab strip into view so the jump is visible
  const jumpToTab = (key: string) => {
    handleTabChange(key);
    tabsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <Page id="file-info" className="full-min-width" title={`File · ${sha256}`}>
      {loading && <LoadingSpinner loading={true} />}
      {!loading &&
        deletionStatus &&
        (deletionStatus == 'Success' ? (
          <AlertBanner severity={Severity.Success}>Submission deleted successfully!</AlertBanner>
        ) : (
          <AlertBanner>{deletionStatus}</AlertBanner>
        ))}
      {!loading && getFileError && getFileError != '' && <AlertBanner>{getFileError}</AlertBanner>}
      <FileInfo
        details={details}
        setDetails={setDetails}
        groupDetails={groupDetails}
        setDeletionStatus={setDeletionStatus}
        onNavigateTab={jumpToTab}
      />
      <div ref={tabsRef}>
        <Tabs tabs={FILE_TABS} active={activeTab} onChange={handleTabChange} aria-label="File details sections" className="mt-4" />
      </div>
      <GraphDataProvider initial={associationInitial}>
        <TabPanels className="mt-4">
          <TabPanel $active={activeTab === 'results'}>
            <Results
              sha256={sha256!}
              results={results}
              setResults={setResults}
              numResults={numResults}
              allowHashUpdate={allowResultsHashUpdate}
              setNumResults={(num: number) => setNumResults(num)}
            />
          </TabPanel>
          <TabPanel $active={activeTab === 'entities'}>
            <FileEntities sha256={sha256!} inView={entitiesTabSelected} />
          </TabPanel>
          <TabPanel $active={activeTab === 'associations'}>
            <AssociationGraph inView={viewGraph} />
          </TabPanel>
          <TabPanel $active={activeTab === 'runpipelines'}>
            <RunPipelines sha256={sha256!} />
          </TabPanel>
          <TabPanel $active={activeTab === 'comments'}>
            <Comments sha256={sha256!} />
          </TabPanel>
          <TabPanel $active={activeTab === 'reactionstatus'}>
            <ReactionStatus sha256={sha256!} autoRefresh={reactionsTabSelected} />
          </TabPanel>
          <TabPanel $active={activeTab === 'download'}>
            <Download sha256={sha256!} />
          </TabPanel>
        </TabPanels>
      </GraphDataProvider>
    </Page>
  );
};

export default FileDetails;
