import { useCallback, useEffect, useMemo, useState } from 'react';
import type { FC } from 'react';
import { Accordion, Badge, Button } from 'react-bootstrap';

// project imports
import PipelineAccordionItem from '@components/pages/pipelines/PipelineAccordionItem';
import CreatePipelineModal from '@components/pages/pipelines/CreatePipelineModal';
import { PIPELINE_CREATE_TEMPLATE } from '@components/pages/pipelines/CreatePipelineModal';
import { orderComparePipeline } from '@components/pages/files/reactions/pipelines';
import Page from '@components/pages/Page';
import { OmnibarPipelines } from '@components/pages/search/omnibar/Bars';
import type { Clause } from '@components/pages/search/omnibar/ClauseTypes';
import { getGroupsFromClauses, matchesStringClauses } from '@components/pages/search/omnibar/utils';
import NoResultsBanner from '@components/shared/alerts/NoResultsBanner';
import Title from '@components/shared/titles/Title';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import { OverlayTipRight, OverlayTipLeft } from '@components/shared/overlay/tips';
import { getPipeline, listPipelines, updatePipeline } from '@thorpi/pipelines';
import { useAuth } from '@utilities/auth';
import { fetchGroups } from '@utilities/fetch';
import { generateCopyName } from '@utilities/naming';
import { getThoriumRole } from '@utilities/role';
import { editorObjectToPipelineUpdate, pipelineToEditorObject } from '@utilities/transforms/pipeline';
import type { Group } from '@models/groups';
import type { Pipeline, PipelineUpdate } from '@models/pipelines';
import { RoleKey } from '@models/users';

/** Filter pipelines client-side by the omnibar clauses (name, creator, group). */
const filterPipelines = (pipelines: Pipeline[], clauses: Clause[]): Pipeline[] => {
  const groups = getGroupsFromClauses(clauses);

  return pipelines.filter((pipeline) => {
    const nameFilter = matchesStringClauses(clauses, 'name', pipeline.name);
    const creatorFilter = matchesStringClauses(clauses, 'creator', pipeline.creator);
    const groupFilter = groups.length > 0 ? groups.includes(pipeline.group) : true;
    return nameFilter && creatorFilter && groupFilter;
  });
};

const PipelineBrowsing: FC = () => {
  const [loading, setLoading] = useState(false);
  const [pipelines, setPipelines] = useState<Pipeline[]>([]);
  const [groups, setGroups] = useState<Record<string, Group>>({});
  const [activeKeys, setActiveKeys] = useState<string[]>([]);
  const [clauses, setClauses] = useState<Clause[]>([]);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [copySource, setCopySource] = useState<Pipeline | null>(null);
  const { userInfo, checkCookie } = useAuth();

  const filteredPipelines = filterPipelines(pipelines, clauses);

  const handleAccordionSelect = useCallback((eventKey: string | string[] | null | undefined) => {
    if (eventKey === null || eventKey === undefined) {
      setActiveKeys([]);
      return;
    }
    setActiveKeys(Array.isArray(eventKey) ? [...eventKey] : [eventKey]);
  }, []);

  const expandAccordionKey = useCallback((key: string) => {
    setActiveKeys((prev) => (prev.includes(key) ? prev : [...prev, key]));
  }, []);

  const fetchPipelinesData = async () => {
    setLoading(true);
    const results = await Promise.all(Object.keys(groups).map((group) => listPipelines(group, () => void checkCookie(), true, null, 1000)));
    setPipelines(results.filter(Boolean).flat() as Pipeline[]);
    setLoading(false);
  };

  // Refetch a single pipeline and replace just that entry in place, so an edit re-renders only
  // the changed pipeline without reloading the whole list or collapsing the open accordion.
  const refreshPipeline = useCallback(async (group: string, name: string) => {
    const fresh = await getPipeline(group, name, () => {});
    if (fresh) {
      setPipelines((prev) => prev.map((p) => (p.group === group && p.name === name ? fresh : p)));
    }
  }, []);

  useEffect(() => {
    void fetchGroups(setGroups as (groups: { [name: string]: Group } | Group[] | string[]) => void, () => {}, true);
  }, []);

  useEffect(() => {
    void fetchPipelinesData();
  }, [groups]);

  async function handlePipelineUpdate(
    editorObj: Record<string, unknown>,
    originalPipeline: Pipeline,
    setUpdateError: (e: string) => void,
  ): Promise<boolean> {
    const result = editorObjectToPipelineUpdate(editorObj, originalPipeline);
    if (!result) {
      setUpdateError('Invalid pipeline data');
      return false;
    }
    if (await updatePipeline(result.group, result.name, result.data as PipelineUpdate, setUpdateError)) {
      // Refresh only the edited pipeline so the open accordion stays put and only its content rerenders.
      void refreshPipeline(originalPipeline.group, originalPipeline.name);
      return true;
    }
    return false;
  }

  const canCreatePipeline = userInfo
    ? ([RoleKey.Developer, RoleKey.Analyst, RoleKey.Admin] as string[]).includes(getThoriumRole(userInfo.role))
    : false;

  const handleOpenCreate = useCallback(() => {
    setCopySource(null);
    setShowCreateModal(true);
  }, []);

  const handleCopyPipeline = useCallback((pipeline: Pipeline) => {
    setCopySource(pipeline);
    setShowCreateModal(true);
  }, []);

  const handleCloseCreateModal = useCallback(() => {
    setShowCreateModal(false);
    setCopySource(null);
  }, []);

  const initialModalData = useMemo(() => {
    if (!copySource) return PIPELINE_CREATE_TEMPLATE;
    const allNames = pipelines.map((p) => p.name);
    const copyName = generateCopyName(copySource.name, allNames);
    return { ...pipelineToEditorObject(copySource), name: copyName };
  }, [copySource, pipelines]);

  const pipelineCountTip =
    userInfo && getThoriumRole(userInfo.role) == RoleKey.Admin
      ? `There are a total of ${pipelines.length} Thorium pipelines.`
      : `There are a total of ${pipelines.length} Thorium pipelines owned by your groups.`;

  const createPipelineTip = canCreatePipeline
    ? `Create a new pipeline. You must be a
      Thorium developer, analyst, or admin to create a pipeline.`
    : `You must be a Thorium developer or
      admin to create a pipeline.`;

  return (
    <Page title="Pipelines · Thorium">
      <div className="d-flex justify-content-between ">
        <div>
          <h2>
            <OverlayTipRight tip={pipelineCountTip}>
              <Badge bg="" className="count-badge">
                {pipelines.length}
              </Badge>
            </OverlayTipRight>
          </h2>
        </div>
        <Title>Pipelines</Title>
        <div>
          <h2>
            <OverlayTipLeft tip={createPipelineTip}>
              <Button className="ok-btn" onClick={handleOpenCreate} disabled={!canCreatePipeline}>
                +
              </Button>
            </OverlayTipLeft>
          </h2>
        </div>
      </div>
      <div className="d-flex justify-content-center">
        <OmnibarPipelines clauses={clauses} setClauses={setClauses} pipelines={pipelines} />
      </div>
      <LoadingSpinner loading={loading}></LoadingSpinner>
      {!loading && filteredPipelines.length === 0 && <NoResultsBanner type="Pipelines" />}
      <Accordion alwaysOpen activeKey={activeKeys} onSelect={handleAccordionSelect}>
        {filteredPipelines
          .sort((a, b) => orderComparePipeline(a, b))
          .map((pipeline) => (
            <PipelineAccordionItem
              key={`${pipeline.name}_${pipeline.group}`}
              pipeline={pipeline}
              groups={groups}
              canCreatePipeline={canCreatePipeline}
              onUpdate={handlePipelineUpdate}
              onRefresh={() => void fetchPipelinesData()}
              refreshPipeline={(group, name) => void refreshPipeline(group, name)}
              onExpand={expandAccordionKey}
              onCopy={handleCopyPipeline}
            />
          ))}
      </Accordion>
      <CreatePipelineModal
        show={showCreateModal}
        onHide={handleCloseCreateModal}
        onCreated={() => void fetchPipelinesData()}
        initialData={initialModalData}
      />
    </Page>
  );
};

export default PipelineBrowsing;
