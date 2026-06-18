import { useEffect, useImperativeHandle, useMemo, useState } from 'react';
import type { FC, Ref } from 'react';
import { FaQuestionCircle } from 'react-icons/fa';

// project imports
import { pipelineChecker } from './checker';
import Fields from './Fields';
import type { PipelineFieldsValue } from './Fields';
import OrderField from './OrderField';
import {
  AlertSpacer,
  CreatorBadge,
  DetailCol,
  EditForm,
  FormatBar,
  HeaderCol,
  InfoRow,
  OrderErrorSpacer,
  SpacedInfoRow,
  SpacedSectionRow,
  ToggleBar,
} from './PipelineInfo.styled';
import { EditFieldCol, EditMiddle } from './shared.styled';
import TriggerDisplay from './TriggerDisplay';
import Triggers from './Triggers';
import { PipelineFormMode } from './types';
import Markdown from '@components/shared/syntax/Markdown';
import { BansContainer, BanItem } from '@components/shared/browsing';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import ImagePipelineEditor from '@components/shared/inputs/code/CodeEditor/ImagePipelineEditor';
import FormatToggle from '@components/shared/inputs/code/CodeEditor/FormatToggle';
import ViewModeToggle, { ViewMode } from '@components/shared/inputs/code/CodeEditor/ViewModeToggle';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import PipelineOrderFlow from '@components/shared/pipeline/PipelineOrderFlow';
import { ApplyBtn, DiscardBtn, OrderChangeBar } from '@components/shared/pipeline/PipelineOrderFlow.styled';
import SimpleSubtitle from '@components/shared/titles/SimpleSubtitle';
import { listImages } from '@thorpi/images';
import { updatePipeline } from '@thorpi/pipelines';
import { useAuth } from '@utilities/auth';
import { canModifyPipeline } from '@utilities/permissions';
import { FormatType } from '@utilities/rules/types';
import { pipelineToEditorObject } from '@utilities/transforms/pipeline';
import type { Group } from '@models/groups';
import type { EventTrigger, Pipeline, PipelineBan, PipelineUpdate } from '@models/pipelines';

/// Top-level editor keys managed by the Fields form
const FIELDS_KEYS = new Set(['name', 'group', 'description', 'sla']);

export interface PipelineInfoHandle {
  handleUpdate: () => void;
}

interface PipelineInfoProps {
  ref: Ref<PipelineInfoHandle>;
  pipeline: Pipeline;
  groups: Record<string, Group>;
  inEditMode: boolean;
  onExitEditMode: () => void;
  onUpdate: (editorObj: Record<string, unknown>, pipeline: Pipeline, setError: (e: string) => void) => Promise<boolean>;
  // Refresh only this pipeline's content in place (no full list reload / accordion collapse).
  refreshPipeline: (group: string, name: string) => void;
}

const PipelineInfo: FC<PipelineInfoProps> = ({ ref, pipeline, groups, inEditMode, onExitEditMode, onUpdate, refreshPipeline }) => {
  const { userInfo } = useAuth();
  const [updateError, setUpdateError] = useState('');
  const [editorObj, setEditorObj] = useState<Record<string, unknown>>(() => pipelineToEditorObject(pipeline));
  const [viewMode, setViewMode] = useState<ViewMode>(ViewMode.Form);
  const [format, setFormat] = useState<FormatType>(FormatType.YAML);
  const [parseValid, setParseValid] = useState(false);
  const [fieldErrors, setFieldErrors] = useState(false);
  const [triggerErrors, setTriggerErrors] = useState(false);
  // Per-field error state is only surfaced after a failed Accept attempt, so freshly added
  // (empty) fields/triggers don't immediately show errors while still being edited.
  const [displayErrors, setDisplayErrors] = useState(false);
  // Bumped whenever editorObj is re-seeded from the pipeline so the form sections
  // re-derive their internal state instead of keeping stale edits.
  const [formResetKey, setFormResetKey] = useState(0);
  const [pendingOrder, setPendingOrder] = useState<(string | string[])[] | null>(null);
  const [orderUpdateError, setOrderUpdateError] = useState('');
  const pipelineTriggers = pipeline.triggers ?? {};

  // reordering a pipeline runs the backend develop_many check (same gate as a full edit)
  const group = groups[pipeline.group];
  const userCanModifyOrder = !!userInfo && !!group && canModifyPipeline(pipeline, group, userInfo);

  const bannedImages = useMemo(() => {
    const names = new Set<string>();
    if (pipeline.bans) {
      for (const ban of Object.values(pipeline.bans)) {
        if (ban.ban_kind?.BannedImage) names.add(ban.ban_kind.BannedImage.image);
      }
    }
    return names;
  }, [pipeline.bans]);

  useEffect(() => {
    if (inEditMode) {
      setPendingOrder(null);
      setOrderUpdateError('');
      setEditorObj(pipelineToEditorObject(pipeline));
      setFormResetKey((k) => k + 1);
      setViewMode(ViewMode.Form);
      setParseValid(true);
      setUpdateError('');
      setDisplayErrors(false);
    } else {
      setEditorObj(pipelineToEditorObject(pipeline));
      setUpdateError('');
      pipelineChecker.clearValidImageNames();
    }
  }, [inEditMode]);

  const editorGroup = typeof editorObj?.group === 'string' ? editorObj.group : '';

  useEffect(() => {
    if (!editorGroup) return;
    let cancelled = false;
    void listImages(editorGroup, () => {}, false, null, 1000).then((result) => {
      if (cancelled) return;
      if (result && 'names' in result) pipelineChecker.setValidImageNames(editorGroup, result.names);
    });
    return () => {
      cancelled = true;
    };
  }, [editorGroup]);

  const handleOrderApply = async () => {
    if (!pendingOrder) return;
    if (await updatePipeline(pipeline.group, pipeline.name, { order: pendingOrder } as PipelineUpdate, setOrderUpdateError)) {
      setPendingOrder(null);
      setOrderUpdateError('');
      // Refresh only this pipeline so the reordered content rerenders without collapsing the accordion.
      refreshPipeline(pipeline.group, pipeline.name);
    }
  };

  const handleOrderDiscard = () => {
    setPendingOrder(null);
    setOrderUpdateError('');
  };

  const handleEditorChange = (obj: Record<string, unknown> | null) => {
    if (obj) {
      setEditorObj(obj);
      setParseValid(true);
    } else {
      setParseValid(false);
    }
  };

  // Switching views is lossless since both read/write the shared editorObj; the form
  // view always holds a parseable object, so re-mark it valid when returning to it.
  const handleViewModeChange = (mode: ViewMode) => {
    if (mode === viewMode) return;
    if (mode === ViewMode.Form) setParseValid(true);
    setViewMode(mode);
  };

  // Fields manages multiple top-level keys — remove old keys before merging new ones
  const handleFieldsChange = (fields: PipelineFieldsValue) => {
    setEditorObj((prev) => {
      const rest: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(prev)) {
        if (!FIELDS_KEYS.has(k)) rest[k] = v;
      }
      return { ...rest, ...(fields as unknown as Record<string, unknown>) };
    });
  };

  const handleUpdate = async () => {
    if (!parseValid) return;
    if (viewMode === ViewMode.Form && (fieldErrors || triggerErrors)) {
      setUpdateError('Please resolve missing fields or invalid entries');
      setDisplayErrors(true);
      return;
    }
    if (await onUpdate(editorObj, pipeline, setUpdateError)) {
      onExitEditMode();
    }
  };

  useImperativeHandle(ref, () => ({ handleUpdate: () => void handleUpdate() }));

  return (
    <EditForm>
      {updateError != '' && (
        <AlertSpacer>
          <AlertBanner>{updateError}</AlertBanner>
        </AlertSpacer>
      )}
      {inEditMode && (
        <ToggleBar>
          <ViewModeToggle viewMode={viewMode} onViewModeChange={handleViewModeChange} />
        </ToggleBar>
      )}
      {inEditMode && viewMode === ViewMode.Editor ? (
        <>
          <FormatBar>
            <FormatToggle format={format} onFormatChange={setFormat} />
          </FormatBar>
          <ImagePipelineEditor value={editorObj} onChange={handleEditorChange} checker={pipelineChecker} format={format} height="400px" />
        </>
      ) : inEditMode && viewMode === ViewMode.Form ? (
        <>
          <Fields
            value={editorObj as unknown as PipelineFieldsValue}
            groups={[]}
            onChange={handleFieldsChange}
            onValidate={setFieldErrors}
            showErrors={displayErrors}
            resetKey={formResetKey}
            mode={PipelineFormMode.Edit}
          />
          <SpacedSectionRow>
            <EditMiddle>
              <SimpleSubtitle>
                <b>Order</b>
              </SimpleSubtitle>
            </EditMiddle>
            <EditFieldCol>
              <OrderField
                order={Array.isArray(editorObj.order) ? (editorObj.order as (string | string[])[]) : []}
                onChange={(o) => setEditorObj((prev) => ({ ...prev, order: o }))}
                group={pipeline.group}
                bannedImages={bannedImages}
              />
            </EditFieldCol>
          </SpacedSectionRow>
          <SpacedSectionRow>
            <EditMiddle>
              <SimpleSubtitle>
                <b>Triggers</b>
              </SimpleSubtitle>
            </EditMiddle>
            <EditFieldCol>
              <Triggers
                value={(editorObj.triggers as Record<string, EventTrigger>) ?? {}}
                onChange={(t) => setEditorObj((prev) => ({ ...prev, triggers: t }))}
                onValidate={setTriggerErrors}
                resetKey={formResetKey}
                showErrors={displayErrors}
                mode={PipelineFormMode.Edit}
              />
            </EditFieldCol>
          </SpacedSectionRow>
        </>
      ) : (
        <>
          <InfoRow>
            <HeaderCol>
              <SimpleSubtitle>
                <b>Creator</b>
              </SimpleSubtitle>
            </HeaderCol>
            <DetailCol>
              <CreatorBadge>{pipeline.creator}</CreatorBadge>
            </DetailCol>
          </InfoRow>
          {pipeline.bans && Object.keys(pipeline.bans).length > 0 && (
            <SpacedInfoRow>
              <HeaderCol>
                <SimpleSubtitle>
                  <b>Bans</b>
                </SimpleSubtitle>
              </HeaderCol>
              <DetailCol>
                <BansContainer>
                  {Object.values(pipeline.bans).map((ban: PipelineBan) => (
                    <BanItem key={ban.id} severity={Severity.Warning}>
                      {ban.ban_kind?.Generic && <>Generic ban: {ban.ban_kind.Generic.msg}</>}
                      {ban.ban_kind?.BannedImage && (
                        <>
                          Banned image: <code>{ban.ban_kind.BannedImage.image}</code>
                        </>
                      )}
                      {'. Banned on '}
                      {new Date(ban.time_banned).toLocaleString()}
                    </BanItem>
                  ))}
                </BansContainer>
              </DetailCol>
            </SpacedInfoRow>
          )}
          <SpacedInfoRow>
            <HeaderCol>
              <SimpleSubtitle>
                <b>Description</b>
              </SimpleSubtitle>
            </HeaderCol>
            <DetailCol>
              <Markdown>{pipeline.description ?? ''}</Markdown>
            </DetailCol>
          </SpacedInfoRow>
          <SpacedInfoRow>
            <HeaderCol>
              <SimpleSubtitle>
                <b>Order</b>
              </SimpleSubtitle>
            </HeaderCol>
            <DetailCol>
              <PipelineOrderFlow
                order={pendingOrder ?? pipeline.order}
                onOrderChange={userCanModifyOrder ? setPendingOrder : undefined}
                bannedImages={bannedImages}
                group={pipeline.group}
              />
              {pendingOrder && (
                <OrderChangeBar>
                  <ApplyBtn type="button" onClick={() => void handleOrderApply()}>
                    Apply
                  </ApplyBtn>
                  <DiscardBtn type="button" onClick={handleOrderDiscard}>
                    Discard
                  </DiscardBtn>
                </OrderChangeBar>
              )}
              {orderUpdateError && (
                <OrderErrorSpacer>
                  <AlertBanner>{orderUpdateError}</AlertBanner>
                </OrderErrorSpacer>
              )}
            </DetailCol>
          </SpacedInfoRow>
          <TriggerDisplay triggers={pipelineTriggers} />
          <SpacedInfoRow>
            <HeaderCol>
              <OverlayTipRight tip={`The length of the SLA in seconds.`}>
                <SimpleSubtitle>
                  <b>SLA</b> <FaQuestionCircle />
                </SimpleSubtitle>
              </OverlayTipRight>
            </HeaderCol>
            <DetailCol>
              <p>{pipeline.sla}</p>
            </DetailCol>
          </SpacedInfoRow>
        </>
      )}
    </EditForm>
  );
};

export default PipelineInfo;
