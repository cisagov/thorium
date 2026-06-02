import { useEffect, useImperativeHandle, useMemo, useState } from 'react';
import type { FC, Ref } from 'react';
import { Badge, Form } from 'react-bootstrap';
import { FaQuestionCircle } from 'react-icons/fa';

// project imports
import { pipelineChecker } from './checker';
import { InfoRow, HeaderCol, DetailCol } from './PipelineInfo.styled';
import Markdown from '@components/shared/syntax/Markdown';
import TriggerDisplay from './TriggerDisplay';
import { BansContainer, BanItem } from '@components/shared/browsing';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import { BUTTON_BAR_MARGIN } from '@components/shared/buttons/tokens';
import ImagePipelineEditor from '@components/shared/inputs/code/CodeEditor/ImagePipelineEditor';
import FormatToggle from '@components/shared/inputs/code/CodeEditor/FormatToggle';
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
import type { Pipeline, PipelineBan, PipelineUpdate } from '@models/pipelines';

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
  const [editorObj, setEditorObj] = useState<Record<string, unknown> | null>(null);
  const [format, setFormat] = useState<FormatType>(FormatType.YAML);
  const [parseValid, setParseValid] = useState(false);
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
      setParseValid(true);
      setUpdateError('');
    } else {
      setEditorObj(null);
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

  const handleUpdate = async () => {
    if (editorObj && parseValid) {
      if (await onUpdate(editorObj, pipeline, setUpdateError)) {
        onExitEditMode();
      }
    }
  };

  useImperativeHandle(ref, () => ({ handleUpdate: () => void handleUpdate() }));

  return (
    <Form>
      {updateError != '' && (
        <div style={{ marginBottom: '0.25rem' }}>
          <AlertBanner>{updateError}</AlertBanner>
        </div>
      )}
      {inEditMode ? (
        <>
          <div style={{ marginBottom: BUTTON_BAR_MARGIN }}>
            <FormatToggle format={format} onFormatChange={setFormat} />
          </div>
          <ImagePipelineEditor
            value={editorObj || {}}
            onChange={handleEditorChange}
            checker={pipelineChecker}
            format={format}
            height="400px"
          />
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
              <Badge bg="" className="bg-blue">
                {pipeline.creator}
              </Badge>
            </DetailCol>
          </InfoRow>
          {pipeline.bans && Object.keys(pipeline.bans).length > 0 && (
            <InfoRow style={{ marginTop: '0.25rem' }}>
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
            </InfoRow>
          )}
          <InfoRow style={{ marginTop: '0.25rem' }}>
            <HeaderCol>
              <SimpleSubtitle>
                <b>Description</b>
              </SimpleSubtitle>
            </HeaderCol>
            <DetailCol>
              <Markdown>{pipeline.description ?? ''}</Markdown>
            </DetailCol>
          </InfoRow>
          <InfoRow style={{ marginTop: '0.25rem' }}>
            <HeaderCol>
              <OverlayTipRight
                tip={`The order of images to run. Sequential steps run one after another.
                Parallel steps (stacked vertically) run simultaneously.`}
              >
                <SimpleSubtitle>
                  <b>Order</b> <FaQuestionCircle />
                </SimpleSubtitle>
              </OverlayTipRight>
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
                <div style={{ marginTop: BUTTON_BAR_MARGIN }}>
                  <AlertBanner>{orderUpdateError}</AlertBanner>
                </div>
              )}
            </DetailCol>
          </InfoRow>
          <TriggerDisplay triggers={pipelineTriggers} />
          <InfoRow style={{ marginTop: '0.25rem' }}>
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
          </InfoRow>
        </>
      )}
    </Form>
  );
};

export default PipelineInfo;
