import { useRef, useState } from 'react';
import type { FC } from 'react';

// project imports
import PipelineInfo from './PipelineInfo';
import type { PipelineInfoHandle } from './PipelineInfo';
import { AccordionBody, AccordionHeader, AccordionItem } from '@components/shared/accordion';
import { HeaderActions, HeaderBtn, BanWarningIcon, DeleteConfirmModal } from '@components/shared/browsing';
import { ButtonVariant } from '@components/shared/buttons';
import { OverlayTipBottom, OverlayTipLeft, OverlayTipRight } from '@components/shared/overlay/tips';
import { useAuth } from '@utilities/auth';
import { canDeletePipeline, canModifyPipeline } from '@utilities/permissions';
import { deletePipeline } from '@thorpi/pipelines';
import type { Group } from '@models/groups';
import type { Pipeline } from '@models/pipelines';

interface PipelineAccordionItemProps {
  pipeline: Pipeline;
  groups: Record<string, Group>;
  canCreatePipeline: boolean;
  onUpdate: (editorObj: Record<string, unknown>, pipeline: Pipeline, setError: (e: string) => void) => Promise<boolean>;
  // Full list reload (with spinner) used after this pipeline is deleted.
  onRefresh: () => void;
  // Single-pipeline content refresh used after an in-place edit (e.g. order change).
  refreshPipeline: (group: string, name: string) => void;
  onExpand: (key: string) => void;
  onCopy: (pipeline: Pipeline) => void;
}

const PipelineAccordionItem: FC<PipelineAccordionItemProps> = ({
  pipeline,
  groups,
  canCreatePipeline,
  onUpdate,
  onRefresh,
  refreshPipeline,
  onExpand,
  onCopy,
}) => {
  const { userInfo } = useAuth();
  const [inEditMode, setEditMode] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deleteError, setDeleteError] = useState('');
  const pipelineInfoRef = useRef<PipelineInfoHandle>(null);

  // gate edit/delete against the backend pipeline authorization predicates
  const group = groups[pipeline.group];
  const userCanModify = !!userInfo && !!group && canModifyPipeline(pipeline, group, userInfo);
  const userCanDelete = !!userInfo && !!group && canDeletePipeline(pipeline, group, userInfo);

  const hasBans = pipeline.bans && Object.keys(pipeline.bans).length > 0;

  const handleCloseDeleteModal = () => {
    setShowDeleteModal(false);
    setDeleteError('');
  };

  const handleDelete = async () => {
    if (await deletePipeline(pipeline.group, pipeline.name, setDeleteError)) {
      onRefresh();
      handleCloseDeleteModal();
    }
  };

  return (
    <AccordionItem eventKey={`${pipeline.name}_${pipeline.group}`}>
      <AccordionHeader>
        <div className="accordion-item-name">
          <div className="text">{pipeline.name}</div>
        </div>
        <div className="accordion-item-relation" />
        <div className="accordion-item-ownership">
          <OverlayTipLeft tip={`This pipeline is owned by the ${pipeline.group} group.`}>
            <small>
              <i>{pipeline.group}</i>
            </small>
          </OverlayTipLeft>
        </div>
        <div className="accordion-item-status">
          {hasBans && (
            <OverlayTipRight tip="This pipeline has active bans and cannot run.">
              <BanWarningIcon />
            </OverlayTipRight>
          )}
        </div>
        <HeaderActions onClick={(e) => e.stopPropagation()}>
          {!inEditMode && canCreatePipeline && (
            <OverlayTipBottom tip={`Create a new pipeline using ${pipeline.name} as a template.`}>
              <HeaderBtn $variant={ButtonVariant.Ok} data-testid="header-btn-copy" onClick={() => onCopy(pipeline)}>
                Copy
              </HeaderBtn>
            </OverlayTipBottom>
          )}
          {!inEditMode && userCanModify && (
            <OverlayTipBottom tip="Edit this pipeline.">
              <HeaderBtn
                $variant={ButtonVariant.Secondary}
                data-testid="header-btn-edit"
                onClick={() => {
                  setEditMode(true);
                  onExpand(`${pipeline.name}_${pipeline.group}`);
                }}
              >
                Edit
              </HeaderBtn>
            </OverlayTipBottom>
          )}
          {inEditMode && userCanModify && (
            <OverlayTipBottom tip="Submit pending updates.">
              <HeaderBtn
                $variant={ButtonVariant.Ok}
                data-testid="header-btn-accept"
                onClick={() => pipelineInfoRef.current?.handleUpdate()}
              >
                Accept
              </HeaderBtn>
            </OverlayTipBottom>
          )}
          {inEditMode && userCanModify && (
            <OverlayTipBottom tip="Discard pending changes.">
              <HeaderBtn $variant={ButtonVariant.Secondary} data-testid="header-btn-discard" onClick={() => setEditMode(false)}>
                Discard
              </HeaderBtn>
            </OverlayTipBottom>
          )}
          {userCanDelete && (
            <OverlayTipBottom tip="Delete this pipeline.">
              <HeaderBtn $variant={ButtonVariant.Warning} data-testid="header-btn-delete" onClick={() => setShowDeleteModal(true)}>
                Delete
              </HeaderBtn>
            </OverlayTipBottom>
          )}
        </HeaderActions>
      </AccordionHeader>
      <AccordionBody>
        <PipelineInfo
          ref={pipelineInfoRef}
          pipeline={pipeline}
          groups={groups}
          inEditMode={inEditMode}
          onExitEditMode={() => setEditMode(false)}
          onUpdate={onUpdate}
          refreshPipeline={refreshPipeline}
        />
      </AccordionBody>
      <DeleteConfirmModal
        show={showDeleteModal}
        onHide={handleCloseDeleteModal}
        onConfirm={() => void handleDelete()}
        itemName={pipeline.name}
        itemType="pipeline"
        error={deleteError}
      />
    </AccordionItem>
  );
};

export default PipelineAccordionItem;
