import { JSX } from 'react';
import { Row } from 'react-bootstrap';
import { MdRule } from 'react-icons/md';
import styled from 'styled-components';

// project imports
import { EntityDetailsConfig } from './configs';
import { DetailsMetadataProps } from '../EntityDetails';
import InfoValue from '@entities/shared/InfoValue';
import FieldBadge from '@components/shared/badges/FieldBadge';
import { getEntity } from '@thorpi/entities';
import { Entities } from '@models/entities';
import { BlankSigmaRule, SigmaActionToTake, SigmaRule, SigmaRuleAppliesTo, SigmaRuleMetaFields } from '@models/entities/rules/sigma';
import InfoHeader from '@components/entities/shared/InfoHeader';
import CodeEditor from '@components/shared/inputs/code/CodeEditor/CodeEditor';
import { SigmaRuleChecker } from '@utilities/rules/sigma';
import SelectInputArray from '@components/shared/inputs/selectable/SelectInputArray';
import NumberInput from '@components/shared/inputs/NumberInput';
import { ActionsEditor } from '@components/entities/create/configs/SigmaRuleCreateConfig';
import SigmaSVG from '@assets/icons/sigma.svg?raw';
import SigmaIcon from '@components/shared/icons/SigmaIcon';
import { FormatType } from '@utilities/rules/types';

const RulePreview = styled.pre`
  background-color: var(--thorium-secondary-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 0.375rem;
  padding: 0.75rem;
  font-size: 0.8rem;
  color: var(--thorium-text);
  //max-height: 400px;
  overflow: auto;
  white-space: pre-wrap;
  margin: 0;
`;

const ActionViewCard = styled.div`
  background-color: var(--thorium-secondary-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 0.375rem;
  padding: 0.5rem 0.75rem;
  margin-bottom: 0.5rem;
  font-size: 0.875rem;
`;

const ActionViewLabel = styled.span`
  font-weight: 600;
  font-size: 0.75rem;
  color: var(--thorium-secondary-text);
  text-transform: uppercase;
  letter-spacing: 0.03em;
`;

const ActionsView: React.FC<{ actions: SigmaActionToTake[] }> = ({ actions }) => {
  if (actions.length === 0) return <span style={{ color: 'var(--thorium-secondary-text)' }}>None</span>;
  return (
    <>
      {actions.map((action, i) => (
        <ActionViewCard key={i}>
          <ActionViewLabel>Flag #{i + 1}</ActionViewLabel>
          {action.Flag.content && (
            <div>
              <strong>Content:</strong> {action.Flag.content}
            </div>
          )}
          <div>
            <strong>Reasoning:</strong> {action.Flag.reasoning}
          </div>
        </ActionViewCard>
      ))}
    </>
  );
};

const SigmaRuleMetaInfo = ({ entity, pendingEntity, handleUpdate, editing }: DetailsMetadataProps<Entities.SigmaRule>): JSX.Element => {
  // handle any updates to SigmaRule entity metadata
  function updatePendingMeta<T extends keyof SigmaRuleMetaFields>(field: T, value: SigmaRuleMetaFields[T]) {
    const updates: SigmaRuleMetaFields = structuredClone(pendingEntity.metadata.SigmaRule);
    updates[field] = value;
    handleUpdate('metadata', { SigmaRule: updates });
  }
  const sigmaChecker = new SigmaRuleChecker();
  return (
    <>
      <Row className="mt-3">
        <InfoHeader>Rule</InfoHeader>
        <InfoValue>
          {editing ? (
            <CodeEditor
              value={pendingEntity.metadata.SigmaRule.rule}
              onChange={(text) => updatePendingMeta('rule', text)}
              checker={sigmaChecker}
              format={FormatType.YAML}
              height="400px"
            />
          ) : (
            <RulePreview>{entity.metadata.SigmaRule.rule}</RulePreview>
          )}
        </InfoValue>
      </Row>
      <hr className="my-3" />
      <Row>
        <InfoHeader>Applies To</InfoHeader>
        <InfoValue>
          {editing ? (
            <SelectInputArray
              isCreatable={false}
              options={Object.values(SigmaRuleAppliesTo)}
              values={pendingEntity.metadata.SigmaRule.applies_to}
              onChange={(selected) => updatePendingMeta('applies_to', selected as SigmaRuleAppliesTo[])}
            />
          ) : (
            <FieldBadge color="Gray" noNull field={entity.metadata.SigmaRule.applies_to} />
          )}
        </InfoValue>
      </Row>
      <hr className="my-3" />
      <Row>
        <InfoHeader>Score</InfoHeader>
        <InfoValue>
          {editing ? (
            <NumberInput
              value={pendingEntity.metadata.SigmaRule.score}
              onChange={(v) => updatePendingMeta('score', v ?? 0)}
              min={0}
              step={10}
            />
          ) : (
            <>{entity.metadata.SigmaRule.score}</>
          )}
        </InfoValue>
      </Row>
      <hr className="my-3" />
      <Row>
        <InfoHeader>Actions</InfoHeader>
        <InfoValue>
          {editing ? (
            <ActionsEditor
              actions={pendingEntity.metadata.SigmaRule.actions}
              onChange={(actions) => updatePendingMeta('actions', actions)}
            />
          ) : (
            <ActionsView actions={entity.metadata.SigmaRule.actions} />
          )}
        </InfoValue>
      </Row>
    </>
  );
};

const getSigmaRuleDetails = async (collectionId: string, setError: (e: string) => void, updateEntity: (c: SigmaRule) => void) => {
  getEntity(collectionId, setError).then((data) => {
    if (data && data.kind === Entities.SigmaRule) {
      updateEntity(data as SigmaRule);
    }
  });
};

const SigmaRuleDetailsConfig: EntityDetailsConfig<Entities.SigmaRule> = {
  getEntityDetails: getSigmaRuleDetails,
  EntityMetaInfo: SigmaRuleMetaInfo,
  BlankEntity: BlankSigmaRule,
  icon: (size: number) => <SigmaIcon size={size} />,
};

export default SigmaRuleDetailsConfig;
