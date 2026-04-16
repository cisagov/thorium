import { JSX } from 'react';
import { Row, Form, Card } from 'react-bootstrap';

// project imports
import { EntityCreateConfig } from './config';
import { CreateMetadataProps } from '../EntityCreate';
import InfoHeader from '@entities/shared/InfoHeader';
import InfoValue from '@entities/shared/InfoValue';
import { Entities } from '@models/entities/entities';
import {
  BlankCreateSigmaRule,
  SigmaActionToTake,
  SigmaAutoFlag,
  SigmaRuleAppliesTo,
  SigmaRuleMetaFields,
} from '@models/entities/rules/sigma';
import CodeEditor from '@components/shared/inputs/code/CodeEditor/CodeEditor';
import { SigmaRuleChecker } from '@utilities/rules/sigma';
import SelectInputArray from '@components/shared/inputs/selectable/SelectInputArray';
import styled from 'styled-components';
import NumberInput from '@components/shared/inputs/NumberInput';
import { FormatType } from '@utilities/rules/types';

const ActionCard = styled(Card)`
  margin-bottom: 0.75rem;
  background-color: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
`;

const ActionCardBody = styled(Card.Body)`
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
`;

const ActionHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const ActionLabel = styled.span`
  font-weight: 600;
  font-size: 0.85rem;
  color: var(--thorium-secondary-text);
  text-transform: uppercase;
  letter-spacing: 0.03em;
`;

const RemoveButton = styled.button`
  background: none;
  border: none;
  color: var(--thorium-secondary-text);
  font-size: 1.25rem;
  cursor: pointer;
  padding: 0 0.25rem;
  line-height: 1;

  &:hover {
    color: var(--bs-danger);
  }
`;

const AddActionButton = styled.button`
  color: var(--thorium-button-text);
  background-color: var(--thorium-empty-bg);
  border: 1px solid var(--thorium-empty-bg);
  border-radius: 4px;
  padding: 0.25rem 0.75rem;
  font-size: 0.875rem;
  cursor: pointer;

  &:hover {
    filter: brightness(80%);
  }

  &:active {
    filter: brightness(70%);
    box-shadow: 0 0 0 0.1rem var(--thorium-highlight-panel-border);
  }
`;

interface ActionsEditorProps {
  actions: SigmaActionToTake[];
  onChange: (actions: SigmaActionToTake[]) => void;
}

export const ActionsEditor: React.FC<ActionsEditorProps> = ({ actions, onChange }) => {
  const addAction = () => {
    onChange([...actions, { Flag: { content: '', reasoning: '' } }]);
  };

  const updateAction = (index: number, flag: SigmaAutoFlag) => {
    const next = structuredClone(actions);
    next[index] = { Flag: flag };
    onChange(next);
  };

  const removeAction = (index: number) => {
    onChange(actions.filter((_, i) => i !== index));
  };

  return (
    <>
      {actions.map((action, i) => (
        <ActionCard key={i}>
          <ActionCardBody>
            <ActionHeader>
              <ActionLabel>Flag Action #{i + 1}</ActionLabel>
              <RemoveButton onClick={() => removeAction(i)} title="Remove action">
                &times;
              </RemoveButton>
            </ActionHeader>
            <Form.Group>
              <Form.Label>Content</Form.Label>
              <Form.Control
                type="text"
                placeholder="Interesting characteristic (optional)"
                value={action.Flag.content ?? ''}
                onChange={(e) => updateAction(i, { ...action.Flag, content: e.target.value || undefined })}
              />
            </Form.Group>
            <Form.Group>
              <Form.Label>
                Reasoning<sub>*</sub>
              </Form.Label>
              <Form.Control
                as="textarea"
                rows={2}
                placeholder="Reason for this flag (required)"
                value={action.Flag.reasoning}
                onChange={(e) => updateAction(i, { ...action.Flag, reasoning: e.target.value })}
              />
            </Form.Group>
          </ActionCardBody>
        </ActionCard>
      ))}
      <AddActionButton onClick={addAction}>+ Add Flag Action</AddActionButton>
    </>
  );
};

const SigmaRuleMetaInfo = ({ entity, onChange }: CreateMetadataProps<Entities.SigmaRule>): JSX.Element => {
  // helper to update nested metadata
  const updatePendingMeta = <T extends keyof SigmaRuleMetaFields>(field: T, value: SigmaRuleMetaFields[T]) => {
    const updates: SigmaRuleMetaFields = structuredClone(entity.metadata.SigmaRule);
    updates[field] = value;
    onChange('metadata', { SigmaRule: updates });
  };
  const sigmaChecker = new SigmaRuleChecker();
  return (
    <>
      <Row>
        <InfoHeader>Rule</InfoHeader>
        <InfoValue>
          <CodeEditor
            value={entity.metadata.SigmaRule.rule}
            onChange={(text) => updatePendingMeta('rule', text)}
            checker={sigmaChecker}
            format={FormatType.YAML}
            height="400px"
          />
        </InfoValue>
      </Row>
      <Row>
        <InfoHeader>Applies To</InfoHeader>
        <InfoValue className="mt-2">
          <SelectInputArray
            isCreatable={false}
            options={Object.values(SigmaRuleAppliesTo)}
            values={entity.metadata.SigmaRule.applies_to}
            onChange={(selected) => updatePendingMeta('applies_to', selected as SigmaRuleAppliesTo[])}
          />
        </InfoValue>
      </Row>
      <Row>
        <InfoHeader>Score</InfoHeader>
        <InfoValue className="mt-2">
          <NumberInput
            step={10}
            value={entity.metadata.SigmaRule.score}
            onChange={(score) => updatePendingMeta('score', score ?? 0)}
            min={0}
          />
        </InfoValue>
      </Row>
      <Row>
        <InfoHeader>Actions</InfoHeader>
        <InfoValue className="mt-2">
          <ActionsEditor actions={entity.metadata.SigmaRule.actions} onChange={(actions) => updatePendingMeta('actions', actions)} />
        </InfoValue>
      </Row>
    </>
  );
};

const SigmaRuleCreateConfig: EntityCreateConfig<Entities.SigmaRule> = {
  kind: Entities.SigmaRule,
  EntityMetadata: SigmaRuleMetaInfo,
  BlankCreateEntity: BlankCreateSigmaRule,
};

export default SigmaRuleCreateConfig;
