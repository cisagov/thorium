import React, { useEffect, useState } from 'react';
import styled from 'styled-components';

// project imports
import { Button } from '@components/shared/buttons';
import { ButtonVariant } from '@components/shared/buttons/types';
import SelectInputArray from '@components/shared/inputs/selectable/SelectInputArray';
import { AssociationKind, associationKindLabel } from '@models/associations';

const Backdrop = styled.div`
  position: fixed;
  inset: 0;
  z-index: 5000;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.5);
`;

const Dialog = styled.div`
  width: 460px;
  max-width: 90vw;
  background: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 8px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
  padding: 18px 20px;
  color: var(--thorium-text);
`;

const Title = styled.h5`
  margin: 0 0 12px;
  font-weight: 600;
`;

const Body = styled.div`
  font-size: 0.9rem;
  margin-bottom: 16px;
`;

const Field = styled.label`
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 12px;
  font-size: 0.85rem;
  color: var(--thorium-secondary-text);
`;

const Select = styled.select`
  padding: 6px 8px;
  border-radius: 6px;
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-text);
  border: 1px solid var(--thorium-panel-border);
  font-size: 0.9rem;

  &:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px var(--thorium-highlight-text);
  }
`;

const Actions = styled.div`
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 20px;
`;

export interface AssociationKindModalProps {
  show: boolean;
  /** Name of the entity being associated, shown in the prompt. */
  entityName: string;
  /** Groups to pre-populate the group selector with (e.g. the tool result's groups). */
  defaultGroups: string[];
  /** All groups offered as options (e.g. the groups the user is a member of). */
  groupOptions: string[];
  /** Called with the chosen association kind and groups when confirmed. */
  onConfirm: (kind: AssociationKind, groups: string[]) => void;
  onCancel: () => void;
}

/**
 * Prompt for the {@link AssociationKind} and the groups to use when creating an entity and
 * associating it with the current file. Styled-components dialog (no react-bootstrap). Defaults to
 * `AssociatedWith` and the supplied default groups.
 */
const AssociationKindModal: React.FC<AssociationKindModalProps> = ({
  show,
  entityName,
  defaultGroups,
  groupOptions,
  onConfirm,
  onCancel,
}) => {
  const [kind, setKind] = useState<AssociationKind>(AssociationKind.AssociatedWith);
  const [groups, setGroups] = useState<string[]>(defaultGroups);

  // reset to the defaults each time the prompt opens (it may reopen for a different entity)
  useEffect(() => {
    if (show) {
      setKind(AssociationKind.AssociatedWith);
      setGroups(defaultGroups);
    }
  }, [show, defaultGroups]);

  if (!show) return null;
  return (
    <Backdrop onMouseDown={onCancel}>
      <Dialog onMouseDown={(e) => e.stopPropagation()}>
        <Title>Create &amp; associate entity</Title>
        <Body>
          Create <b>{entityName}</b> and associate it with this file.
        </Body>
        <Field>
          Association type
          <Select value={kind} onChange={(e) => setKind(e.target.value as AssociationKind)}>
            {Object.values(AssociationKind).map((k) => (
              <option key={k} value={k}>
                {associationKindLabel(k)}
              </option>
            ))}
          </Select>
        </Field>
        <Field>
          Groups
          <SelectInputArray values={groups} options={groupOptions} isCreatable={false} onChange={(next) => setGroups(next)} />
        </Field>
        <Actions>
          <Button variant={ButtonVariant.Ghost} onClick={onCancel}>
            Cancel
          </Button>
          <Button variant={ButtonVariant.Primary} disabled={groups.length === 0} onClick={() => onConfirm(kind, groups)}>
            Create
          </Button>
        </Actions>
      </Dialog>
    </Backdrop>
  );
};

export default AssociationKindModal;
