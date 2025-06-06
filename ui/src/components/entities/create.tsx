import { useState, createContext, useContext, JSX } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Alert, Button, Card, Row, Form } from 'react-bootstrap';
import styled from 'styled-components';

// project imports
import { InfoHeader, InfoValue, OverlayTipBottom, Page, SelectInputArray, Title } from '@components';
import { useAuth } from '@utilities';
import { CreateEntity, Entities } from '@models';
import { createEntity } from '@thorpi';
import { buildCreateEntityForm, copyEntityFields } from './utilities';

export type MetadataComponent = (
  entity: CreateEntity,
  onChange: <K extends keyof CreateEntity>(field: K, value: CreateEntity[K]) => void,
) => JSX.Element;

type EntityCreateContextType = {
  kind: Entities;
  entity: CreateEntity;
  metadata: MetadataComponent;
  updatePendingEntity: <K extends keyof CreateEntity>(field: K, value: CreateEntity[K]) => void;
  error: string; // any error message returned when trying to create a new entity
  setError: (error: string) => void; // set create error message callback
};

// Page context
const EntityContext = createContext<EntityCreateContextType | undefined>(undefined);

// custom device create context hook
const useEntityContext = () => {
  const context = useContext(EntityContext);
  if (context === undefined) {
    throw new Error('useEntityContext must be used within a EntityContextProvider');
  }
  return context;
};

// Entity shared fields
const EntityInfo = () => {
  const { entity, metadata, updatePendingEntity } = useEntityContext();
  const { userInfo } = useAuth();
  return (
    <>
      <Card className="panel">
        <Card.Body>
          <Row>
            <InfoHeader>
              Name<sub>*</sub>
            </InfoHeader>
            <InfoValue>
              <Form.Group>
                <Form.Control
                  onChange={(e) => updatePendingEntity('name', String(e.target.value))}
                  value={entity.name ? entity.name : ''}
                ></Form.Control>
              </Form.Group>
            </InfoValue>
          </Row>
          <hr className="my-3" />
          <Row>
            <InfoHeader>
              Groups<sub>*</sub>
            </InfoHeader>
            <InfoValue>
              <SelectInputArray
                isCreatable={false}
                options={userInfo?.groups ? userInfo.groups : []}
                values={entity.groups}
                onChange={(groups) => updatePendingEntity('groups', groups)}
              />
            </InfoValue>
          </Row>
          <hr className="my-3" />
          {metadata(entity, updatePendingEntity)}
          <hr className="my-3" />
          <Row>
            <InfoHeader>Description</InfoHeader>
            <InfoValue>
              <Form.Group>
                <Form.Control
                  onChange={(e) => updatePendingEntity('description', String(e.target.value))}
                  value={entity.description ? entity.description : ''}
                  as="textarea"
                ></Form.Control>
              </Form.Group>
            </InfoValue>
          </Row>
          <Row className="mt-4">
            <InfoHeader />
            <InfoValue>* Field is required to create a new device</InfoValue>
          </Row>
        </Card.Body>
      </Card>
    </>
  );
};

const EntityCreateButton = () => {
  const navigate = useNavigate();
  const { entity, kind, setError } = useEntityContext();
  const { userInfo } = useAuth();
  // user must have roles in one of the groups
  const userCanCreate = true; //TODO grab group membership of selected groups and check roles
  const CreateEntityMessage = userCanCreate
    ? `Create a new ${kind}. You must be a user, manager, or owner in a selected group to create this ${kind}.`
    : `You must be a user, manager, or owner in a selected group to create this ${kind}.`;

  const handleCreateEntity = (): void => {
    createEntity(buildCreateEntityForm(entity, kind), setError).then((response) => {
      if (response != null) {
        navigate(`/${kind}/${response.id}`);
      }
    });
  };

  return (
    <div className="d-flex justify-content-center pt-4">
      <OverlayTipBottom tip={CreateEntityMessage}>
        {/* @ts-ignore*/}
        <Button className="primary-btn" variant="info" disabled={!userCanCreate} onClick={() => handleCreateEntity()}>
          Create
        </Button>
      </OverlayTipBottom>
    </div>
  );
};

const CreateEntityTitle = styled.div`
  display: grid;
  place-items: center;
  padding-bottom: 1rem;
  padding-top: 0.5rem;
`;

type EntityDetailsProps<T extends CreateEntity> = {
  blank: T;
  kind: Entities;
  metadata: MetadataComponent;
};

export function EntityCreate<T extends CreateEntity>({ blank, kind, metadata }: EntityDetailsProps<T>): JSX.Element {
  const { state } = useLocation();
  // pull device state if its there
  const [entity, setEntity] = useState<CreateEntity>(state?.entity ? copyEntityFields(state.entity, blank) : blank);
  const [error, setError] = useState<string>('');
  // Update pending device info fields by key
  function updatePendingEntity<K extends keyof CreateEntity>(field: K, value: CreateEntity[K]): void {
    const updates: CreateEntity = structuredClone(entity);
    updates[field] = value;
    setEntity(updates);
  }

  return (
    <EntityContext.Provider
      value={{
        entity,
        metadata,
        updatePendingEntity,
        error,
        setError,
        kind,
      }}
    >
      <Page className="full-min-width" title={`Create ${kind}`}>
        <CreateEntityTitle>
          <Title className="title">New {`${kind}`}</Title>
        </CreateEntityTitle>
        <EntityInfo />
        {error != '' && <Alert variant="danger text-center">{error}</Alert>}
        <EntityCreateButton />
      </Page>
    </EntityContext.Provider>
  );
}
