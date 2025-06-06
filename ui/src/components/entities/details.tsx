import React, { useEffect, useState, createContext, useContext, Fragment, JSX } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Alert, Button, Card, Row, Col, Form, Modal } from 'react-bootstrap';
import { FaBackspace, FaRegEdit, FaSave, FaTrash } from 'react-icons/fa';
import { FaFileCirclePlus, FaSquarePlus } from 'react-icons/fa6';

// project imports
import {
  AssociationTree,
  AssociationGraph,
  CondensedEntityTags,
  FieldBadge,
  InfoHeader,
  InfoValue,
  OverlayTipBottom,
  Page,
  SelectInputArray,
  Subtitle,
  Title,
} from '@components';
import { useAuth } from '@utilities';
import { Entity } from '@models';
import { deleteEntity, updateEntity } from '@thorpi';
import { buildUpdateEntityForm } from './utilities';
import styled from 'styled-components';

export type MetadataComponent = (
  entity: Entity,
  pendingEntity: Entity,
  handleUpdate: <K extends keyof Entity>(field: K, value: Entity[K]) => void,
  editing: boolean,
) => JSX.Element;

type EntityDetailsContextType = {
  entity: Entity;
  metadata: MetadataComponent;
  pendingEntity: Entity;
  updatePendingEntity: <K extends keyof Entity>(field: K, value: Entity[K]) => void;
  editing: boolean;
  applyEntityUpdates: (entity: Entity, pendingEntity: Entity) => void;
  toggleEditMode: () => void;
  error: string; // any error message returned when trying to update an existing device
  setError: (error: string) => void; // set update error message callback
};

// Page context
const EntityContext = createContext<EntityDetailsContextType | undefined>(undefined);

// custom device create context hook
const useEntityContext = () => {
  const context = useContext(EntityContext);
  if (context === undefined) {
    throw new Error('useEntityContext must be used within a EntityContextProvider');
  }
  return context;
};

const EntityInfo = () => {
  const { entity, metadata, pendingEntity, editing, updatePendingEntity } = useEntityContext();
  const { userInfo } = useAuth();
  return (
    <>
      <Card className="panel">
        <Card.Body>
          {!editing && (
            <>
              <Row>
                <InfoHeader>ID</InfoHeader>
                <Col>{entity.id}</Col>
              </Row>
              <hr className="my-3" />
            </>
          )}
          <Row className="d-flex flex-row justify-content-center">
            <InfoHeader>Name</InfoHeader>
            <InfoValue>
              {editing ? (
                <Form.Group>
                  <Form.Control
                    onChange={(e) => updatePendingEntity('name', String(e.target.value))}
                    value={pendingEntity.name ? pendingEntity.name : ''}
                  ></Form.Control>
                </Form.Group>
              ) : (
                <>{entity.name}</>
              )}
            </InfoValue>
          </Row>
          <hr className="my-3" />
          {!editing && (
            <>
              <Row>
                <InfoHeader>Type</InfoHeader>
                <Col>{entity.kind}</Col>
              </Row>
              <hr className="my-3" />
            </>
          )}
          {metadata(entity, pendingEntity, updatePendingEntity, editing)}
          <hr className="my-3" />
          <Row>
            <InfoHeader>Description</InfoHeader>
            <InfoValue>
              {editing ? (
                <Form.Group>
                  <Form.Control
                    onChange={(e) => updatePendingEntity('description', String(e.target.value))}
                    value={pendingEntity.description ? pendingEntity.description : ''}
                    as="textarea"
                  ></Form.Control>
                </Form.Group>
              ) : (
                <>{entity.description}</>
              )}
            </InfoValue>
          </Row>
          <hr className="my-3" />
          {!editing && (
            <>
              <Row>
                <InfoHeader>Submitter</InfoHeader>
                <Col>{entity.submitter}</Col>
              </Row>
              <hr className="my-3" />
            </>
          )}
          <Row>
            <InfoHeader>Groups</InfoHeader>
            <InfoValue>
              {editing ? (
                <SelectInputArray
                  isCreatable={false}
                  options={userInfo?.groups ? userInfo.groups : []}
                  values={pendingEntity.groups}
                  onChange={(groups) => updatePendingEntity('groups', groups)}
                />
              ) : (
                <FieldBadge color="gray" field={entity.groups} />
              )}
            </InfoValue>
          </Row>
          {!editing && (
            <>
              <hr className="my-3" />
              <Row>
                <InfoHeader>Created On</InfoHeader>
                <Col>{entity.created}</Col>
              </Row>
            </>
          )}
        </Card.Body>
      </Card>
      {entity.tags != undefined && (
        <Card className="panel mt-4">
          <Card.Body>
            <Row>
              <center>
                <Subtitle>Tags</Subtitle>
              </center>
              <CondensedEntityTags resource={entity.kind} tags={entity.tags} />
            </Row>
          </Card.Body>
        </Card>
      )}
    </>
  );
};

type ButtonProps = {
  disabled: boolean;
};

const DeleteButton: React.FC<ButtonProps> = ({ disabled }) => {
  const navigate = useNavigate();
  const { entity, setError } = useEntityContext();
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [disableConfirmButton, setDisableConfirmButton] = useState(false);
  //close delete modal
  const handleCloseDeleteModal = (): void => {
    setShowDeleteModal(false);
  };
  // open delete modal
  const handleOpenDeleteModal = (): void => {
    setShowDeleteModal(true);
  };
  // handle removal of devices using trash button
  const handleRemoveClick = async () => {
    setDisableConfirmButton(true);
    deleteEntity(entity.id, setError).then((res) => {
      if (res === true) {
        setShowDeleteModal(false);
        navigate(`/${entity.kind.toLowerCase()}s`);
      }
    });
  };

  return (
    <>
      <OverlayTipBottom tip={`Delete "${entity.name}"`}>
        {/* @ts-ignore */}
        <Button className="icon-btn mx-1" variant="" disabled={disabled} onClick={handleOpenDeleteModal}>
          <FaTrash size={20} />
        </Button>
      </OverlayTipBottom>
      <Modal show={showDeleteModal} onHide={handleCloseDeleteModal} backdrop="static" keyboard={false}>
        <Modal.Header closeButton>
          <Modal.Title>Confirm deletion</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <div className="text-center">
            Do you really want to delete this {`${entity.kind}`}?
            <div className="pt-4">
              <b>{entity.name}</b>
            </div>
            <b>{entity.id}</b>
          </div>
        </Modal.Body>
        <Modal.Footer className="d-flex justify-content-center">
          {/* @ts-ignore */}
          <Button className="danger-btn" onClick={handleRemoveClick} disabled={disableConfirmButton}>
            Confirm
          </Button>
          <Button className="primary-btn" onClick={handleCloseDeleteModal}>
            Cancel
          </Button>
        </Modal.Footer>
      </Modal>
    </>
  );
};

const UploadFileButton: React.FC<ButtonProps> = ({ disabled }) => {
  const navigate = useNavigate();
  const { entity } = useEntityContext();
  return (
    <>
      <OverlayTipBottom tip={`Upload files associated with this ${entity.kind}.`}>
        {/*@ts-ignore*/}
        <Button className="icon-btn mx-1" variant="" disabled={disabled} onClick={() => navigate(`/upload`, { state: { entity: entity } })}>
          <FaFileCirclePlus size={20} />
        </Button>
      </OverlayTipBottom>
    </>
  );
};

const CreateButton: React.FC<ButtonProps> = ({ disabled }) => {
  const navigate = useNavigate();
  const { entity } = useEntityContext();
  return (
    <>
      <OverlayTipBottom tip={`Copy this ${entity.kind}.`}>
        {/*@ts-ignore*/}
        <Button
          className="icon-btn mx-1"
          variant=""
          disabled={disabled}
          onClick={() => navigate(`/create/${entity.kind.toLowerCase()}`, { state: { entity: entity } })}
        >
          <FaSquarePlus size={20} />
        </Button>
      </OverlayTipBottom>
    </>
  );
};

const EditButton: React.FC<ButtonProps> = ({ disabled }) => {
  const { entity, pendingEntity, applyEntityUpdates, editing, toggleEditMode } = useEntityContext();
  return (
    <Fragment>
      <OverlayTipBottom tip={editing ? `Cancel changes to "${entity.name}"` : `Edit "${entity.name}"`}>
        <Button className="icon-btn mx-1" variant="" disabled={disabled} onClick={toggleEditMode}>
          {editing ? <FaBackspace size="24" /> : <FaRegEdit size="24" />}
        </Button>
      </OverlayTipBottom>
      {editing && (
        <OverlayTipBottom tip={`Save pending changes`}>
          <Button className="icon-btn mx-1" variant="" disabled={disabled} onClick={() => applyEntityUpdates(entity, pendingEntity)}>
            <FaSave size={20} />
          </Button>
        </OverlayTipBottom>
      )}
    </Fragment>
  );
};

interface DeviceButtonProps {
  className?: string;
}

const EntityButtons: React.FC<DeviceButtonProps> = ({ className }) => {
  const { userInfo } = useAuth();
  const { entity } = useEntityContext();
  // submitter and User++ can modify an entity
  // modify/create needs to have permissions within the group no in Thorium
  const canModify = true; // TODO: grab group membership of selected groups and check roles
  const canCreate = true; // TODO: any user can create, but they may not be able to submit to the selected groups
  return (
    <div className={`d-flex justify-content-center ${className}`}>
      <EditButton disabled={!canModify} />
      <DeleteButton disabled={!canModify} />
      <CreateButton disabled={!canCreate} />
      <UploadFileButton disabled={!canCreate} />
    </div>
  );
};

type EntityHeaderProps = {
  icon: (size: number) => JSX.Element; // default icon to display
};

const IconTitle = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
`;

const EntityHeader: React.FC<EntityHeaderProps> = ({ icon }) => {
  const { entity } = useEntityContext();
  return (
    <Card className="panel">
      <Card.Body>
        <IconTitle>
          <div className="pe-8">{icon(100)}</div>
          <Title className="title">{entity.name}</Title>
        </IconTitle>
      </Card.Body>
    </Card>
  );
};

type EntityDetailsProps<T extends Entity> = {
  getEntityDetails: (entityID: string, setError: (err: string) => void, updateEntity: (entity: T) => void) => void;
  blank: T;
  metadata: MetadataComponent;
  icon: (size: number) => JSX.Element;
};

export function EntityDetails<T extends Entity>({ getEntityDetails, blank, metadata, icon }: EntityDetailsProps<T>): JSX.Element {
  const { entityID } = useParams();
  // pull device state if its there
  const [entity, setEntity] = useState<Entity>(blank);
  const [pendingEntity, setPendingEntity] = useState<Entity>(blank);
  const [error, setError] = useState<string>('');
  const [editing, setEditing] = useState(false);

  // Switch between edit and view modes
  const toggleEditMode = () => {
    setEditing((prev) => !prev);
  };

  // Update pending device info fields by key
  function updatePendingEntity<K extends keyof Entity>(field: K, value: Entity[K]): void {
    const updates: Entity = structuredClone(pendingEntity);
    updates[field] = value;
    setPendingEntity(updates);
  }

  // update entity state
  const handleEntityUpdate = (newEntity: T) => {
    setEntity(newEntity);
    setPendingEntity(newEntity);
  };

  // Send device updates to the API
  const applyEntityUpdates = (entity: Entity, pendingEntity: Entity) => {
    if (entityID) {
      updateEntity(entityID, buildUpdateEntityForm(entity, pendingEntity), setError).then((success: boolean) => {
        if (success) {
          setEntity(pendingEntity);
          toggleEditMode();
          getEntityDetails(entityID, setError, handleEntityUpdate);
        }
      });
    }
  };

  useEffect(() => {
    // get entity details on page load
    if (entityID !== undefined) {
      getEntityDetails(entityID, setError, handleEntityUpdate);
    }
  }, []);

  return (
    <EntityContext.Provider
      value={{
        entity,
        metadata,
        pendingEntity,
        updatePendingEntity,
        applyEntityUpdates,
        editing,
        toggleEditMode,
        error,
        setError,
      }}
    >
      <Page className="full-min-width" title={`${entity.kind} · ${entity.name}`}>
        <EntityHeader icon={icon} />
        <EntityButtons className="py-3" />
        {error != '' && (
          <Alert variant="danger" className="text-center mb-3">
            {error}
          </Alert>
        )}
        <EntityInfo />
        <Card className="panel mt-4">
          <Card.Body>
            <div className="d-flex justify-content-center">
              <Subtitle>Graph</Subtitle>
            </div>
            {entityID && <AssociationGraph inView initial={{ entities: [entityID ? entityID : ''] }} />}
          </Card.Body>
        </Card>
        <Card className="panel mt-4">
          <Card.Body>
            <div className="text-center">
              <Subtitle>Associations</Subtitle>
            </div>
            {entityID && <AssociationTree initial={{ entities: [entityID ? entityID : ''] }} />}
          </Card.Body>
        </Card>
      </Page>
    </EntityContext.Provider>
  );
}
