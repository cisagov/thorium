// EntityDetails.tsx
import React, { useEffect, useState, createContext, useContext, Fragment, JSX, useMemo, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Button, Card, Row, Col, Form, Modal } from 'react-bootstrap';
import { FaBackspace, FaRegEdit, FaSave, FaTrash } from 'react-icons/fa';
import { FaFileCirclePlus, FaSquarePlus } from 'react-icons/fa6';
import styled from 'styled-components';

// project imports
import ListCollectionButton from './ListCollectionsButton';
import { EntityDetailsConfig } from './configs/configs';
const AssociationGraph = React.lazy(() => import('@components/associations/graph/AssociationGraph'));
import { buildUpdateEntityForm } from '../utilities';
import Markdown from '@components/shared/syntax/Markdown';
import InfoHeader from '../shared/InfoHeader';
import InfoValue from '../shared/InfoValue';
import EntityGraphicUpload from '../shared/EntityGraphicUpload';
import Page from '@components/pages/Page';
import Title from '@components/shared/titles/Title';
import Subtitle from '@components/shared/titles/Subtitle';
import FieldBadge from '@components/shared/badges/FieldBadge';
import { OverlayTipBottom } from '@components/shared/overlay/tips';
import SelectInputArray from '@components/shared/inputs/selectable/SelectInputArray';
import CondensedEntityTags from '@components/tags/condensed/CondensedEntityTags';
import { useAuth } from '@utilities/auth';
import { deleteEntity, updateEntity, fetchEntityImage } from '@thorpi/entities';
import { CollectionKind } from '@models/entities/collections';
import { Entities, EntityTypeMap } from '@models/entities';
import { ButtonProps } from '@models/components';
import { GraphDataProvider } from '@components/associations/data/GraphDataContext';
import { getBrowsingPathByEntity } from '../browsing/EntityBrowsingRoutes';
import AlertBanner from '@components/shared/alerts/AlertBanner';

export type DetailsMetadataProps<T extends keyof EntityTypeMap> = {
  entity: EntityTypeMap[T];
  pendingEntity: EntityTypeMap[T];
  handleUpdate: <K extends keyof EntityTypeMap[T]>(field: K, value: EntityTypeMap[T][K]) => void;
  editing: boolean;
};

export type MetadataComponent<K extends keyof EntityTypeMap> = React.ComponentType<DetailsMetadataProps<K>>;

type EntityDetailsContextType<T extends keyof EntityTypeMap> = {
  entity: EntityTypeMap[T];
  Metadata: MetadataComponent<T>;
  pendingEntity: EntityTypeMap[T];
  updatePendingEntity: <K extends keyof EntityTypeMap[T]>(field: K, value: EntityTypeMap[T][K]) => void;
  editing: boolean;
  applyEntityUpdates: (entity: EntityTypeMap[T], pendingEntity: EntityTypeMap[T]) => Promise<void>;
  toggleEditMode: () => void;
  error: string;
  setError: (error: string) => void;
  supportsGraphic: boolean;
  graphicFile: File | null;
  setGraphicFile: (file: File | null) => void;
  clearGraphic: boolean;
  setClearGraphic: (clear: boolean) => void;
  existingImageUrl: string | null;
  existingImageIsSvg: boolean;
};

type EntityHeaderProps = {
  icon: (size: number) => JSX.Element;
  className?: string;
};

const EntityGraphic = styled.img`
  width: 80px;
  height: 80px;
  object-fit: contain;
  border-radius: 8px;
  background-color: var(--thorium-panel-bg);
`;

const SvgGraphic = styled.div<{ $src: string }>`
  width: 80px;
  height: 80px;
  background-color: currentColor;
  mask-image: url(${(props) => props.$src});
  mask-size: contain;
  mask-repeat: no-repeat;
  mask-position: center;
  -webkit-mask-image: url(${(props) => props.$src});
  -webkit-mask-size: contain;
  -webkit-mask-repeat: no-repeat;
  -webkit-mask-position: center;
`;

export function createEntityDetailsPage<T extends keyof EntityTypeMap>(config: EntityDetailsConfig<T>) {
  const EntityDetailsContext = createContext<EntityDetailsContextType<T> | undefined>(undefined);

  const useEntityContext = () => {
    const context = useContext(EntityDetailsContext);
    if (context === undefined) {
      throw new Error('useEntityContext must be used within an EntityDetailsContext.Provider');
    }
    return context;
  };

  const EntityInfo = () => {
    const {
      entity,
      Metadata,
      pendingEntity,
      editing,
      updatePendingEntity,
      supportsGraphic,
      graphicFile,
      setGraphicFile,
      clearGraphic,
      setClearGraphic,
      existingImageUrl,
    } = useEntityContext();
    const { userInfo } = useAuth();

    const handleGraphicChange = useCallback(
      (file: File | null) => {
        setGraphicFile(file);
        if (file) setClearGraphic(false);
      },
      [setGraphicFile, setClearGraphic],
    );

    const handleClearExisting = useCallback(() => {
      setClearGraphic(true);
      setGraphicFile(null);
    }, [setClearGraphic, setGraphicFile]);
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
                    <Form.Control onChange={(e) => updatePendingEntity('name', String(e.target.value))} value={pendingEntity.name ?? ''} />
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
            <Metadata entity={entity} pendingEntity={pendingEntity} handleUpdate={updatePendingEntity} editing={editing} />
            <hr className="my-3" />
            <Row>
              <InfoHeader>Description</InfoHeader>
              <InfoValue>
                {editing ? (
                  <Form.Group>
                    <Form.Control
                      onChange={(e) => updatePendingEntity('description', String(e.target.value))}
                      value={pendingEntity.description ?? ''}
                      as="textarea"
                    />
                  </Form.Group>
                ) : (
                  <Markdown>{entity.description ?? ''}</Markdown>
                )}
              </InfoValue>
            </Row>
            {supportsGraphic && editing && (
              <>
                <hr className="my-3" />
                <Row>
                  <InfoHeader>Graphic</InfoHeader>
                  <InfoValue>
                    <EntityGraphicUpload
                      file={graphicFile}
                      onChange={handleGraphicChange}
                      existingImageUrl={clearGraphic ? null : existingImageUrl}
                      onClearExisting={handleClearExisting}
                    />
                  </InfoValue>
                </Row>
              </>
            )}
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
                    options={userInfo?.groups ?? []}
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
        {entity.tags !== undefined && (
          <Card className="panel mt-4">
            <Card.Body>
              <center>
                <Subtitle>Tags</Subtitle>
              </center>
              <CondensedEntityTags resource={entity.kind} tags={entity.tags} />
            </Card.Body>
          </Card>
        )}
      </>
    );
  };

  const DeleteButton: React.FC<ButtonProps> = ({ disabled = false }) => {
    const navigate = useNavigate();
    const { entity, setError } = useEntityContext();
    const [showDeleteModal, setShowDeleteModal] = useState(false);
    const [disableConfirmButton, setDisableConfirmButton] = useState(false);

    const handleRemoveClick = async () => {
      setDisableConfirmButton(true);
      const res = await deleteEntity(entity.id, setError);
      if (res === true) {
        setShowDeleteModal(false);
        void navigate(`${getBrowsingPathByEntity(entity.kind)}`);
      }
    };

    return (
      <>
        <OverlayTipBottom tip={`Delete "${entity.name}"`}>
          <Button className="icon-btn mx-1" variant="" disabled={disabled} onClick={() => setShowDeleteModal(true)}>
            <FaTrash size={20} />
          </Button>
        </OverlayTipBottom>

        <Modal show={showDeleteModal} onHide={() => setShowDeleteModal(false)} backdrop="static" keyboard={false}>
          <Modal.Header closeButton>
            <Modal.Title>Confirm deletion</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <div className="text-center">
              Do you really want to delete this {entity.kind}?
              <div className="pt-4">
                <b>{entity.name}</b>
              </div>
              <b>{entity.id}</b>
            </div>
          </Modal.Body>
          <Modal.Footer className="d-flex justify-content-center">
            <Button
              className="danger-btn"
              onClick={() => {
                void handleRemoveClick();
              }}
              disabled={disableConfirmButton}
            >
              Confirm
            </Button>
            <Button className="primary-btn" onClick={() => setShowDeleteModal(false)}>
              Cancel
            </Button>
          </Modal.Footer>
        </Modal>
      </>
    );
  };

  const UploadFileButton: React.FC<ButtonProps> = ({ disabled = false }) => {
    const navigate = useNavigate();
    const { entity } = useEntityContext();

    return (
      <OverlayTipBottom tip={`Upload files associated with this ${entity.kind}.`}>
        <Button
          className="icon-btn mx-1"
          variant=""
          disabled={disabled}
          onClick={() => {
            void navigate('/analyze', { state: { entity } });
          }}
        >
          <FaFileCirclePlus size={20} />
        </Button>
      </OverlayTipBottom>
    );
  };

  const CreateButton: React.FC<ButtonProps> = ({ className, disabled = false }) => {
    const navigate = useNavigate();
    const { entity } = useEntityContext();

    return (
      <OverlayTipBottom tip={`Copy this ${entity.kind}.`}>
        <Button
          className={`${className} icon-btn mx-1`}
          variant=""
          disabled={disabled}
          onClick={() => {
            void navigate(`/create/${entity.kind.toLowerCase()}`, { state: { entity } });
          }}
        >
          <FaSquarePlus size={20} />
        </Button>
      </OverlayTipBottom>
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
          <OverlayTipBottom tip="Save pending changes">
            <Button
              className="icon-btn mx-1"
              variant=""
              disabled={disabled}
              onClick={() => {
                void applyEntityUpdates(entity, pendingEntity);
              }}
            >
              <FaSave size={20} />
            </Button>
          </OverlayTipBottom>
        )}
      </Fragment>
    );
  };

  const EntityButtons: React.FC<ButtonProps> = ({ className }) => {
    const { entity } = useEntityContext();
    const canModify = true;
    const canCreate = true;

    return (
      <div className={`d-flex justify-content-center ${className}`}>
        <EditButton disabled={!canModify} />
        <DeleteButton disabled={!canModify} />
        <CreateButton disabled={!canCreate} />
        <UploadFileButton disabled={!canCreate} />

        {entity.kind === Entities.Collection && entity.metadata.Collection.collection_kind === CollectionKind.Files && (
          <ListCollectionButton collection={entity} />
        )}
      </div>
    );
  };

  const IconTitle = styled.div`
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    color: var(--thorium-text);
  `;

  const EntityHeader: React.FC<EntityHeaderProps> = ({ className, icon }) => {
    const { entity, existingImageUrl, existingImageIsSvg, graphicFile } = useEntityContext();
    const headerImageUrl = graphicFile ? URL.createObjectURL(graphicFile) : existingImageUrl;
    const isSvg = !graphicFile && existingImageIsSvg;

    return (
      <Card className={`panel ${className ?? ''}`}>
        <Card.Body>
          <IconTitle>
            <div className="pe-8">
              {headerImageUrl ? (
                isSvg ? (
                  <SvgGraphic $src={headerImageUrl} />
                ) : (
                  <EntityGraphic src={headerImageUrl} alt={entity.name} />
                )
              ) : (
                icon(100)
              )}
            </div>
            <Title className="title">{entity.name}</Title>
          </IconTitle>
        </Card.Body>
      </Card>
    );
  };

  const BoundEntityDetailsPage = () => {
    const { entityID } = useParams();
    const [entity, setEntity] = useState<EntityTypeMap[T]>(config.BlankEntity);
    const [pendingEntity, setPendingEntity] = useState<EntityTypeMap[T]>(config.BlankEntity);
    const [error, setError] = useState('');
    const [editing, setEditing] = useState(false);
    const [graphicFile, setGraphicFile] = useState<File | null>(null);
    const [clearGraphic, setClearGraphic] = useState(false);
    const [existingImageUrl, setExistingImageUrl] = useState<string | null>(null);
    const [existingImageIsSvg, setExistingImageIsSvg] = useState(false);

    const toggleEditMode = () => {
      setEditing((prev) => {
        if (prev) {
          setGraphicFile(null);
          setClearGraphic(false);
        }
        return !prev;
      });
    };

    function updatePendingEntity<K extends keyof EntityTypeMap[T]>(field: K, value: EntityTypeMap[T][K]) {
      const updates: EntityTypeMap[T] = structuredClone(pendingEntity);
      updates[field] = value;
      setPendingEntity(updates);
    }

    const handleEntityUpdate = (newEntity: EntityTypeMap[T]) => {
      setEntity(newEntity);
      setPendingEntity(newEntity);
    };

    const applyEntityUpdates = async (entity: EntityTypeMap[T], pendingEntity: EntityTypeMap[T]) => {
      if (entityID) {
        const success = await updateEntity(entityID, buildUpdateEntityForm(entity, pendingEntity, graphicFile, clearGraphic), setError);
        if (success) {
          setEntity(pendingEntity);
          setGraphicFile(null);
          setClearGraphic(false);
          toggleEditMode();
          config.getEntityDetails(entityID, setError, handleEntityUpdate);
          if (config.supportsGraphic) {
            const img = await fetchEntityImage(entityID);
            setExistingImageUrl(img?.url ?? null);
            setExistingImageIsSvg(img?.isSvg ?? false);
          }
        }
      }
    };

    useEffect(() => {
      if (entityID) {
        config.getEntityDetails(entityID, setError, handleEntityUpdate);
        if (config.supportsGraphic) {
          void fetchEntityImage(entityID).then((img) => {
            setExistingImageUrl(img?.url ?? null);
            setExistingImageIsSvg(img?.isSvg ?? false);
          });
        }
      }
    }, [entityID]);

    const associationInitial = useMemo(() => ({ entities: entityID ? [entityID] : [] }), [entityID]);

    return (
      <EntityDetailsContext.Provider
        value={{
          entity,
          Metadata: config.EntityMetaInfo,
          pendingEntity,
          updatePendingEntity,
          editing,
          applyEntityUpdates,
          toggleEditMode,
          error,
          setError,
          supportsGraphic: config.supportsGraphic ?? false,
          graphicFile,
          setGraphicFile,
          clearGraphic,
          setClearGraphic,
          existingImageUrl,
          existingImageIsSvg,
        }}
      >
        <Page className="full-min-width" title={`${entity.kind} · ${entity.name}`}>
          <EntityHeader icon={config.icon} />
          <EntityButtons className="py-3" />
          {error !== '' && <AlertBanner className="mb-3">{error}</AlertBanner>}
          <EntityInfo />
          {entityID && (
            <GraphDataProvider initial={associationInitial}>
              <Card className="panel mt-4">
                <Card.Body>
                  <div className="d-flex justify-content-center">
                    <Subtitle>Associations</Subtitle>
                  </div>
                  <AssociationGraph inView bordered={false} />
                </Card.Body>
              </Card>
            </GraphDataProvider>
          )}
        </Page>
      </EntityDetailsContext.Provider>
    );
  };

  return BoundEntityDetailsPage;
}
