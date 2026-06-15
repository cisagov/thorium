import React, { useState, createContext, useContext, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Button, Card, Row, Form } from 'react-bootstrap';
import styled from 'styled-components';

// project imports
import { EntityCreateConfig } from './configs/config';
import InfoHeader from '../shared/InfoHeader';
import InfoValue from '../shared/InfoValue';
import EntityGraphicUpload from '../shared/EntityGraphicUpload';
import { buildCreateEntityForm, copyEntityFields } from '../utilities';
import Page from '@components/pages/Page';
import Title from '@components/shared/titles/Title';
import { OverlayTipBottom } from '@components/shared/overlay/tips';
import SelectInputArray from '@components/shared/inputs/selectable/SelectInputArray';
import { createEntity } from '@thorpi/entities';
import { useAuth } from '@utilities/auth';
import { entityLabel, EntityCreateTypeMap, EntityTypeMap, UISupportedEntityCreateKind } from '@models/entities/entities';
import { getDetailsBasePathByEntity } from '../details/EntityDetailsRoutes';
import AlertBanner from '@components/shared/alerts/AlertBanner';

// spec: ./EntityCreate.spec.md

export type CreateMetadataProps<K extends UISupportedEntityCreateKind> = {
  entity: EntityCreateTypeMap[K];
  onChange: <F extends keyof EntityCreateTypeMap[K]>(field: F, value: EntityCreateTypeMap[K][F]) => void;
};

export type CreateMetadataComponent<K extends UISupportedEntityCreateKind> = React.ComponentType<CreateMetadataProps<K>>;

type EntityCreateContextType<K extends UISupportedEntityCreateKind> = {
  kind: K;
  entity: EntityCreateTypeMap[K];
  Metadata: CreateMetadataComponent<K>;
  updatePendingEntity: <F extends keyof EntityCreateTypeMap[K]>(field: F, value: EntityCreateTypeMap[K][F]) => void;
  error: string;
  setError: (error: string) => void;
  supportsGraphic: boolean;
  graphicFile: File | null;
  setGraphicFile: (file: File | null) => void;
};

export function createEntityCreatePage<K extends UISupportedEntityCreateKind>(config: EntityCreateConfig<K>) {
  const EntityCreateContext = createContext<EntityCreateContextType<K> | undefined>(undefined);
  const useEntityContext = () => {
    const context = useContext(EntityCreateContext);
    if (context === undefined) {
      throw new Error('useEntityContext must be used within an EntityCreateContext.Provider');
    }
    return context;
  };

  const EntityInfo = () => {
    const { entity, Metadata, updatePendingEntity, supportsGraphic, graphicFile, setGraphicFile } = useEntityContext();
    const { userInfo } = useAuth();
    const handleGraphicChange = useCallback((file: File | null) => setGraphicFile(file), [setGraphicFile]);
    return (
      <Card className="panel">
        <Card.Body>
          <Row>
            <InfoHeader>
              Name<sub>*</sub>
            </InfoHeader>
            <InfoValue>
              <Form.Group>
                <Form.Control onChange={(e) => updatePendingEntity('name', String(e.target.value))} value={entity.name ?? ''} />
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
                options={[...(userInfo?.groups ?? [])].sort((a, b) => a.localeCompare(b))}
                values={entity.groups}
                onChange={(groups) => updatePendingEntity('groups', groups)}
              />
            </InfoValue>
          </Row>
          <hr className="my-3" />
          <Metadata entity={entity} onChange={updatePendingEntity} />
          <hr className="my-3" />
          <Row>
            <InfoHeader>Description</InfoHeader>
            <InfoValue>
              <Form.Group>
                <Form.Control
                  onChange={(e) => updatePendingEntity('description', String(e.target.value))}
                  value={entity.description ?? ''}
                  as="textarea"
                />
              </Form.Group>
            </InfoValue>
          </Row>
          {supportsGraphic && (
            <>
              <hr className="my-3" />
              <Row>
                <InfoHeader>Graphic</InfoHeader>
                <InfoValue>
                  <EntityGraphicUpload file={graphicFile} onChange={handleGraphicChange} />
                </InfoValue>
              </Row>
            </>
          )}
          <Row className="mt-4">
            <InfoHeader />
            <InfoValue>* Field is required to create a new {entityLabel(entity.kind).toLowerCase()}</InfoValue>
          </Row>
        </Card.Body>
      </Card>
    );
  };

  const EntityCreateButton = () => {
    const navigate = useNavigate();
    const { entity, kind, setError, graphicFile } = useEntityContext();
    const userCanCreate = true;
    const createEntityMessage = userCanCreate
      ? `Create a new ${entityLabel(kind)}. You must be a user, manager, or owner in a selected group to create this ${entityLabel(kind)}.`
      : `You must be a user, manager, or owner in a selected group to create this ${entityLabel(kind)}.`;
    const handleCreateEntity = (): void => {
      void createEntity(buildCreateEntityForm(entity, graphicFile ?? undefined), setError).then((response) => {
        if (response != null) {
          void navigate(`${getDetailsBasePathByEntity(kind)}/${response.id}`);
        }
      });
    };
    return (
      <div className="d-flex justify-content-center pt-4">
        <OverlayTipBottom tip={createEntityMessage}>
          <Button className="secondary-btn" variant="info" disabled={!userCanCreate} onClick={handleCreateEntity}>
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

  const BoundEntityCreatePage = () => {
    const { state } = useLocation() as { state: { entity?: EntityTypeMap[K] } | null };
    const initialEntity: EntityCreateTypeMap[K] =
      state?.entity && state.entity.kind === config.kind
        ? (copyEntityFields(state.entity, config.BlankCreateEntity) as EntityCreateTypeMap[K])
        : config.BlankCreateEntity;
    const [entity, setEntity] = useState<EntityCreateTypeMap[K]>(initialEntity);
    const [error, setError] = useState<string>('');
    const [graphicFile, setGraphicFile] = useState<File | null>(null);
    function updatePendingEntity<F extends keyof EntityCreateTypeMap[K]>(field: F, value: EntityCreateTypeMap[K][F]): void {
      const updates = structuredClone(entity);
      updates[field] = value;
      setEntity(updates);
    }
    return (
      <EntityCreateContext.Provider
        value={{
          entity,
          Metadata: config.EntityMetadata,
          updatePendingEntity,
          error,
          setError,
          kind: config.kind,
          supportsGraphic: config.supportsGraphic ?? true,
          graphicFile,
          setGraphicFile,
        }}
      >
        <Page className="full-min-width" title={`Create ${entityLabel(config.kind)}`}>
          <CreateEntityTitle>
            <Title className="title">New {entityLabel(config.kind)}</Title>
          </CreateEntityTitle>
          <EntityInfo />
          {error !== '' && <AlertBanner>{error}</AlertBanner>}
          <EntityCreateButton />
        </Page>
      </EntityCreateContext.Provider>
    );
  };
  return BoundEntityCreatePage;
}

export default createEntityCreatePage;
