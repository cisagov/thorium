import { useCallback, useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Button, Row, Col } from 'react-bootstrap';
import { FaAngleDown, FaAngleUp } from 'react-icons/fa';

// project imports
import {
  Fields,
  Arguments,
  Dependencies,
  Resources,
  EnvironmentVariables,
  Volumes,
  NetworkPolicies,
  SecurityContext,
  OutputCollection,
  ChildFilters,
  KVM,
  Modifiers,
  ImageFormMode,
} from '@components/pages/images';
import type { FieldsValue, ResourcesValue } from '@components/pages/images';
import { ImageCreateWrapper, ImageFieldsSection, AdvancedHidden } from '@components/pages/images/shared.styled';
import Page from '@components/pages/Page';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import AlertBanner from '@components/shared/alerts/AlertBanner';
import ImagePipelineEditor from '@components/shared/inputs/code/CodeEditor/ImagePipelineEditor';
import FormatToggle from '@components/shared/inputs/code/CodeEditor/FormatToggle';
import ViewModeToggle, { ViewMode } from '@components/shared/inputs/code/CodeEditor/ViewModeToggle';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import { useAuth } from '@utilities/auth';
import { fetchImages, fetchGroups } from '@utilities/fetch';
import { ImageChecker } from '@utilities/rules/tools/image';
import { imageToEditorObject, editorObjectToImageCreate } from '@utilities/transforms/image';
import { createImage } from '@thorpi/images';
import type {
  Image,
  ImageArgs,
  Dependencies as DependenciesType,
  SecurityContext as SecurityContextType,
  ChildFilters as ChildFiltersType,
  Kvm,
  ImageRequest,
} from '@models/images';
import { BlankOutputCollection, ImageScaler } from '@models/images';
import type { Volume } from '@models/volumes';
import type { OutputCollection as OutputCollectionType } from '@models/results';
import type { Group } from '@models/groups';
import { RoleKey } from '@models/users';
import { getThoriumRole } from '@utilities/role';
import { FormatType } from '@utilities/rules/types';

const imageChecker = new ImageChecker();

const IMAGE_CREATE_TEMPLATE: Record<string, unknown> = {
  group: '',
  name: '',
  scaler: 'K8s',
  image: '',
  timeout: 300,
  display_type: 'JSON',
};

const FIELDS_KEYS = new Set([
  'name',
  'group',
  'version',
  'description',
  'scaler',
  'image',
  'timeout',
  'lifetime',
  'display_type',
  'spawn_limit',
  'collect_logs',
  'generator',
]);

type EnvValue = Record<string, string | null>;

const ImageCreate: React.FC = () => {
  const navigate = useNavigate();
  const { state } = useLocation() as { state: Image | null };
  const { userInfo, checkCookie } = useAuth();

  const [hideAdvanced, setHideAdvanced] = useState(true);
  const [displayErrors, setDisplayErrors] = useState(false);
  const [createImageErrors, setCreateImageErrors] = useState('');
  const [loading, setLoading] = useState(false);
  const [groups, setGroups] = useState<string[]>([]);
  const [images, setImages] = useState<string[]>([]);
  let cancelUpdate = false;

  // Single source of truth: both form and editor views read/write this object
  const [editorObj, setEditorObj] = useState<Record<string, unknown>>(state ? imageToEditorObject(state) : IMAGE_CREATE_TEMPLATE);
  const [viewMode, setViewMode] = useState<ViewMode>(ViewMode.Form);
  const [editorFormat, setEditorFormat] = useState<FormatType>(FormatType.YAML);
  const [editorParseValid, setEditorParseValid] = useState(false);

  const [imageFieldErrors, setImageFieldErrors] = useState(true);
  const [resourceErrors, setResourceErrors] = useState(false);
  const [argErrors, setArgErrors] = useState(false);
  const [dependencyErrors, setDependencyErrors] = useState(false);
  const [volumeErrors, setVolumeErrors] = useState(false);
  const [outputCollectionErrors, setOutputCollectionErrors] = useState(false);

  const formMode = state ? ImageFormMode.Copy : ImageFormMode.Create;
  const scaler = typeof editorObj.scaler === 'string' ? editorObj.scaler : 'K8s';

  // Lossless view switching — no confirmation needed since both views share editorObj
  const handleViewModeChange = (mode: ViewMode) => {
    if (mode !== viewMode) setViewMode(mode);
  };

  const handleEditorChange = (obj: Record<string, unknown> | null) => {
    if (obj) {
      setEditorObj(obj);
      setEditorParseValid(true);
    } else {
      setEditorParseValid(false);
    }
  };

  // Fields manages multiple top-level keys — remove old keys before merging new ones
  const handleFieldsChange = useCallback((fields: FieldsValue) => {
    setEditorObj((prev) => {
      const rest: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(prev)) {
        if (!FIELDS_KEYS.has(k)) rest[k] = v;
      }
      return { ...rest, ...(fields as unknown as Record<string, unknown>) };
    });
  }, []);

  useEffect(() => {
    void fetchGroups(setGroups as (groups: { [name: string]: Group } | Group[] | string[]) => void, null as never, false);
  }, []);

  useEffect(() => {
    const group = typeof editorObj.group === 'string' ? editorObj.group : '';
    if (group) void fetchImages([group], setImages, cancelUpdate, () => void checkCookie(), setLoading, false);
    return () => {
      cancelUpdate = true;
    };
  }, [editorObj.group]);

  useEffect(() => {
    if (!(imageFieldErrors || volumeErrors || dependencyErrors || outputCollectionErrors || resourceErrors)) {
      setCreateImageErrors('');
    }
  }, [imageFieldErrors, volumeErrors, dependencyErrors, outputCollectionErrors, resourceErrors]);

  const handleImageCreate = async () => {
    if (viewMode === ViewMode.Form) {
      if (imageFieldErrors || resourceErrors || argErrors || outputCollectionErrors || dependencyErrors || volumeErrors) {
        setCreateImageErrors('Please resolve missing fields or invalid entries');
        setDisplayErrors(true);
        return;
      }
      setCreateImageErrors('');
    }

    const data = editorObjectToImageCreate(editorObj);
    if (!data) {
      setCreateImageErrors('Image group and name are required');
      return;
    }

    if (await createImage(data as unknown as ImageRequest, setCreateImageErrors)) {
      void navigate('/images');
    } else {
      setDisplayErrors(true);
    }
  };

  return (
    <Page title="Create Image">
      <ImageCreateWrapper>
        <Row>
          <center>
            <h3>Create An Image</h3>
          </center>
        </Row>
        <Row className="mt-2 mb-3">
          <Col className="d-flex justify-content-center">
            <ViewModeToggle viewMode={viewMode} onViewModeChange={handleViewModeChange} />
          </Col>
        </Row>
        {viewMode === ViewMode.Editor ? (
          <>
            <Row className="mb-2">
              <Col>
                <FormatToggle format={editorFormat} onFormatChange={setEditorFormat} />
              </Col>
            </Row>
            <ImagePipelineEditor
              value={editorObj}
              onChange={handleEditorChange}
              checker={imageChecker}
              format={editorFormat}
              height="600px"
            />
          </>
        ) : (
          <>
            <ImageFieldsSection className="mt-4">
              <h5>Image</h5>
              <Fields
                value={editorObj as unknown as FieldsValue}
                groups={groups ?? []}
                onChange={handleFieldsChange}
                onValidate={setImageFieldErrors}
                showErrors={!!state || displayErrors}
                mode={formMode}
              />
            </ImageFieldsSection>
            <Row>
              <Col className="d-flex justify-content-center">
                <OverlayTipRight tip={`${hideAdvanced ? 'Expand' : 'Hide'} optional fields`}>
                  <div className="icon-btn" onClick={() => setHideAdvanced(!hideAdvanced)}>
                    {hideAdvanced ? <FaAngleDown size="36" /> : <FaAngleUp size="36" />}
                  </div>
                </OverlayTipRight>
              </Col>
            </Row>
            <hr className="mt-0" />
            <AdvancedHidden $hidden={hideAdvanced}>
              <Resources
                value={editorObj.resources ?? {}}
                onChange={(resources: ResourcesValue) => setEditorObj((prev) => ({ ...prev, resources }))}
                onValidate={setResourceErrors}
                mode={formMode}
              />
              <hr />
              <Arguments
                value={editorObj.args ?? {}}
                onChange={(args: ImageArgs) => setEditorObj((prev) => ({ ...prev, args }))}
                onValidate={setArgErrors}
                mode={formMode}
              />
              <hr />
              <OutputCollection
                value={(editorObj.output_collection as OutputCollectionType | undefined) ?? BlankOutputCollection}
                onChange={(oc: OutputCollectionType) => setEditorObj((prev) => ({ ...prev, output_collection: oc }))}
                groups={userInfo?.groups ?? []}
                mode={formMode}
                onValidate={setOutputCollectionErrors}
                disabled={scaler === 'External'}
              />
              <hr />
              <Dependencies
                images={images}
                value={editorObj.dependencies ?? {}}
                onChange={(deps: DependenciesType) => setEditorObj((prev) => ({ ...prev, dependencies: deps }))}
                onValidate={setDependencyErrors}
                mode={formMode}
                disabled={scaler === 'External'}
              />
              <hr />
              <EnvironmentVariables
                value={(editorObj.env ?? {}) as EnvValue}
                onChange={(env: EnvValue) => setEditorObj((prev) => ({ ...prev, env }))}
                mode={formMode}
              />
              <hr />
              <Volumes
                value={(editorObj.volumes ?? []) as Volume[]}
                onChange={(volumes: Volume[]) => setEditorObj((prev) => ({ ...prev, volumes }))}
                mode={formMode}
                onValidate={setVolumeErrors}
                disabled={scaler !== String(ImageScaler.K8s)}
              />
              <hr />
              <NetworkPolicies
                value={(editorObj.network_policies ?? []) as string[]}
                onChange={(np: string[]) => setEditorObj((prev) => ({ ...prev, network_policies: np }))}
                mode={formMode}
              />
              <hr />
              <SecurityContext
                value={editorObj.security_context ?? {}}
                onChange={(sc: SecurityContextType) => setEditorObj((prev) => ({ ...prev, security_context: sc }))}
                mode={formMode}
                disabled={scaler === 'External' || !userInfo || getThoriumRole(userInfo.role) !== RoleKey.Admin}
              />
              <hr />
              <ChildFilters
                value={editorObj.child_filters ?? {}}
                onChange={(cf: ChildFiltersType) => setEditorObj((prev) => ({ ...prev, child_filters: cf }))}
                mode={formMode}
                disabled={scaler === 'External'}
              />
              <hr />
              <Modifiers
                value={(editorObj.modifiers ?? '') as string}
                onChange={(m: string) => setEditorObj((prev) => ({ ...prev, modifiers: m || undefined }))}
                mode={formMode}
              />
              {scaler === String(ImageScaler.Kvm) && (
                <>
                  <hr />
                  <KVM
                    value={(editorObj.kvm ?? { xml: '', qcow2: '' }) as Kvm}
                    onChange={(kvm: Kvm) => setEditorObj((prev) => ({ ...prev, kvm }))}
                    mode={formMode}
                  />
                </>
              )}
            </AdvancedHidden>
          </>
        )}
        <Row className="d-flex justify-content-center">
          <Col>{createImageErrors && <AlertBanner className="m-2">{createImageErrors}</AlertBanner>}</Col>
        </Row>
        <Row>
          <LoadingSpinner loading={loading}></LoadingSpinner>
        </Row>
        <Row className="mt-3">
          <Col className="d-flex justify-content-center">
            <Button className="secondary-btn" onClick={() => void navigate(-1)}>
              Cancel
            </Button>
            <Button
              className="ok-btn"
              disabled={viewMode === ViewMode.Editor && !editorParseValid}
              onClick={() => void handleImageCreate()}
            >
              Create
            </Button>
          </Col>
        </Row>
      </ImageCreateWrapper>
    </Page>
  );
};

export default ImageCreate;
