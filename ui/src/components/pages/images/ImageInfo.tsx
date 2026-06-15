import { useEffect, useImperativeHandle, useState } from 'react';
import type { FC, Ref } from 'react';

// project imports
import { imageChecker, FIELDS_KEYS, formatBanKind } from './utilities';
import type { SectionItem, EnvValue } from './utilities';
import {
  ColumnCard,
  FieldsRow,
  CenteredContent,
  FormWrapper,
  ErrorRow,
  CenterRow,
  ImageBansContainer,
  SECTION_COL_WIDTH_PX,
  SECTION_GAP,
} from './ImageInfo.styled';
import {
  Fields,
  NetworkPolicies,
  Resources,
  Arguments,
  OutputCollection,
  Dependencies,
  EnvironmentVariables,
  Volumes,
  SecurityContext,
  ChildFilters,
  KVM,
  Modifiers,
  ImageFormMode,
} from '@components/pages/images';
import type { FieldsValue, ResourcesValue } from '@components/pages/images';
import { BanItem } from '@components/shared/browsing';
import { BalancedColumns } from '@components/shared/layout/BalancedColumns';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import ImagePipelineEditor from '@components/shared/inputs/code/CodeEditor/ImagePipelineEditor';
import FormatToggle from '@components/shared/inputs/code/CodeEditor/FormatToggle';
import ViewModeToggle, { ViewMode } from '@components/shared/inputs/code/CodeEditor/ViewModeToggle';
import { BUTTON_BAR_MARGIN } from '@components/shared/buttons/tokens';
import { updateImage } from '@thorpi/images';
import { useAuth } from '@utilities/auth';
import { fetchSingleImage } from '@utilities/fetch';
import { getThoriumRole } from '@utilities/role';
import { FormatType } from '@utilities/rules/types';
import { editorObjectToImageUpdate, imageToEditorObject } from '@utilities/transforms/image';
import type { Group } from '@models/groups';
import type {
  Image,
  ImageArgs,
  ImageBan,
  Dependencies as DependenciesType,
  SecurityContext as SecurityContextType,
  ChildFilters as ChildFiltersType,
  Kvm,
} from '@models/images';
import { BlankOutputCollection, ImageScaler } from '@models/images';
import type { Volume } from '@models/volumes';
import type { OutputCollection as OutputCollectionType } from '@models/results';
import { RoleKey } from '@models/users';

// spec: ./ImageInfo.spec.md

export interface ImageInfoHandle {
  handleUpdate: () => void;
}

interface ImageInfoProps {
  ref: Ref<ImageInfoHandle>;
  images: Image[];
  image: Image;
  groups: Record<string, Group>;
  setImages: (images: Image[]) => void;
  inEditMode: boolean;
  onExitEditMode: () => void;
  userCanModify: boolean;
}

const ImageInfo: FC<ImageInfoProps> = ({ ref, images, image, setImages, inEditMode, onExitEditMode }) => {
  const [updateError, setUpdateError] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentImage, setCurrentImage] = useState<Image>(image);
  const { userInfo } = useAuth();

  const [editorObj, setEditorObj] = useState<Record<string, unknown>>(() => imageToEditorObject(currentImage));
  // Bumped whenever editorObj is re-seeded from freshly-loaded image data so the section
  // form components re-derive their internal state instead of keeping a stale snapshot.
  const [formResetKey, setFormResetKey] = useState(0);
  const [viewMode, setViewMode] = useState<ViewMode>(ViewMode.Form);
  const [editorFormat, setEditorFormat] = useState<FormatType>(FormatType.YAML);
  const [editorParseValid, setEditorParseValid] = useState(false);

  const [fieldsErrors, setFieldsErrors] = useState(false);
  const [resourceErrors, setResourceErrors] = useState(false);
  const [argErrors, setArgErrors] = useState(false);
  const [volumeErrors, setVolumeErrors] = useState(false);
  const [dependencyErrors, setDependencyErrors] = useState(false);
  const [outputCollectionErrors, setOutputCollectionErrors] = useState(false);

  const mode = inEditMode ? ImageFormMode.Edit : ImageFormMode.View;
  const scaler = typeof editorObj.scaler === 'string' ? editorObj.scaler : currentImage.scaler;

  useEffect(() => {
    setEditorObj(imageToEditorObject(currentImage));
    setViewMode(ViewMode.Form);
    setUpdateError('');
    setFormResetKey((k) => k + 1);
  }, [inEditMode]);

  const handleViewModeChange = (viewModeTarget: ViewMode) => {
    if (viewModeTarget !== viewMode) setViewMode(viewModeTarget);
  };

  const handleEditorChange = (obj: Record<string, unknown> | null) => {
    if (obj) {
      setEditorObj(obj);
      setEditorParseValid(true);
    } else {
      setEditorParseValid(false);
    }
  };

  const handleFieldsChange = (fields: FieldsValue) => {
    setEditorObj((prev) => {
      const rest: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(prev)) {
        if (!FIELDS_KEYS.has(k)) rest[k] = v;
      }
      return { ...rest, ...(fields as unknown as Record<string, unknown>) };
    });
  };

  useEffect(() => {
    if (!(fieldsErrors || resourceErrors || argErrors || volumeErrors || dependencyErrors || outputCollectionErrors)) {
      setUpdateError('');
    }
  }, [fieldsErrors, resourceErrors, argErrors, volumeErrors, dependencyErrors, outputCollectionErrors]);

  const handleUpdate = async () => {
    if (viewMode === ViewMode.Editor && !editorParseValid) {
      setUpdateError('Editor contains invalid syntax');
      return;
    }
    if (viewMode === ViewMode.Form) {
      if (fieldsErrors || resourceErrors || argErrors || outputCollectionErrors || dependencyErrors || volumeErrors) {
        setUpdateError('Please resolve missing fields or invalid entries');
        return;
      }
      setUpdateError('');
    }

    const result = editorObjectToImageUpdate(editorObj, currentImage);
    if (!result) {
      setUpdateError('Invalid image data');
      return;
    }

    if (await updateImage(result.group, result.name, result.data, setUpdateError)) {
      // Refetch only the saved image and re-seed both the view (currentImage) and the form
      // (editorObj) from the fresh data before leaving edit mode, so re-opening the form
      // shows the saved values instead of the pre-save snapshot. Also replace just this
      // image in the parent list so the accordion header/count/filters reflect the edit
      // without reloading the whole list or collapsing the open accordion.
      await fetchSingleImage(
        image,
        (fresh) => {
          setCurrentImage(fresh);
          setEditorObj(imageToEditorObject(fresh));
          setImages(images.map((img) => (img.group === image.group && img.name === image.name ? fresh : img)));
        },
        setLoading,
      );
      onExitEditMode();
      setUpdateError('');
    }
  };

  useImperativeHandle(ref, () => ({ handleUpdate: () => void handleUpdate() }));

  const groupImages = images.filter((someImage) => currentImage.group == someImage.group);
  const imageNames = [...new Set(groupImages.map((img) => img.name))];

  return (
    <div>
      {updateError && (
        <ErrorRow>
          <AlertBanner>{updateError}</AlertBanner>
        </ErrorRow>
      )}
      {inEditMode && (
        <CenterRow>
          <ViewModeToggle viewMode={viewMode} onViewModeChange={handleViewModeChange} />
        </CenterRow>
      )}
      {inEditMode && viewMode === ViewMode.Editor ? (
        <>
          <div style={{ marginBottom: BUTTON_BAR_MARGIN }}>
            <FormatToggle format={editorFormat} onFormatChange={setEditorFormat} />
          </div>
          <ImagePipelineEditor
            value={editorObj}
            onChange={handleEditorChange}
            checker={imageChecker}
            format={editorFormat}
            height="600px"
          />
        </>
      ) : inEditMode && viewMode === ViewMode.Form ? (
        <FormWrapper>
          <Fields
            value={editorObj as unknown as FieldsValue}
            resetKey={formResetKey}
            groups={[]}
            onChange={handleFieldsChange}
            showErrors={true}
            onValidate={setFieldsErrors}
            mode={mode}
            creator={currentImage.creator}
            runtime={currentImage.runtime}
            usedBy={currentImage.used_by}
          />
          <hr className="my-3" />
          <Resources
            value={editorObj.resources ?? {}}
            resetKey={formResetKey}
            onChange={(resources: ResourcesValue) => setEditorObj((prev) => ({ ...prev, resources }))}
            onValidate={setResourceErrors}
            mode={mode}
          />
          <hr className="my-3" />
          <Arguments
            value={editorObj.args ?? {}}
            resetKey={formResetKey}
            onChange={(args: ImageArgs) => setEditorObj((prev) => ({ ...prev, args }))}
            onValidate={setArgErrors}
            mode={mode}
          />
          <hr className="my-3" />
          <OutputCollection
            value={(editorObj.output_collection as OutputCollectionType | undefined) ?? BlankOutputCollection}
            resetKey={formResetKey}
            onChange={(oc: OutputCollectionType) => setEditorObj((prev) => ({ ...prev, output_collection: oc }))}
            groups={userInfo?.groups ?? []}
            mode={mode}
            onValidate={setOutputCollectionErrors}
            disabled={scaler === String(ImageScaler.External)}
          />
          <hr className="my-3" />
          <Dependencies
            value={editorObj.dependencies ?? {}}
            resetKey={formResetKey}
            images={imageNames}
            mode={mode}
            onChange={(deps: DependenciesType) => setEditorObj((prev) => ({ ...prev, dependencies: deps }))}
            onValidate={setDependencyErrors}
            disabled={scaler === String(ImageScaler.External)}
          />
          {scaler === String(ImageScaler.K8s) && (
            <>
              <hr className="my-3" />
              <EnvironmentVariables
                value={(editorObj.env ?? {}) as EnvValue}
                resetKey={formResetKey}
                onChange={(env: EnvValue) => setEditorObj((prev) => ({ ...prev, env }))}
                mode={mode}
              />
            </>
          )}
          {scaler === String(ImageScaler.K8s) && (
            <>
              <hr className="my-3" />
              <Volumes
                value={(editorObj.volumes ?? []) as Volume[]}
                resetKey={formResetKey}
                onChange={(volumes: Volume[]) => setEditorObj((prev) => ({ ...prev, volumes }))}
                mode={mode}
                onValidate={setVolumeErrors}
              />
              <hr className="my-3" />
              <NetworkPolicies
                value={(editorObj.network_policies ?? []) as string[]}
                onChange={(np: string[]) => setEditorObj((prev) => ({ ...prev, network_policies: np }))}
                mode={mode}
              />
            </>
          )}
          <hr className="my-3" />
          <ChildFilters
            value={editorObj.child_filters ?? {}}
            resetKey={formResetKey}
            onChange={(cf: ChildFiltersType) => setEditorObj((prev) => ({ ...prev, child_filters: cf }))}
            mode={mode}
            disabled={scaler === String(ImageScaler.External)}
          />
          <hr className="my-3" />
          <Modifiers
            value={(editorObj.modifiers ?? '') as string}
            onChange={(m: string) => setEditorObj((prev) => ({ ...prev, modifiers: m || undefined }))}
            mode={mode}
          />
          {scaler === String(ImageScaler.Kvm) && (
            <>
              <hr className="my-3" />
              <KVM
                value={(editorObj.kvm ?? { xml: '', qcow2: '' }) as Kvm}
                onChange={(kvm: Kvm) => setEditorObj((prev) => ({ ...prev, kvm }))}
                mode={mode}
              />
            </>
          )}
          {scaler === String(ImageScaler.K8s) && (
            <>
              <hr className="my-3" />
              <SecurityContext
                value={editorObj.security_context ?? {}}
                onChange={(sc: SecurityContextType) => setEditorObj((prev) => ({ ...prev, security_context: sc }))}
                mode={userInfo && getThoriumRole(userInfo.role) == RoleKey.Admin ? ImageFormMode.Edit : ImageFormMode.View}
                disabled={!userInfo || getThoriumRole(userInfo.role) !== RoleKey.Admin}
              />
            </>
          )}
        </FormWrapper>
      ) : (
        <>
          {currentImage.bans && Object.keys(currentImage.bans).length > 0 && (
            <ImageBansContainer style={{ marginBottom: 8 }}>
              {Object.values(currentImage.bans).map((ban: ImageBan) => (
                <BanItem key={ban.id} severity={Severity.Warning}>
                  {formatBanKind(ban.ban_kind)}
                  {'. Banned on '}
                  {new Date(ban.time_banned).toLocaleString()}
                </BanItem>
              ))}
            </ImageBansContainer>
          )}
          <FieldsRow>
            <CenteredContent>
              <Fields
                value={currentImage}
                groups={[]}
                onChange={handleFieldsChange}
                showErrors={true}
                onValidate={setFieldsErrors}
                mode={ImageFormMode.View}
                creator={currentImage.creator}
                runtime={currentImage.runtime}
                usedBy={currentImage.used_by}
              />
            </CenteredContent>
          </FieldsRow>
          {(() => {
            const deps = currentImage.dependencies ?? {};
            const oc = currentImage.output_collection ?? {};
            const cf = currentImage.child_filters ?? {};
            const env = currentImage.env ?? {};
            const vols = currentImage.volumes ?? [];
            const netPols = currentImage.network_policies ?? [];

            const sections: SectionItem[] = [];
            sections.push({
              key: 'resources',
              content: (
                <Resources
                  value={currentImage.resources ?? {}}
                  onChange={(resources: ResourcesValue) => setEditorObj((prev) => ({ ...prev, resources }))}
                  onValidate={setResourceErrors}
                  mode={ImageFormMode.View}
                />
              ),
            });
            sections.push({
              key: 'arguments',
              content: (
                <Arguments
                  value={currentImage.args ?? {}}
                  onChange={(args: ImageArgs) => setEditorObj((prev) => ({ ...prev, args }))}
                  onValidate={setArgErrors}
                  mode={ImageFormMode.View}
                />
              ),
            });
            if (scaler !== String(ImageScaler.External)) {
              sections.push({
                key: 'output-collection',
                content: (
                  <OutputCollection
                    value={oc}
                    onChange={(o: OutputCollectionType) => setEditorObj((prev) => ({ ...prev, output_collection: o }))}
                    groups={userInfo?.groups ?? []}
                    mode={ImageFormMode.View}
                    onValidate={setOutputCollectionErrors}
                  />
                ),
              });
              sections.push({
                key: 'dependencies',
                content: (
                  <Dependencies
                    value={deps}
                    images={imageNames}
                    mode={ImageFormMode.View}
                    onChange={(d: DependenciesType) => setEditorObj((prev) => ({ ...prev, dependencies: d }))}
                    onValidate={setDependencyErrors}
                  />
                ),
              });
              sections.push({
                key: 'child-filters',
                content: (
                  <ChildFilters
                    value={cf}
                    onChange={(c: ChildFiltersType) => setEditorObj((prev) => ({ ...prev, child_filters: c }))}
                    mode={ImageFormMode.View}
                  />
                ),
              });
            }
            if (scaler === String(ImageScaler.K8s)) {
              sections.push({
                key: 'environment',
                content: (
                  <EnvironmentVariables
                    value={env}
                    onChange={(e: EnvValue) => setEditorObj((prev) => ({ ...prev, env: e }))}
                    mode={ImageFormMode.View}
                  />
                ),
              });
            }
            sections.push({
              key: 'modifiers',
              content: (
                <Modifiers
                  value={currentImage.modifiers ?? ''}
                  onChange={(m: string) => setEditorObj((prev) => ({ ...prev, modifiers: m || undefined }))}
                  mode={ImageFormMode.View}
                />
              ),
            });
            if (scaler === String(ImageScaler.K8s)) {
              sections.push({
                key: 'volumes',
                content: (
                  <Volumes
                    value={vols}
                    onChange={(v: Volume[]) => setEditorObj((prev) => ({ ...prev, volumes: v }))}
                    mode={ImageFormMode.View}
                    onValidate={setVolumeErrors}
                  />
                ),
              });
              sections.push({
                key: 'network-policies',
                content: (
                  <NetworkPolicies
                    value={netPols}
                    onChange={(np: string[]) => setEditorObj((prev) => ({ ...prev, network_policies: np }))}
                    mode={ImageFormMode.View}
                  />
                ),
              });
            }
            if (scaler === String(ImageScaler.Kvm)) {
              sections.push({
                key: 'kvm',
                content: (
                  <KVM
                    value={currentImage.kvm ?? { xml: '', qcow2: '' }}
                    onChange={(kvm: Kvm) => setEditorObj((prev) => ({ ...prev, kvm }))}
                    mode={ImageFormMode.View}
                  />
                ),
              });
            }
            if (scaler === String(ImageScaler.K8s)) {
              sections.push({
                key: 'security-context',
                content: (
                  <SecurityContext
                    value={currentImage.security_context ?? {}}
                    onChange={(sc: SecurityContextType) => setEditorObj((prev) => ({ ...prev, security_context: sc }))}
                    mode={ImageFormMode.View}
                    disabled={true}
                  />
                ),
              });
            }

            // measured balancing: BalancedColumns derives the column count from the container
            // width (two on normal screens, one when narrow) and flows each card into the
            // currently-shortest column using real DOM heights; capped at two columns
            return (
              <BalancedColumns
                columnWidth={SECTION_COL_WIDTH_PX}
                maxColumns={2}
                gap={SECTION_GAP}
                items={sections.map((s) => (
                  <ColumnCard key={s.key}>{s.content}</ColumnCard>
                ))}
              />
            );
          })()}
        </>
      )}
      <LoadingSpinner loading={loading} />
    </div>
  );
};

export default ImageInfo;
