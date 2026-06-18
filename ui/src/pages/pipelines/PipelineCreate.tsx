import { useCallback, useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

// project imports
import { Fields, OrderField, PipelineFormMode, Triggers } from '@components/pages/pipelines';
import type { PipelineFieldsValue } from '@components/pages/pipelines';
import { pipelineChecker } from '@components/pages/pipelines/checker';
import {
  ActionRow,
  AlertWrap,
  CenterRow,
  CreateTitle,
  FieldCol,
  FormatRow,
  FormSection,
  PipelineCreateWrapper,
  SectionRow,
  TitleCol,
  ToggleRow,
} from '@components/pages/pipelines/shared.styled';
import Page from '@components/pages/Page';
import { Button, ButtonVariant } from '@components/shared/buttons';
import AlertBanner from '@components/shared/alerts/AlertBanner';
import { FieldError } from '@components/shared/inputs/FieldError';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import FormatToggle from '@components/shared/inputs/code/CodeEditor/FormatToggle';
import ImagePipelineEditor from '@components/shared/inputs/code/CodeEditor/ImagePipelineEditor';
import ViewModeToggle, { ViewMode } from '@components/shared/inputs/code/CodeEditor/ViewModeToggle';
import { listImages } from '@thorpi/images';
import { createPipeline } from '@thorpi/pipelines';
import { useAuth } from '@utilities/auth';
import { fetchGroups } from '@utilities/fetch';
import { FormatType } from '@utilities/rules/types';
import { useDebouncedValue } from '@utilities/useDebouncedValue';
import { editorObjectToPipelineCreate, pipelineToEditorObject } from '@utilities/transforms/pipeline';
import type { EventTrigger, Pipeline, PipelineCreate as PipelineCreateType } from '@models/pipelines';
import type { Group } from '@models/groups';

/// The blank editor object used to seed a new pipeline
// `sla` is intentionally omitted so the field starts blank; the default (DEFAULT_SLA) is shown as
// placeholder text and applied on submit when left empty.
export const PIPELINE_CREATE_TEMPLATE: Record<string, unknown> = {
  group: '',
  name: '',
  order: [],
  description: '',
};

/// Top-level editor keys managed by the Fields form
const FIELDS_KEYS = new Set(['name', 'group', 'description', 'sla']);

const PipelineCreate: React.FC = () => {
  const navigate = useNavigate();
  const { state } = useLocation() as { state: Pipeline | null };
  const { checkCookie } = useAuth();
  const [groups, setGroups] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [displayErrors, setDisplayErrors] = useState(false);
  const [createError, setCreateError] = useState('');
  // Single source of truth: both form and editor views read/write this object
  const [editorObj, setEditorObj] = useState<Record<string, unknown>>(state ? pipelineToEditorObject(state) : PIPELINE_CREATE_TEMPLATE);
  const [viewMode, setViewMode] = useState<ViewMode>(ViewMode.Form);
  const [editorFormat, setEditorFormat] = useState<FormatType>(FormatType.YAML);
  const [editorParseValid, setEditorParseValid] = useState(false);
  const [fieldErrors, setFieldErrors] = useState(true);
  const [triggerErrors, setTriggerErrors] = useState(false);

  const formMode = state ? PipelineFormMode.Copy : PipelineFormMode.Create;
  const group = typeof editorObj.group === 'string' ? editorObj.group : '';
  const order = Array.isArray(editorObj.order) ? (editorObj.order as (string | string[])[]) : [];
  const triggers = (editorObj.triggers ?? {}) as Record<string, EventTrigger>;
  // an order with at least one image is required to create a pipeline
  const orderError = order.length === 0;

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
  const handleFieldsChange = useCallback((fields: PipelineFieldsValue) => {
    setEditorObj((prev) => {
      const rest: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(prev)) {
        if (!FIELDS_KEYS.has(k)) rest[k] = v;
      }
      // Changing the group invalidates the order: its images belong to the old group and may not
      // exist in the new one, so reset the order whenever the group changes.
      if (fields.group !== prev.group) rest.order = [];
      return { ...rest, ...(fields as unknown as Record<string, unknown>) };
    });
  }, []);

  useEffect(() => {
    void fetchGroups(setGroups as (groups: { [name: string]: Group } | Group[] | string[]) => void, () => void checkCookie(), false);
  }, []);

  // Clear the bottom error banner once the form's field/trigger/order validation passes, so a
  // resolved set of "missing field" errors doesn't leave a stale alert behind.
  useEffect(() => {
    if (viewMode === ViewMode.Form && !fieldErrors && !triggerErrors && !orderError) {
      setCreateError('');
    }
  }, [viewMode, fieldErrors, triggerErrors, orderError]);

  // Debounce so we don't refetch the image list on every keystroke while the group is typed.
  const debouncedGroup = useDebouncedValue(group, 400);
  useEffect(() => {
    if (!debouncedGroup) return;
    let cancelled = false;
    void listImages(debouncedGroup, () => {}, false, null, 1000).then((result) => {
      if (cancelled) return;
      if (result && 'names' in result) pipelineChecker.setValidImageNames(debouncedGroup, result.names);
    });
    return () => {
      cancelled = true;
    };
  }, [debouncedGroup]);

  const handlePipelineCreate = async () => {
    if (viewMode === ViewMode.Form && (fieldErrors || triggerErrors || orderError)) {
      setCreateError('Please resolve missing fields or invalid entries');
      setDisplayErrors(true);
      return;
    }
    const data = editorObjectToPipelineCreate(editorObj);
    if (!data) {
      setCreateError('Pipeline group, name, and order are required');
      setDisplayErrors(true);
      return;
    }
    setLoading(true);
    try {
      if (await createPipeline(data as PipelineCreateType, setCreateError)) {
        void navigate('/pipelines');
      } else {
        setDisplayErrors(true);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Page title="Create Pipeline">
      <PipelineCreateWrapper>
        <CreateTitle>Create A Pipeline</CreateTitle>
        <ToggleRow>
          <ViewModeToggle viewMode={viewMode} onViewModeChange={handleViewModeChange} />
        </ToggleRow>
        {viewMode === ViewMode.Editor ? (
          <>
            <FormatRow>
              <FormatToggle format={editorFormat} onFormatChange={setEditorFormat} />
            </FormatRow>
            <ImagePipelineEditor
              value={editorObj}
              onChange={handleEditorChange}
              checker={pipelineChecker}
              format={editorFormat}
              height="600px"
            />
          </>
        ) : (
          <>
            <FormSection>
              <TitleCol>
                <h5>Pipeline</h5>
              </TitleCol>
              <FieldCol>
                <Fields
                  value={editorObj as unknown as PipelineFieldsValue}
                  groups={groups}
                  onChange={handleFieldsChange}
                  onValidate={setFieldErrors}
                  showErrors={displayErrors}
                  mode={formMode}
                />
              </FieldCol>
            </FormSection>
            <hr />
            <SectionRow>
              <TitleCol>
                <h5>Order</h5>
              </TitleCol>
              <FieldCol>
                <OrderField order={order} onChange={(o) => setEditorObj((prev) => ({ ...prev, order: o }))} group={group} />
                {displayErrors && orderError && <FieldError>At least one image is required.</FieldError>}
              </FieldCol>
            </SectionRow>
            <hr />
            <SectionRow>
              <TitleCol>
                <h5>Triggers</h5>
              </TitleCol>
              <FieldCol>
                <Triggers
                  value={triggers}
                  onChange={(t) => setEditorObj((prev) => ({ ...prev, triggers: t }))}
                  onValidate={setTriggerErrors}
                  showErrors={displayErrors}
                  mode={formMode}
                />
              </FieldCol>
            </SectionRow>
          </>
        )}
        {createError && (
          <AlertWrap>
            <AlertBanner>{createError}</AlertBanner>
          </AlertWrap>
        )}
        <CenterRow>
          <LoadingSpinner loading={loading}></LoadingSpinner>
        </CenterRow>
        <ActionRow>
          <Button variant={ButtonVariant.Secondary} data-testid="create-cancel" onClick={() => void navigate(-1)}>
            Cancel
          </Button>
          <Button
            variant={ButtonVariant.Ok}
            data-testid="create-submit"
            disabled={(viewMode === ViewMode.Editor && !editorParseValid) || loading}
            onClick={() => void handlePipelineCreate()}
          >
            Create
          </Button>
        </ActionRow>
      </PipelineCreateWrapper>
    </Page>
  );
};

export default PipelineCreate;
