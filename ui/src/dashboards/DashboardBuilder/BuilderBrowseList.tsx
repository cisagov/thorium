// spec: ./SPEC.md

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { FaPlus, FaXmark } from 'react-icons/fa6';

// project imports
import { descriptorFor } from './builderDescriptors';
import { selectionKey } from './builderReducer';
import ResourceTypePicker from './ResourceTypePicker';
import { BrowseMode, BuilderSelection, SelectionKind, TAG_MODE } from './types';
import { AddButton, BrowseControls, BrowseRow, BuilderSection, EmptyBrowse, Pager, PagerButton, SectionLabel, TagModeRow } from './styles';
import { DEFAULT_LIST_LIMIT } from '@components/entities/utilities';
import { EntityBrowsingConfig } from '@components/entities/browsing/configs/config';
import BrowsingFilters from '@components/entities/browsing/filters/BrowsingFilters';
import EntityList from '@components/entities/browsing/EntityList';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import { OverlayTipLeft } from '@components/shared/overlay/tips';
import Tabs from '@components/shared/tabs/Tabs';
import { TabItem } from '@components/shared/tabs/types';
import TagSelect from '@components/shared/inputs/tags/TagSelect/TagSelect';
import { listFiles } from '@thorpi/files';
import { Entities } from '@models/entities/entities';
import { Filters } from '@models/search';
import { Sample } from '@models/files';
import { TagEntry } from '@models/tags';

/**
 * Props for {@link BuilderBrowseList}.
 */
interface BuilderBrowseListProps {
  /// The current browse mode (an entity kind or Tag mode).
  mode: BrowseMode;
  /// Called with the new mode when the user picks a different resource type from the header dropdown.
  onModeChange: (mode: BrowseMode) => void;
  /// The set of selection identity keys already added (drives the added row's remove control).
  selectedKeys: Set<string>;
  /// The current username, used for the File "Your Submissions" tab filter.
  username: string | null;
  /// Called with a built selection when the user adds a row (or a tag pair).
  onAdd: (selection: BuilderSelection) => void;
  /// Called with an already-added selection when the user removes it from the row.
  onRemove: (selection: BuilderSelection) => void;
}

/// The tag key files carry identifying who submitted them; used for the "Your Submissions" filter.
const SUBMITTER_TAG = 'submitter';

/**
 * The per-row add/remove control: a plus-icon button when the selection isn't added, or an X button
 * that removes it from the selected resources when it is.
 *
 * @param props.selection - The selection this row adds/removes.
 * @param props.added - Whether the selection is already added.
 * @param props.onAdd - Called with the selection to add it.
 * @param props.onRemove - Called with the selection to remove it.
 * @returns The add/remove control wrapped in an overlay tip.
 */
const RowAddControl: React.FC<{
  selection: BuilderSelection;
  added: boolean;
  onAdd: (s: BuilderSelection) => void;
  onRemove: (s: BuilderSelection) => void;
}> = ({ selection, added, onAdd, onRemove }) => {
  if (added) {
    return (
      <OverlayTipLeft tip={`Remove ${selection.label} from dashboard.`}>
        <AddButton $square $remove onClick={() => onRemove(selection)} aria-label={`Remove ${selection.label} from dashboard`}>
          <FaXmark size={11} />
        </AddButton>
      </OverlayTipLeft>
    );
  }
  return (
    <OverlayTipLeft tip={`Add ${selection.label} to dashboard.`}>
      <AddButton $square onClick={() => onAdd(selection)} aria-label={`Add ${selection.label} to dashboard`}>
        <FaPlus size={11} />
      </AddButton>
    </OverlayTipLeft>
  );
};

/// The submissions tabs: the current user's uploads vs everyone's.
enum SubmissionsTab {
  /// Files submitted by the current user (via the `submitter` tag).
  Yours = 'your',
  /// All recent submissions, regardless of submitter.
  All = 'all',
}

/**
 * The file rows for one submissions tab: a file row + add control each, or an empty placeholder.
 *
 * @param props.files - The files to render.
 * @param props.render - The config's `renderEntity` for a file row.
 * @param props.selectedKeys - Already-added selection keys.
 * @param props.onAdd - Add callback.
 * @param props.onRemove - Remove callback for an already-added file.
 * @param props.emptyLabel - Placeholder text when there are no files.
 * @returns The rendered rows (or an empty placeholder).
 */
const FileRows: React.FC<{
  files: Sample[];
  render: (file: Sample, idx: number, filters?: Filters) => React.ReactNode;
  selectedKeys: Set<string>;
  onAdd: (s: BuilderSelection) => void;
  onRemove: (s: BuilderSelection) => void;
  emptyLabel: string;
  filters?: Filters;
}> = ({ files, render, selectedKeys, onAdd, onRemove, emptyLabel, filters }) => {
  if (files.length === 0) {
    return <EmptyBrowse>{emptyLabel}</EmptyBrowse>;
  }
  return (
    <>
      {files.map((file, idx) => {
        const selection = descriptorFor(SelectionKind.File, file);
        return (
          <BrowseRow key={file.sha256}>
            {/* pass the omnibar filters so the file row's renderEntity honors the hideTags exclusion */}
            <div>{render(file, idx, filters)}</div>
            <RowAddControl selection={selection} added={selectedKeys.has(selectionKey(selection))} onAdd={onAdd} onRemove={onRemove} />
          </BrowseRow>
        );
      })}
    </>
  );
};

/**
 * The File submissions browse list, presented as two tabs.
 *
 * Issues TWO `listFiles` requests — one filtered by the `submitter` tag = current username ("Your
 * Submissions"), one unfiltered ("All Submissions") — and renders the active tab's files. After each load
 * the active tab defaults to "Your Submissions" when the user has any, otherwise "All Submissions"; the
 * "Your Submissions" tab is disabled when there is no username. Partial-failure fallback: if one request
 * fails the other's list is still available with an inline error banner.
 *
 * @param props.filters - The omnibar-derived base filters (merged with the submitter filter).
 * @param props.username - The current username (drives the submitter filter).
 * @param props.selectedKeys - Already-added selection keys.
 * @param props.onAdd - Add callback.
 * @param props.onRemove - Remove callback for an already-added file.
 * @returns The tabbed file browse list.
 */
const FileUploadsFirst: React.FC<{
  filters: Filters;
  username: string | null;
  selectedKeys: Set<string>;
  onAdd: (s: BuilderSelection) => void;
  onRemove: (s: BuilderSelection) => void;
}> = ({ filters, username, selectedKeys, onAdd, onRemove }) => {
  // each tab keeps its own accumulated pages + next-page cursor + current page index, so Back/Next can walk
  // beyond the first page (the previous version fetched one page and dropped the cursor, capping at the limit)
  const [yours, setYours] = useState<Sample[]>([]);
  const [all, setAll] = useState<Sample[]>([]);
  const [yoursCursor, setYoursCursor] = useState<string | null>(null);
  const [allCursor, setAllCursor] = useState<string | null>(null);
  const [yoursPage, setYoursPage] = useState(0);
  const [allPage, setAllPage] = useState(0);
  const [active, setActive] = useState<SubmissionsTab>(SubmissionsTab.All);
  const [loading, setLoading] = useState(false);
  const [paging, setPaging] = useState(false);
  const [error, setError] = useState('');
  const render = EntityBrowsingConfig[Entities.File].renderEntity;
  const limit = filters.limit ? filters.limit : DEFAULT_LIST_LIMIT;

  // the current user's submissions filter (submitter tag, case-insensitive so a username's casing doesn't
  // hide results); memoized so it's stable for both the initial load effect and the Next-page fetch
  const yoursFilters = useMemo<Filters>(
    () =>
      username ? { ...filters, tags: { ...(filters.tags ?? {}), [SUBMITTER_TAG]: [username] }, tags_case_insensitive: true } : filters,
    [filters, username],
  );

  useEffect(() => {
    // skip until the omnibar has pushed real filters (mirrors EntityList's mount guard)
    if (filters == null || Object.keys(filters).length === 0) {
      return;
    }
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError('');
      // listFiles follows the thorpi convention of never throwing — it reports failures via its
      // errorHandler and resolves an empty list — so a real failure is only distinguishable from an empty
      // result through the handler. Capture each request's error via its own handler rather than promise status
      let yoursError = '';
      let allError = '';
      // request 1: the current user's first page; request 2: everyone's first page (unfiltered by submitter)
      const [yoursRes, allRes] = await Promise.all([
        username ? listFiles(yoursFilters, (message) => (yoursError = message), true, null) : Promise.resolve({ files: [], cursor: null }),
        listFiles(filters, (message) => (allError = message), true, null),
      ]);
      if (cancelled) {
        return;
      }
      const yoursFiles = yoursRes?.files ?? [];
      const allFiles = allRes?.files ?? [];
      // reset each tab to its first page + fresh next-page cursor
      setYours(yoursFiles);
      setAll(allFiles);
      setYoursCursor(yoursRes?.cursor ?? null);
      setAllCursor(allRes?.cursor ?? null);
      setYoursPage(0);
      setAllPage(0);
      // default to "All Submissions" when the user has none of their own, otherwise their own
      setActive(yoursFiles.length > 0 ? SubmissionsTab.Yours : SubmissionsTab.All);
      // partial-failure fallback: surface an error only when a request actually reported one
      if (yoursError && allError) {
        setError('Failed to load submissions.');
      } else if (yoursError) {
        setError('Could not load your submissions; showing all recent submissions.');
      } else if (allError) {
        setError('Could not load all recent submissions; showing your submissions.');
      }
      setLoading(false);
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [filters, yoursFilters, username]);

  // the active tab's accumulated files, page index, cursor, and filter — so the pager logic is tab-agnostic
  const isYours = active === SubmissionsTab.Yours;
  const activeFiles = isYours ? yours : all;
  const activePage = isYours ? yoursPage : allPage;
  const activeCursor = isYours ? yoursCursor : allCursor;
  const setActivePage = isYours ? setYoursPage : setAllPage;
  const pageFiles = activeFiles.slice(activePage * limit, activePage * limit + limit);
  const loadedPages = Math.ceil(activeFiles.length / limit);
  const hasBack = activePage > 0;
  // a next page exists when more is already loaded OR the server has more (a non-null cursor)
  const hasNext = activePage + 1 < loadedPages || activeCursor != null;

  const goBack = () => setActivePage((p) => Math.max(0, p - 1));
  const goNext = async () => {
    // advance within already-loaded pages without a fetch
    if (activePage + 1 < loadedPages) {
      setActivePage((p) => p + 1);
      return;
    }
    // at the end of the loaded set: fetch the next page from the server, append it, then advance
    if (activeCursor == null) {
      return;
    }
    setPaging(true);
    const res = await listFiles(isYours ? yoursFilters : filters, () => setError('Could not load more submissions.'), true, activeCursor);
    const newFiles = res?.files ?? [];
    if (isYours) {
      setYours((prev) => [...prev, ...newFiles]);
      setYoursCursor(res?.cursor ?? null);
    } else {
      setAll((prev) => [...prev, ...newFiles]);
      setAllCursor(res?.cursor ?? null);
    }
    setPaging(false);
    // only advance if the fetch actually returned rows (a stale cursor returning nothing keeps the page)
    if (newFiles.length > 0) {
      setActivePage((p) => p + 1);
    }
  };

  // count = the loaded-so-far total; a non-null cursor means the server has more, shown as `N+`. Both grow as
  // the user pages Next (more files accumulate), and the `+` drops once a tab's cursor is exhausted.
  const tabs: TabItem<SubmissionsTab>[] = [
    { key: SubmissionsTab.Yours, label: 'Your Submissions', count: yours.length, countMore: yoursCursor != null, disabled: !username },
    { key: SubmissionsTab.All, label: 'All Submissions', count: all.length, countMore: allCursor != null },
  ];

  return (
    <>
      {error !== '' && <AlertBanner severity={Severity.Warning}>{error}</AlertBanner>}
      <LoadingSpinner loading={loading} />
      {!loading && (
        <>
          <Tabs aria-label="Submissions" tabs={tabs} active={active} onChange={setActive} />
          <FileRows
            files={pageFiles}
            render={render}
            selectedKeys={selectedKeys}
            onAdd={onAdd}
            onRemove={onRemove}
            emptyLabel={isYours ? 'You have no submissions.' : 'No submissions found.'}
            filters={filters}
          />
          {(hasBack || hasNext) && (
            <Pager>
              <PagerButton onClick={goBack} disabled={!hasBack || paging}>
                Back
              </PagerButton>
              <PagerButton onClick={() => void goNext()} disabled={!hasNext || paging}>
                Next
              </PagerButton>
            </Pager>
          )}
        </>
      )}
    </>
  );
};

/**
 * The Tag-mode entry: a `TagSelect` key/value field whose Add pushes a `{kind: Tag, key, value}`.
 *
 * @param props.onAdd - Add callback.
 * @returns The tag entry row.
 */
const TagModeEntry: React.FC<{ onAdd: (s: BuilderSelection) => void }> = ({ onAdd }) => {
  const [tags, setTags] = useState<TagEntry[]>([]);
  // only complete (non-empty key AND value) tags are addable
  const entry = tags.find((t) => t.key.trim() !== '' && t.value.trim() !== '');
  const handleAdd = () => {
    if (!entry) {
      return;
    }
    onAdd(descriptorFor(SelectionKind.Tag, { key: entry.key, value: entry.value }));
    setTags([]);
  };
  return (
    <TagModeRow>
      <TagSelect tags={tags} setTags={setTags} placeholderText="Enter a tag key and value…" />
      <OverlayTipLeft tip="Add tag to dashboard.">
        <AddButton $square onClick={handleAdd} disabled={!entry} aria-label="Add tag to dashboard">
          <FaPlus size={11} />
        </AddButton>
      </OverlayTipLeft>
    </TagModeRow>
  );
};

/**
 * The builder's browse area for the selected resource type.
 *
 * For an entity kind it drives the shared config-driven {@link EntityList} with the kind's
 * `fetchEntities`/`entityHeaders`, wrapping each rendered row in a plus-icon add control (already-added
 * rows show an X that removes the selection). For {@link Entities.File} it swaps in the tabbed submissions list
 * (Your / All). For Tag mode it swaps the list for a `TagSelect` key/value entry. The omnibar mirrors the
 * standard browse pages via {@link BrowsingFilters}.
 *
 * @param props - See {@link BuilderBrowseListProps}.
 * @returns The browse section.
 */
const BuilderBrowseList: React.FC<BuilderBrowseListProps> = ({ mode, onModeChange, selectedKeys, username, onAdd, onRemove }) => {
  const [filters, setFilters] = useState<Filters>({});
  const [loading, setLoading] = useState(false);
  // receive the omnibar-derived filters from BrowsingFilters (URL-backed) and hand them to the list
  const handleFilters = useCallback((next: Filters) => setFilters(next), []);

  // wrap the kind's renderEntity so each row gets an add control mapped to a File/Repo/Entity selection
  const displayEntity = useCallback(
    (entity: unknown, idx: number, rowFilters?: Filters) => {
      const kind = mode as Entities;
      const config = EntityBrowsingConfig[kind];
      // map the browse kind to the selection kind (File/Repo browse to their own selection kinds;
      // every other entity kind is an Entity selection)
      const selectionKind =
        kind === Entities.File ? SelectionKind.File : kind === Entities.Repo ? SelectionKind.Repo : SelectionKind.Entity;
      const selection = descriptorFor(selectionKind, entity as never);
      return (
        <BrowseRow>
          <div>{config.renderEntity(entity as never, idx, rowFilters)}</div>
          <RowAddControl selection={selection} added={selectedKeys.has(selectionKey(selection))} onAdd={onAdd} onRemove={onRemove} />
        </BrowseRow>
      );
    },
    [mode, selectedKeys, onAdd, onRemove],
  );

  // the resource-type dropdown sits inline, directly in front of the omnibar (or the Tag-mode entry) so
  // it clearly controls what the filters/list below show; the filters wrapper flexes to fill the row
  if (mode === TAG_MODE) {
    return (
      <BuilderSection>
        <SectionLabel>Add dashboard Resources</SectionLabel>
        <BrowseControls>
          <ResourceTypePicker mode={mode} onChange={onModeChange} />
          <TagModeEntry onAdd={onAdd} />
        </BrowseControls>
      </BuilderSection>
    );
  }

  // after the TAG_MODE guard above, mode is narrowed to an entity kind
  const kind = mode;
  const config = EntityBrowsingConfig[kind];

  return (
    <BuilderSection>
      <SectionLabel>Add dashboard Resources</SectionLabel>
      <BrowseControls>
        <ResourceTypePicker mode={mode} onChange={onModeChange} />
        {/* BrowsingFilters is a fragment (two Rows); wrap it in one fill div so the row has exactly two
            flex children [picker][filters] and the filters area fills the remaining width */}
        <div>
          <BrowsingFilters onChange={handleFilters} kind={kind} />
        </div>
      </BrowseControls>
      {kind === Entities.File ? (
        <FileUploadsFirst filters={filters} username={username} selectedKeys={selectedKeys} onAdd={onAdd} onRemove={onRemove} />
      ) : (
        <EntityList
          // key on the kind so switching browse type mounts a fresh list rather than showing the
          // previous kind's rows until new filters arrive
          key={kind}
          type={kind}
          entityHeaders={config.entityHeaders}
          displayEntity={displayEntity}
          fetchEntities={config.fetchEntities}
          filters={filters}
          loading={loading}
          setLoading={setLoading}
        />
      )}
    </BuilderSection>
  );
};

export default BuilderBrowseList;
