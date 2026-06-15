// spec: ./SPEC.md

import React, { useCallback, useEffect, useMemo, useReducer, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';

// project imports
import {
  builderReducer,
  DEFAULT_BUILDER_STATE,
  LABEL_KEY_PARAM,
  LABEL_VALUE_PARAM,
  selectionKey,
  selectionsToLabelParams,
} from './builderReducer';
import { selectionsToSeedParams } from './builderDescriptors';
import { BuilderActionKind, BuilderSelection, BrowseMode } from './types';
import { Entities } from '@models/entities/entities';
import BuilderBrowseList from './BuilderBrowseList';
import DepthControl from './DepthControl';
import SelectionPanel from './SelectionPanel';
import { BuilderLayout, BuilderMenu, CreateButton, SectionHr, SummaryTile } from './styles';
import { decodeSeedParams, SEED_PARAM_KEYS } from '../Dashboard/seedParams';
import Page from '@components/pages/Page';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import { OverlayTipTop } from '@components/shared/overlay/tips';
import { useAuth } from '@utilities/auth';

/// The localStorage key recording that the user has dismissed the first-visit intro banner.
const INTRO_DISMISSED_KEY = 'thorium.dashboardBuilder.introDismissed';

/**
 * The Dashboard Builder route page.
 *
 * Assembles a set of seed resources (files, repos, entities, tags) plus a crawl depth, then
 * navigates to `/dashboard/view?<encoded params>` on Create. Selection state is a pure `useReducer`
 * ({@link builderReducer}); it is mirrored to the URL so the builder survives refresh/back and is
 * itself shareable, and it hydrates from the same seed params the dashboard reads (deep links and
 * details-page "Build Dashboard" entry points land here pre-populated). Composes the resource-type
 * picker, the config-driven browse area, the selection panel, the depth control, and the Create
 * button.
 *
 * @returns The builder page.
 */
const DashboardBuilder: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { userInfo } = useAuth();
  const [state, dispatch] = useReducer(builderReducer, DEFAULT_BUILDER_STATE);
  const [mode, setMode] = useState<BrowseMode>(Entities.File);
  const [introDismissed, setIntroDismissed] = useState<boolean>(() => {
    // only show the first-visit banner until the user dismisses it once
    try {
      return localStorage.getItem(INTRO_DISMISSED_KEY) === 'true';
    } catch {
      return false;
    }
  });

  // hydrate selections + depth from the URL once on mount so deep links / entry points populate the
  // builder; the URL is the source of truth for the initial state
  const initialDepth = useMemo(() => decodeSeedParams(searchParams).depth, []);
  const [depth, setDepth] = useState<number>(initialDepth);
  useEffect(() => {
    dispatch({ type: BuilderActionKind.HydrateFromParams, params: searchParams });
  }, []);

  // mirror selections + depth back into the URL whenever they change, so refresh/back restores the
  // builder and its URL is shareable (replace so rapid edits don't flood history)
  useEffect(() => {
    const next = selectionsToSeedParams(state.selected, depth);
    // preserve any non-seed params (e.g. omnibar clauses) already on the URL; the seed keys plus the
    // display-label pairs are rewritten wholesale from current state so stale entries never linger
    const merged = new URLSearchParams(searchParams);
    for (const key of [...SEED_PARAM_KEYS, LABEL_KEY_PARAM, LABEL_VALUE_PARAM]) {
      merged.delete(key);
    }
    next.forEach((value, key) => merged.append(key, value));
    // carry each chip's human-readable label so refresh/share keeps chips readable, not raw ids
    selectionsToLabelParams(state.selected).forEach((value, key) => merged.append(key, value));
    setSearchParams(merged, { replace: true });
    // selected + depth are the only inputs to the encoded URL; searchParams/setSearchParams are
    // intentionally omitted so this fires on selection/depth changes, not on every URL edit
  }, [state.selected, depth]);

  const selectedKeys = useMemo(() => new Set(state.selected.map(selectionKey)), [state.selected]);
  const handleAdd = useCallback((selection: BuilderSelection) => dispatch({ type: BuilderActionKind.Add, selection }), []);
  const handleRemove = useCallback(
    (key: string) => {
      const target = state.selected.find((s) => selectionKey(s) === key);
      if (target) {
        dispatch({ type: BuilderActionKind.Remove, selection: target });
      }
    },
    [state.selected],
  );
  const handleReadd = useCallback(
    (key: string) => {
      const target = state.removed.find((s) => selectionKey(s) === key);
      if (target) {
        dispatch({ type: BuilderActionKind.Readd, selection: target });
      }
    },
    [state.removed],
  );

  const dismissIntro = () => {
    setIntroDismissed(true);
    try {
      localStorage.setItem(INTRO_DISMISSED_KEY, 'true');
    } catch {
      // localStorage may be unavailable (private mode); dismissal just won't persist
    }
  };

  const empty = state.selected.length === 0;
  const handleCreate = () => {
    if (empty) {
      return;
    }
    void navigate(`/dashboard/view?${selectionsToSeedParams(state.selected, depth).toString()}`);
  };

  return (
    <Page title="Dashboard Builder" className="full-min-width">
      <BuilderLayout>
        {!introDismissed && (
          <AlertBanner severity={Severity.Info} dismissible onDismiss={dismissIntro}>
            Build a custom dashboard: <b>browse</b> a resource type, <b>add</b> items (or a tag), then <b>Create</b> to open the dashboard.
            Your selections live in this page&apos;s URL, so it&apos;s shareable and survives a refresh.
          </AlertBanner>
        )}
        {/* the selected resources, pinned above the builder's action menu */}
        <SummaryTile>
          <SelectionPanel selected={state.selected} removed={state.removed} onRemove={handleRemove} onReadd={handleReadd} />
        </SummaryTile>
        {/* borderless centered menu: depth then create */}
        <BuilderMenu>
          <DepthControl depth={depth} onChange={setDepth} />
          <OverlayTipTop tip={empty ? 'Add at least one resource first' : 'Open the dashboard for your selected resources'}>
            {/* focusable span so the tip still fires while the button is disabled */}
            <span tabIndex={empty ? 0 : -1}>
              <CreateButton onClick={handleCreate} disabled={empty} aria-label="Create dashboard">
                Create
              </CreateButton>
            </span>
          </OverlayTipTop>
        </BuilderMenu>
        {/* divider standing in for the (now removed) add-resources card edge, between the controls and browse */}
        <SectionHr />
        <BuilderBrowseList
          mode={mode}
          onModeChange={setMode}
          selectedKeys={selectedKeys}
          username={userInfo?.username ?? null}
          onAdd={handleAdd}
          onRemove={(selection) => handleRemove(selectionKey(selection))}
        />
      </BuilderLayout>
    </Page>
  );
};

export default DashboardBuilder;
