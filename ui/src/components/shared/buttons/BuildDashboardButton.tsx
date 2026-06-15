import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaChartColumn } from 'react-icons/fa6';

// project imports
import IconButton from './IconButton';
import { OverlayTipBottom } from '@components/shared/overlay/tips';
import { seedLabelParams } from '@dashboards/DashboardBuilder/builderReducer';
import { SelectionKind } from '@dashboards/DashboardBuilder/types';

// spec: ./Button.spec.md

/**
 * The kinds of resource a custom dashboard can be seeded from. Each maps to the
 * matching seed query-param key consumed by the dashboard builder / dashboard.
 */
export enum BuildDashboardResource {
  /** A file, seeded by its sha256 (`?sample=<sha256>`). */
  Sample = 'sample',
  /** A config-driven entity, seeded by its uuid (`?entity=<uuid>`). */
  Entity = 'entity',
  /** A repo, seeded by its url (`?repo=<url>`). */
  Repo = 'repo',
}

/**
 * Map a {@link BuildDashboardResource} to the builder's {@link SelectionKind} so the linked-in seed's
 * display label is built with the same rules the builder uses for its own chips.
 */
const RESOURCE_TO_SELECTION_KIND: Record<BuildDashboardResource, SelectionKind.File | SelectionKind.Entity | SelectionKind.Repo> = {
  [BuildDashboardResource.Sample]: SelectionKind.File,
  [BuildDashboardResource.Entity]: SelectionKind.Entity,
  [BuildDashboardResource.Repo]: SelectionKind.Repo,
};

export interface BuildDashboardButtonProps {
  /** The seed param key identifying which resource kind the dashboard is built from. */
  resource: BuildDashboardResource;
  /** The identifier value for the resource (sha256, entity uuid, or repo url). */
  id: string;
  /** Human-readable resource label used in the tooltip (e.g. "file", "device", "repo"). */
  label: string;
  /**
   * The resource's human-readable name (e.g. an entity name or filename), carried through to the
   * builder so the seeded chip reads e.g. `laptop-1 (uuid)` instead of the raw id. Optional; when
   * omitted the builder falls back to the id.
   */
  name?: string;
  /** Whether the button is disabled. */
  disabled?: boolean;
  /** Optional extra class names forwarded to the button. */
  className?: string;
}

/**
 * A shared entry-point button that navigates to the dashboard builder pre-seeded
 * with a single resource. Reused by the file, repo, and generic entity details
 * pages so the "Build Dashboard" affordance stays consistent across resources.
 *
 * @param resource - The seed param key identifying the resource kind.
 * @param id - The identifier value (sha256, entity uuid, or repo url).
 * @param label - Human-readable resource label for the tooltip.
 * @param name - The resource's human-readable name, carried through to the builder chip.
 * @param disabled - Whether the button is disabled.
 * @param className - Optional extra class names forwarded to the button.
 * @returns The rendered build-dashboard icon button wrapped in a tooltip.
 */
const BuildDashboardButton = ({ resource, id, label, name, disabled = false, className }: BuildDashboardButtonProps) => {
  const navigate = useNavigate();
  const handleClick = useCallback(() => {
    // encode the id so shas/uuids/repo urls survive the query string round-trip
    const params = new URLSearchParams({ [resource]: id });
    // carry the human-readable name so the builder seeds a readable chip (e.g. "laptop-1 (uuid)")
    // rather than a bare id
    if (name) {
      seedLabelParams(RESOURCE_TO_SELECTION_KIND[resource], id, name).forEach((value, key) => params.append(key, value));
    }
    void navigate(`/dashboard/build?${params.toString()}`);
  }, [navigate, resource, id, name]);
  return (
    <OverlayTipBottom tip={`Build a dashboard from this ${label}`}>
      <IconButton className={className} disabled={disabled} onClick={handleClick} aria-label={`Build a dashboard from this ${label}`}>
        <FaChartColumn size={20} />
      </IconButton>
    </OverlayTipBottom>
  );
};

export default BuildDashboardButton;
