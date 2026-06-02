// Authorization helpers that mirror the backend `Group`/`User` predicates in
// api/src/models/backends/{users,groups}.rs and the route macros in
// api/src/utils/macros.rs. Centralizing them keeps the UI's action gating in
// lockstep with the API so we never show a control that the backend rejects.

// project imports
import { getThoriumRole } from './role';
import { Group } from '@models/groups';
import { Image, ImageScaler } from '@models/images';
import { Pipeline } from '@models/pipelines';
import { RoleKey, UserInfo } from '@models/users';

// ---- Thorium-wide role predicates (mirror `User::` methods) ----

/**
 * Whether the user is a Thorium admin. Mirrors the backend `User::is_admin`.
 *
 * A non-developer role is the bare string at runtime (e.g. `"Admin"`), so the role is resolved
 * via {@link getThoriumRole} rather than checking `role.Admin`.
 *
 * @param userInfo - The user to check.
 * @returns `true` if the user's Thorium role is Admin.
 */
export function isAdmin(userInfo: UserInfo): boolean {
  return getThoriumRole(userInfo.role) == RoleKey.Admin;
}

/**
 * Whether the user can develop for a specific scaler. Mirrors `User::is_developer(scaler)`.
 *
 * Admins and analysts can develop for any scaler; users with the Developer role can only develop
 * for the scalers their per-scaler sub-permissions enable.
 *
 * @param userInfo - The user to check.
 * @param scaler - The {@link ImageScaler} the user wants to develop for.
 * @returns `true` if the user may develop for that scaler.
 */
export function isDeveloper(userInfo: UserInfo, scaler: ImageScaler): boolean {
  const role = getThoriumRole(userInfo.role);
  if (role == RoleKey.Admin || role == RoleKey.Analyst) {
    return true;
  }
  const dev = userInfo.role.Developer;
  if (role == RoleKey.Developer && dev) {
    switch (scaler) {
      case ImageScaler.K8s:
        return dev.k8s;
      case ImageScaler.BareMetal:
        return dev.bare_metal;
      case ImageScaler.Windows:
        return dev.windows;
      case ImageScaler.External:
        return dev.external;
      case ImageScaler.Kvm:
        return dev.kvm;
      default:
        return false;
    }
  }
  return false;
}

/**
 * Whether the user can develop for at least one scaler.
 *
 * Used to gate generic "create/develop" buttons where the target scaler hasn't been chosen yet.
 * Admins and analysts always qualify; a Developer qualifies if any per-scaler permission is set.
 *
 * @param userInfo - The user to check.
 * @returns `true` if the user can develop for any scaler.
 */
export function isDeveloperAny(userInfo: UserInfo): boolean {
  const role = getThoriumRole(userInfo.role);
  if (role == RoleKey.Admin || role == RoleKey.Analyst) {
    return true;
  }
  const dev = userInfo.role.Developer;
  if (role == RoleKey.Developer && dev) {
    return dev.k8s || dev.bare_metal || dev.windows || dev.external || dev.kvm;
  }
  return false;
}

// ---- Group membership helpers ----

/**
 * Whether a user holds a group role that the backend's develop checks accept.
 *
 * The backend `developer`/`developer_many` checks accept users/managers/owners. The group's
 * analyst list is intentionally excluded — those checks do not honor it, so a group-analyst-only
 * user cannot develop in the group.
 *
 * @param group - The group to check membership in.
 * @param username - The username to look for.
 * @returns `true` if the user is a user, manager, or owner of the group.
 */
function isDeveloperMember(group: Group, username: string): boolean {
  return group.users.combined.includes(username) || group.managers.combined.includes(username) || group.owners.combined.includes(username);
}

// ---- Group-level predicates (mirror `Group::` methods) ----

/**
 * Whether the user owns the group. Mirrors `Group::is_owner`.
 *
 * Governs group deletion and editing the owner list. Admins always qualify.
 *
 * @param group - The group to check.
 * @param userInfo - The user to check.
 * @returns `true` if the user is an admin or an owner of the group.
 */
export function isGroupOwner(group: Group, userInfo: UserInfo): boolean {
  return isAdmin(userInfo) || group.owners.combined.includes(userInfo.username);
}

/**
 * Whether the user can modify the group. Mirrors `Group::modifiable`.
 *
 * Admins, owners, and managers may modify/delete arbitrary group data and edit non-owner
 * membership.
 *
 * @param group - The group to check.
 * @param userInfo - The user to check.
 * @returns `true` if the user is an admin, owner, or manager of the group.
 */
export function canModifyGroup(group: Group, userInfo: UserInfo): boolean {
  return isAdmin(userInfo) || group.owners.combined.includes(userInfo.username) || group.managers.combined.includes(userInfo.username);
}

/**
 * Whether the user can edit data within the group. Mirrors `Group::editable`.
 *
 * Admins or any non-monitor member (users, analysts, managers, owners) qualify; monitors do not.
 *
 * @param group - The group to check.
 * @param userInfo - The user to check.
 * @returns `true` if the user is an admin or a non-monitor member of the group.
 */
export function canEditGroupData(group: Group, userInfo: UserInfo): boolean {
  const user = userInfo.username;
  return (
    isAdmin(userInfo) ||
    group.users.combined.includes(user) ||
    group.analysts.includes(user) ||
    group.managers.combined.includes(user) ||
    group.owners.combined.includes(user)
  );
}

/**
 * Whether the user can view the group. Mirrors `Group::viewable`.
 *
 * Admins or any member of any role (including analysts and monitors) qualify.
 *
 * @param group - The group to check.
 * @param userInfo - The user to check.
 * @returns `true` if the user is an admin or any member of the group.
 */
export function canViewGroup(group: Group, userInfo: UserInfo): boolean {
  const user = userInfo.username;
  return (
    isAdmin(userInfo) ||
    group.users.combined.includes(user) ||
    group.managers.combined.includes(user) ||
    group.analysts.includes(user) ||
    group.owners.combined.includes(user) ||
    group.monitors.combined.includes(user)
  );
}

/**
 * Whether the user can develop for a scaler within the group. Mirrors `Group::developer(scaler)`.
 *
 * True for admins, or when the user both {@link isDeveloper} for the scaler and is a develop-capable
 * member ({@link isDeveloperMember}) of the group.
 *
 * @param group - The group to check.
 * @param userInfo - The user to check.
 * @param scaler - The {@link ImageScaler} to develop for.
 * @returns `true` if the user may develop for that scaler within the group.
 */
export function canDevelopInGroup(group: Group, userInfo: UserInfo, scaler: ImageScaler): boolean {
  return isAdmin(userInfo) || (isDeveloper(userInfo, scaler) && isDeveloperMember(group, userInfo.username));
}

/**
 * Whether the user can develop in the group for at least one scaler.
 *
 * Used when the target scaler isn't known yet; mirrors `developer_many` at the coarse
 * "is a developer of some kind" level. True for admins, or when the user is {@link isDeveloperAny}
 * and a develop-capable member ({@link isDeveloperMember}) of the group.
 *
 * @param group - The group to check.
 * @param userInfo - The user to check.
 * @returns `true` if the user may develop in the group for any scaler.
 */
export function canDevelopAnyInGroup(group: Group, userInfo: UserInfo): boolean {
  return isAdmin(userInfo) || (isDeveloperAny(userInfo) && isDeveloperMember(group, userInfo.username));
}

// ---- Resource action gates (mirror route macros) ----

/**
 * Whether the user can modify (update) an image. Mirrors the `can_develop!` route macro.
 *
 * True for the image's creator, or a develop-capable member for the image's scaler
 * ({@link canDevelopInGroup}).
 *
 * @param image - The image being modified.
 * @param group - The group the image belongs to.
 * @param userInfo - The user to check.
 * @returns `true` if the user may update the image.
 */
export function canModifyImage(image: Image, group: Group, userInfo: UserInfo): boolean {
  return image.creator == userInfo.username || canDevelopInGroup(group, userInfo, image.scaler);
}

/**
 * Whether the user can delete an image. Mirrors the `can_delete!` route macro.
 *
 * True for the image's creator, or someone who can modify the group ({@link canModifyGroup}).
 *
 * @param image - The image being deleted.
 * @param group - The group the image belongs to.
 * @param userInfo - The user to check.
 * @returns `true` if the user may delete the image.
 */
export function canDeleteImage(image: Image, group: Group, userInfo: UserInfo): boolean {
  return image.creator == userInfo.username || canModifyGroup(group, userInfo);
}

/**
 * Whether the user can modify (update) a pipeline. Mirrors the `can_develop_many!` route macro
 * that pipeline updates always run.
 *
 * True for the pipeline's creator, or a develop-capable member of the group
 * ({@link canDevelopAnyInGroup}).
 *
 * @param pipeline - The pipeline being modified.
 * @param group - The group the pipeline belongs to.
 * @param userInfo - The user to check.
 * @returns `true` if the user may update the pipeline.
 */
export function canModifyPipeline(pipeline: Pipeline, group: Group, userInfo: UserInfo): boolean {
  return pipeline.creator == userInfo.username || canDevelopAnyInGroup(group, userInfo);
}

/**
 * Whether the user can delete a pipeline. Mirrors the `can_delete!` route macro.
 *
 * True for the pipeline's creator, or someone who can modify the group ({@link canModifyGroup}).
 *
 * @param pipeline - The pipeline being deleted.
 * @param group - The group the pipeline belongs to.
 * @param userInfo - The user to check.
 * @returns `true` if the user may delete the pipeline.
 */
export function canDeletePipeline(pipeline: Pipeline, group: Group, userInfo: UserInfo): boolean {
  return pipeline.creator == userInfo.username || canModifyGroup(group, userInfo);
}
