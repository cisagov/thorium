import { describe, it, expect } from 'vitest';

// project imports
import {
  isAdmin,
  isDeveloper,
  isDeveloperAny,
  isGroupOwner,
  canModifyGroup,
  canEditGroupData,
  canViewGroup,
  canDevelopInGroup,
  canDevelopAnyInGroup,
  canModifyImage,
  canDeleteImage,
  canModifyPipeline,
  canDeletePipeline,
} from './permissions';
import { Group, GroupUsers } from '@models/groups';
import { Image, ImageScaler } from '@models/images';
import { Pipeline } from '@models/pipelines';
import { ThoriumRole, UserInfo } from '@models/users';

// build a GroupUsers role bucket from a list of combined members
function roleBucket(combined: string[] = []): GroupUsers {
  return { combined, direct: combined, metagroups: [] };
}

// build a minimal Group with the given per-role membership
function makeGroup(overrides: Partial<Group> = {}): Group {
  return {
    name: 'g',
    owners: roleBucket(),
    managers: roleBucket(),
    analysts: [],
    users: roleBucket(),
    monitors: roleBucket(),
    allowed: {
      files: true,
      repos: true,
      tags: true,
      images: true,
      pipelines: true,
      reactions: true,
      results: true,
      comments: true,
      entities: true,
    },
    ...overrides,
  };
}

// build a minimal UserInfo with the given username and Thorium role
function makeUser(username: string, role: ThoriumRole): UserInfo {
  return {
    username,
    role,
    email: `${username}@example.com`,
    groups: [],
    token: 't',
    token_expiration: '2099-01-01T00:00:00Z',
    settings: { theme: 'Dark' },
    local: true,
    verified: true,
  };
}

// runtime role values as the API serializes them (plain strings except Developer)
const ADMIN = 'Admin' as unknown as ThoriumRole;
const ANALYST = 'Analyst' as unknown as ThoriumRole;
const USER = 'User' as unknown as ThoriumRole;
// a developer who can only develop for the K8s scaler
const DEV_K8S = { Developer: { k8s: true, bare_metal: false, windows: false, external: false, kvm: false } } as unknown as ThoriumRole;
// a developer with no scalers enabled
const DEV_NONE = { Developer: { k8s: false, bare_metal: false, windows: false, external: false, kvm: false } } as unknown as ThoriumRole;

function makeImage(creator: string, scaler: ImageScaler, group = 'g'): Image {
  return { group, name: 'img', creator, scaler } as unknown as Image;
}

function makePipeline(creator: string, group = 'g'): Pipeline {
  return { group, name: 'pipe', creator } as unknown as Pipeline;
}

describe('isAdmin', () => {
  it('detects the admin role even though it is a bare string at runtime', () => {
    expect(isAdmin(makeUser('a', ADMIN))).toBe(true);
    expect(isAdmin(makeUser('b', ANALYST))).toBe(false);
    expect(isAdmin(makeUser('c', USER))).toBe(false);
    expect(isAdmin(makeUser('d', DEV_K8S))).toBe(false);
  });
});

describe('isDeveloper', () => {
  it('treats admins and analysts as developers for every scaler', () => {
    for (const scaler of Object.values(ImageScaler)) {
      expect(isDeveloper(makeUser('a', ADMIN), scaler)).toBe(true);
      expect(isDeveloper(makeUser('b', ANALYST), scaler)).toBe(true);
    }
  });

  it('honors per-scaler developer sub-permissions', () => {
    const dev = makeUser('d', DEV_K8S);
    expect(isDeveloper(dev, ImageScaler.K8s)).toBe(true);
    expect(isDeveloper(dev, ImageScaler.BareMetal)).toBe(false);
    expect(isDeveloper(dev, ImageScaler.Windows)).toBe(false);
  });

  it('is false for plain users', () => {
    expect(isDeveloper(makeUser('u', USER), ImageScaler.K8s)).toBe(false);
  });
});

describe('isDeveloperAny', () => {
  it('is true for admins, analysts, and developers with any scaler enabled', () => {
    expect(isDeveloperAny(makeUser('a', ADMIN))).toBe(true);
    expect(isDeveloperAny(makeUser('b', ANALYST))).toBe(true);
    expect(isDeveloperAny(makeUser('d', DEV_K8S))).toBe(true);
  });

  it('is false for developers with no scalers and for plain users', () => {
    expect(isDeveloperAny(makeUser('d', DEV_NONE))).toBe(false);
    expect(isDeveloperAny(makeUser('u', USER))).toBe(false);
  });
});

describe('isGroupOwner (mirrors Group::is_owner)', () => {
  it('is true for admins and group owners only', () => {
    expect(isGroupOwner(makeGroup({ owners: roleBucket(['o']) }), makeUser('o', USER))).toBe(true);
    expect(isGroupOwner(makeGroup(), makeUser('admin', ADMIN))).toBe(true);
    expect(isGroupOwner(makeGroup({ managers: roleBucket(['m']) }), makeUser('m', USER))).toBe(false);
    expect(isGroupOwner(makeGroup({ users: roleBucket(['u']) }), makeUser('u', USER))).toBe(false);
  });
});

describe('canModifyGroup (mirrors Group::modifiable)', () => {
  it('allows admins, owners, and managers', () => {
    expect(canModifyGroup(makeGroup({ owners: roleBucket(['o']) }), makeUser('o', USER))).toBe(true);
    expect(canModifyGroup(makeGroup({ managers: roleBucket(['m']) }), makeUser('m', USER))).toBe(true);
    expect(canModifyGroup(makeGroup(), makeUser('admin', ADMIN))).toBe(true);
  });

  it('denies users, monitors, and global analysts', () => {
    expect(canModifyGroup(makeGroup({ users: roleBucket(['u']) }), makeUser('u', USER))).toBe(false);
    expect(canModifyGroup(makeGroup({ monitors: roleBucket(['mo']) }), makeUser('mo', USER))).toBe(false);
    expect(canModifyGroup(makeGroup({ analysts: ['an'] }), makeUser('an', ANALYST))).toBe(false);
  });
});

describe('canEditGroupData (mirrors Group::editable)', () => {
  it('allows admins and any non-monitor member, including analysts', () => {
    expect(canEditGroupData(makeGroup({ users: roleBucket(['u']) }), makeUser('u', USER))).toBe(true);
    expect(canEditGroupData(makeGroup({ analysts: ['an'] }), makeUser('an', ANALYST))).toBe(true);
    expect(canEditGroupData(makeGroup({ managers: roleBucket(['m']) }), makeUser('m', USER))).toBe(true);
    expect(canEditGroupData(makeGroup({ owners: roleBucket(['o']) }), makeUser('o', USER))).toBe(true);
    expect(canEditGroupData(makeGroup(), makeUser('admin', ADMIN))).toBe(true);
  });

  it('denies monitors and non-members', () => {
    expect(canEditGroupData(makeGroup({ monitors: roleBucket(['mo']) }), makeUser('mo', USER))).toBe(false);
    expect(canEditGroupData(makeGroup(), makeUser('nobody', USER))).toBe(false);
  });
});

describe('canViewGroup (mirrors Group::viewable)', () => {
  it('allows admins and every member type, including monitors and analysts', () => {
    expect(canViewGroup(makeGroup({ monitors: roleBucket(['mo']) }), makeUser('mo', USER))).toBe(true);
    expect(canViewGroup(makeGroup({ analysts: ['an'] }), makeUser('an', ANALYST))).toBe(true);
    expect(canViewGroup(makeGroup(), makeUser('admin', ADMIN))).toBe(true);
  });

  it('denies non-members', () => {
    expect(canViewGroup(makeGroup(), makeUser('nobody', USER))).toBe(false);
  });
});

describe('canDevelopInGroup (mirrors Group::developer)', () => {
  it('admins can develop for any scaler regardless of membership', () => {
    expect(canDevelopInGroup(makeGroup(), makeUser('admin', ADMIN), ImageScaler.K8s)).toBe(true);
  });

  it('an analyst who is ALSO a direct group member can develop (the bug we fixed)', () => {
    const group = makeGroup({ analysts: ['an'], users: roleBucket(['an']) });
    expect(canDevelopInGroup(group, makeUser('an', ANALYST), ImageScaler.BareMetal)).toBe(true);
  });

  it('a GLOBAL-analyst-only user (not a real group member) cannot develop', () => {
    const group = makeGroup({ analysts: ['an'] });
    expect(canDevelopInGroup(group, makeUser('an', ANALYST), ImageScaler.K8s)).toBe(false);
  });

  it('a developer who is a member can only develop for enabled scalers', () => {
    const group = makeGroup({ users: roleBucket(['d']) });
    expect(canDevelopInGroup(group, makeUser('d', DEV_K8S), ImageScaler.K8s)).toBe(true);
    expect(canDevelopInGroup(group, makeUser('d', DEV_K8S), ImageScaler.Windows)).toBe(false);
  });

  it('a developer who is not a group member cannot develop', () => {
    expect(canDevelopInGroup(makeGroup(), makeUser('d', DEV_K8S), ImageScaler.K8s)).toBe(false);
  });

  it('a plain user member cannot develop', () => {
    const group = makeGroup({ users: roleBucket(['u']) });
    expect(canDevelopInGroup(group, makeUser('u', USER), ImageScaler.K8s)).toBe(false);
  });
});

describe('canDevelopAnyInGroup (generic develop capability)', () => {
  it('admins always, develop-capable members yes, others no', () => {
    expect(canDevelopAnyInGroup(makeGroup(), makeUser('admin', ADMIN))).toBe(true);
    expect(canDevelopAnyInGroup(makeGroup({ users: roleBucket(['d']) }), makeUser('d', DEV_K8S))).toBe(true);
    expect(canDevelopAnyInGroup(makeGroup({ analysts: ['an'], users: roleBucket(['an']) }), makeUser('an', ANALYST))).toBe(true);
    // a global-analyst-only user is not a develop-capable member
    expect(canDevelopAnyInGroup(makeGroup({ analysts: ['an'] }), makeUser('an', ANALYST))).toBe(false);
    // a developer who is not a member, and a plain user member, cannot develop
    expect(canDevelopAnyInGroup(makeGroup(), makeUser('d', DEV_K8S))).toBe(false);
    expect(canDevelopAnyInGroup(makeGroup({ users: roleBucket(['u']) }), makeUser('u', USER))).toBe(false);
  });
});

describe('canModifyImage (mirrors can_develop!)', () => {
  it('allows the creator regardless of role', () => {
    const group = makeGroup({ users: roleBucket(['creator']) });
    expect(canModifyImage(makeImage('creator', ImageScaler.K8s), group, makeUser('creator', USER))).toBe(true);
  });

  it('allows an analyst member, denies a global-analyst-only', () => {
    const member = makeGroup({ analysts: ['an'], users: roleBucket(['an']) });
    expect(canModifyImage(makeImage('someone', ImageScaler.K8s), member, makeUser('an', ANALYST))).toBe(true);
    const global = makeGroup({ analysts: ['an'] });
    expect(canModifyImage(makeImage('someone', ImageScaler.K8s), global, makeUser('an', ANALYST))).toBe(false);
  });

  it('allows admins', () => {
    expect(canModifyImage(makeImage('someone', ImageScaler.K8s), makeGroup(), makeUser('admin', ADMIN))).toBe(true);
  });
});

describe('canDeleteImage (mirrors can_delete!)', () => {
  it('allows the creator, owners, managers, and admins', () => {
    expect(canDeleteImage(makeImage('creator', ImageScaler.K8s), makeGroup(), makeUser('creator', USER))).toBe(true);
    expect(canDeleteImage(makeImage('x', ImageScaler.K8s), makeGroup({ owners: roleBucket(['o']) }), makeUser('o', USER))).toBe(true);
    expect(canDeleteImage(makeImage('x', ImageScaler.K8s), makeGroup({ managers: roleBucket(['m']) }), makeUser('m', USER))).toBe(true);
    expect(canDeleteImage(makeImage('x', ImageScaler.K8s), makeGroup(), makeUser('admin', ADMIN))).toBe(true);
  });

  it('denies a global analyst and a non-creator plain user', () => {
    expect(canDeleteImage(makeImage('x', ImageScaler.K8s), makeGroup({ analysts: ['an'] }), makeUser('an', ANALYST))).toBe(false);
    expect(canDeleteImage(makeImage('x', ImageScaler.K8s), makeGroup({ users: roleBucket(['u']) }), makeUser('u', USER))).toBe(false);
  });
});

describe('canModifyPipeline (mirrors can_develop_many!)', () => {
  it('allows the creator and develop-capable members', () => {
    expect(canModifyPipeline(makePipeline('creator'), makeGroup(), makeUser('creator', USER))).toBe(true);
    const devGroup = makeGroup({ users: roleBucket(['d']) });
    expect(canModifyPipeline(makePipeline('x'), devGroup, makeUser('d', DEV_K8S))).toBe(true);
    const anGroup = makeGroup({ analysts: ['an'], managers: roleBucket(['an']) });
    expect(canModifyPipeline(makePipeline('x'), anGroup, makeUser('an', ANALYST))).toBe(true);
    expect(canModifyPipeline(makePipeline('x'), makeGroup(), makeUser('admin', ADMIN))).toBe(true);
  });

  it('denies a global-analyst-only and a plain user member', () => {
    expect(canModifyPipeline(makePipeline('x'), makeGroup({ analysts: ['an'] }), makeUser('an', ANALYST))).toBe(false);
    expect(canModifyPipeline(makePipeline('x'), makeGroup({ users: roleBucket(['u']) }), makeUser('u', USER))).toBe(false);
  });
});

describe('canDeletePipeline (mirrors can_delete!)', () => {
  it('allows the creator, owners/managers, and admins', () => {
    expect(canDeletePipeline(makePipeline('creator'), makeGroup(), makeUser('creator', USER))).toBe(true);
    expect(canDeletePipeline(makePipeline('x'), makeGroup({ managers: roleBucket(['m']) }), makeUser('m', USER))).toBe(true);
    expect(canDeletePipeline(makePipeline('x'), makeGroup(), makeUser('admin', ADMIN))).toBe(true);
  });

  it('denies a global analyst and a non-creator plain user', () => {
    expect(canDeletePipeline(makePipeline('x'), makeGroup({ analysts: ['an'] }), makeUser('an', ANALYST))).toBe(false);
    expect(canDeletePipeline(makePipeline('x'), makeGroup({ users: roleBucket(['u']) }), makeUser('u', USER))).toBe(false);
  });
});
