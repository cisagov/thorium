import { useAuth } from '@utilities/auth';
import Omnibar from './Omnibar';
import { useTagCount } from '@utilities/tags';
import { Clause, ClauseCondition } from './ClauseTypes';
import {
  addGroupOptions,
  addHideTagOptions,
  addIndexOptions,
  addLimitOptions,
  addStringOption,
  addTagOptions,
  addTextOptions,
  OmnibarOptionMap,
} from './options';
import { TimeSelection } from './timepicker/utils';
import styled from 'styled-components';
import { Group, GroupUserCategory } from '@models/groups';
import { getAllGroupUsers } from '@utilities/groups';
import { Pipeline } from '@models/pipelines';
import { Image } from '@models/images';
import { RoleKey, UserInfo } from '@models/users';
import SingleTimePicker from './timepicker/SingleTimePicker';

// spec: ./SPEC.md

type OmnibarTimeContainerProps = {
  clauses: Clause[];
  setClauses: (next: Clause[]) => void;
  time: TimeSelection;
  setTime: (next: TimeSelection) => void;
};

const OmnibarWrapperContainer = styled.div`
  max-width: 1000px;
  width: 100%;
  display: flex;
  justify-content: center;
  position: relative;
  margin-bottom: 20px;
`;

export const OmnibarMainSearch: React.FC<OmnibarTimeContainerProps> = ({ clauses, setClauses, time, setTime }) => {
  const { userInfo } = useAuth();
  const userInfoGroups = userInfo !== null ? userInfo.groups : [];
  const tagOptions = useTagCount();

  let dropdownOptions: OmnibarOptionMap = {};
  dropdownOptions = addGroupOptions(dropdownOptions, userInfoGroups);
  dropdownOptions = addIndexOptions(dropdownOptions);
  dropdownOptions = addTagOptions(dropdownOptions, tagOptions);
  dropdownOptions = addTextOptions(dropdownOptions);
  dropdownOptions = addLimitOptions(dropdownOptions);

  return (
    <OmnibarWrapperContainer>
      <Omnibar
        clauses={clauses}
        dropdownOptions={dropdownOptions}
        setClauses={setClauses}
        placeholder="Search..."
        timepickerVisible={true}
      />
      <SingleTimePicker time={time} setTime={setTime} />
    </OmnibarWrapperContainer>
  );
};

export const OmnibarStandardTimeFilters: React.FC<OmnibarTimeContainerProps> = ({ clauses, setClauses, time, setTime }) => {
  const { userInfo } = useAuth();
  const userInfoGroups = userInfo !== null ? userInfo.groups : [];
  const tagOptions = useTagCount();

  let dropdownOptions: OmnibarOptionMap = {};
  dropdownOptions = addGroupOptions(dropdownOptions, userInfoGroups);
  dropdownOptions = addTagOptions(dropdownOptions, tagOptions);
  dropdownOptions = addHideTagOptions(dropdownOptions, tagOptions);
  dropdownOptions = addLimitOptions(dropdownOptions);

  return (
    <OmnibarWrapperContainer>
      <Omnibar
        clauses={clauses}
        dropdownOptions={dropdownOptions}
        setClauses={setClauses}
        placeholder="Enter a filter..."
        timepickerVisible={true}
      />
      <SingleTimePicker time={time} setTime={setTime} />
    </OmnibarWrapperContainer>
  );
};

interface OmnibarGroupProps {
  clauses: Clause[];
  setClauses: (next: Clause[]) => void;
  groups: Record<string, Group>;
}

function getGroupMemberType(groups: Record<string, Group>, category: GroupUserCategory): string[] {
  const allStrings = Object.values(groups).map((g) => getAllGroupUsers(g[category]));
  return Array.from(new Set(allStrings.flat())).sort();
}

export function OmnibarGroups({ clauses, setClauses, groups }: OmnibarGroupProps) {
  const { userInfo } = useAuth();
  const userInfoGroups = userInfo !== null ? userInfo.groups : [];

  let dropdownOptions: OmnibarOptionMap = {};
  dropdownOptions = addGroupOptions(dropdownOptions, userInfoGroups);
  dropdownOptions = addStringOption(dropdownOptions, 'Owners', getGroupMemberType(groups, 'owners'), [
    ClauseCondition.Is,
    // ClauseCondition.IsNot,
    ClauseCondition.IsOneOf,
  ]);
  dropdownOptions = addStringOption(dropdownOptions, 'Users', getGroupMemberType(groups, 'users'), [
    ClauseCondition.Is,
    // ClauseCondition.IsNot,
    ClauseCondition.IsOneOf,
  ]);
  dropdownOptions = addStringOption(dropdownOptions, 'Managers', getGroupMemberType(groups, 'managers'), [
    ClauseCondition.Is,
    // ClauseCondition.IsNot,
    ClauseCondition.IsOneOf,
  ]);

  return (
    <OmnibarWrapperContainer>
      <Omnibar clauses={clauses} dropdownOptions={dropdownOptions} setClauses={setClauses} placeholder="Enter a filter..." />
    </OmnibarWrapperContainer>
  );
}

interface OmnibarPipelineProps {
  clauses: Clause[];
  setClauses: (next: Clause[]) => void;
  pipelines: Pipeline[];
}

function uniqueSort(list: string[]): string[] {
  return Array.from(new Set(list)).sort();
}

export function OmnibarPipelines({ clauses, setClauses, pipelines }: OmnibarPipelineProps) {
  const { userInfo } = useAuth();
  const userInfoGroups = userInfo !== null ? userInfo.groups : [];

  let dropdownOptions: OmnibarOptionMap = {};
  dropdownOptions = addGroupOptions(dropdownOptions, userInfoGroups);
  dropdownOptions = addStringOption(
    dropdownOptions,
    'name',
    pipelines.map((p) => p.name),
    [ClauseCondition.Includes, ClauseCondition.Is],
  );
  dropdownOptions = addStringOption(dropdownOptions, 'creator', uniqueSort(pipelines.map((pipeline) => pipeline.creator)), [
    ClauseCondition.Includes,
    ClauseCondition.Is,
  ]);

  return (
    <OmnibarWrapperContainer>
      <Omnibar clauses={clauses} dropdownOptions={dropdownOptions} setClauses={setClauses} placeholder="Enter a filter..." />
    </OmnibarWrapperContainer>
  );
}

interface OmnibarUserProps {
  clauses: Clause[];
  setClauses: (next: Clause[]) => void;
  users: UserInfo[];
}

export function OmnibarUsers({ clauses, setClauses, users }: OmnibarUserProps) {
  let dropdownOptions: OmnibarOptionMap = {};
  // suggest every group present across the listed users (admins browse users in all groups)
  dropdownOptions = addGroupOptions(dropdownOptions, uniqueSort(users.flatMap((user) => user.groups)));
  dropdownOptions = addStringOption(dropdownOptions, 'username', uniqueSort(users.map((user) => user.username)), [
    ClauseCondition.Includes,
    ClauseCondition.Is,
  ]);
  dropdownOptions = addStringOption(
    dropdownOptions,
    'email',
    uniqueSort(users.map((user) => user.email)),
    [ClauseCondition.Includes, ClauseCondition.Is],
    {
      creatable: true,
    },
  );
  // role values are the fixed set of Thorium roles (Admin/Analyst/Developer/User)
  dropdownOptions = addStringOption(dropdownOptions, 'role', Object.values(RoleKey), [ClauseCondition.Is, ClauseCondition.IsOneOf], {
    category: 'role',
  });
  // boolean fields: pick true or false
  dropdownOptions = addStringOption(dropdownOptions, 'verified', ['true', 'false'], [ClauseCondition.Is]);
  dropdownOptions = addStringOption(dropdownOptions, 'local', ['true', 'false'], [ClauseCondition.Is]);

  return (
    <OmnibarWrapperContainer>
      <Omnibar clauses={clauses} dropdownOptions={dropdownOptions} setClauses={setClauses} placeholder="Enter a filter..." />
    </OmnibarWrapperContainer>
  );
}

interface OmnibarImageProps {
  clauses: Clause[];
  setClauses: (next: Clause[]) => void;
  images: Image[];
}

export function OmnibarImages({ clauses, setClauses, images }: OmnibarImageProps) {
  const { userInfo } = useAuth();
  const userInfoGroups = userInfo !== null ? userInfo.groups : [];

  let dropdownOptions: OmnibarOptionMap = {};
  dropdownOptions = addGroupOptions(dropdownOptions, userInfoGroups);
  dropdownOptions = addTextOptions(dropdownOptions);
  dropdownOptions = addStringOption(dropdownOptions, 'creator', uniqueSort(images.map((image) => image.creator)), [
    ClauseCondition.Includes,
    ClauseCondition.Is,
  ]);
  dropdownOptions = addStringOption(dropdownOptions, 'name', uniqueSort(images.map((image) => image.name)), [
    ClauseCondition.Includes,
    ClauseCondition.Is,
  ]);

  return (
    <OmnibarWrapperContainer>
      <Omnibar clauses={clauses} dropdownOptions={dropdownOptions} setClauses={setClauses} placeholder="Enter a filter..." />
    </OmnibarWrapperContainer>
  );
}
