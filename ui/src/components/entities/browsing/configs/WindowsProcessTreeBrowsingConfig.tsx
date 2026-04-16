import React from 'react';
import { Link } from 'react-router-dom';
import { Row } from 'react-bootstrap';

// project imports
import { EntityBrowseConfig } from './config';
import {
  BrowsingCard,
  BrowsingContents,
  EntityGroups,
  EntityName,
  EntitySecondary,
  EntitySubmitters,
  LinkFields,
} from '@entities/browsing/shared';
import CondensedEntityTags from '@components/tags/condensed/CondensedEntityTags';
import { Filters } from '@models/search';
import { listEntities } from '@thorpi/entities';
import { Entities } from '@models/entities/entities';
import { WindowsProcessTree } from '@models/entities/process_trees';
import { getDetailsBasePathByEntity } from '@components/entities/details/EntityDetailsRoutes';

// get files using filters and and an optional cursor
const getWindowsProcessTrees = async (filters: Filters, cursor: string | null, errorHandler: (error: string) => void) => {
  // reset cursor when filters have changed, caller must know this
  // get files list from API
  const listFilters = structuredClone(filters);
  listFilters.kinds = [Entities.WindowsProcessTree];
  const { entityList, entityCursor } = await listEntities(listFilters, errorHandler, true, cursor);
  return {
    entitiesList: entityList as WindowsProcessTree[],
    entitiesCursor: entityCursor,
  };
};

const WindowsProcessTreeListHeader = () => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Row>
          <EntityName>Name</EntityName>
          <EntitySecondary>Tools</EntitySecondary>
          <EntityGroups>Group(s)</EntityGroups>
          <EntitySubmitters>Submitter(s)</EntitySubmitters>
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

interface WindowsProcessTreeItemProps {
  tree: WindowsProcessTree; // Process tree details
}

const ProcessTreeItem: React.FC<WindowsProcessTreeItemProps> = ({ tree }) => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link to={`${getDetailsBasePathByEntity(Entities.WindowsProcessTree)}/${tree.id}`} state={{ tree: tree }} className="no-decoration">
          <LinkFields>
            <EntityName>{tree.name}</EntityName>
            <EntitySecondary>
              <i>
                {tree.metadata.WindowsProcessTree.tools &&
                  (tree.metadata.WindowsProcessTree.tools.toString().length > 75
                    ? tree.metadata.WindowsProcessTree.tools.toString().replaceAll(',', ', ').substring(0, 75) + '...'
                    : tree.metadata.WindowsProcessTree.tools.toString().replaceAll(',', ', '))}
              </i>
            </EntitySecondary>
            <EntityGroups>
              <i>
                {tree.groups &&
                  (tree.groups.toString().length > 75
                    ? tree.groups.toString().replaceAll(',', ', ').substring(0, 75) + '...'
                    : tree.groups.toString().replaceAll(',', ', '))}
              </i>
            </EntityGroups>
            <EntitySubmitters>
              {tree.submitter ? <i>{tree.submitter.length > 75 ? tree.submitter.substring(0, 75) + '...' : tree.submitter}</i> : null}
            </EntitySubmitters>
          </LinkFields>
        </Link>
        {tree.tags != undefined && <hr />}
        <Row>
          {tree.tags && Object.keys(tree.tags).length > 1 ? (
            <CondensedEntityTags resource={Entities.WindowsProcessTree} tags={tree.tags} />
          ) : null}
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

const WindowsProcessTreeBrowsingConfig: EntityBrowseConfig<Entities.WindowsProcessTree> = {
  docTitle: 'Windows Process · Thorium',
  title: 'Windows Process Trees',
  typeLabel: '',
  kind: Entities.WindowsProcessTree,
  creatable: true,
  entityHeaders: <WindowsProcessTreeListHeader />,
  renderEntity: (entity) => <ProcessTreeItem tree={entity} />,
  fetchEntities: getWindowsProcessTrees,
};

export default WindowsProcessTreeBrowsingConfig;
