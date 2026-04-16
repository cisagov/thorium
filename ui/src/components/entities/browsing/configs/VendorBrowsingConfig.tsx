import React from 'react';
import { Link } from 'react-router-dom';
import { Row } from 'react-bootstrap';
import { Country } from 'country-list';

// project imports
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
import { EntityBrowseConfig } from './config';
import { Vendor } from '@models/entities/vendors';
import { Entities } from '@models/entities/entities';
import { Filters } from '@models/search';
import { listEntities } from '@thorpi/entities';
import { getDetailsBasePathByEntity } from '@components/entities/details/EntityDetailsRoutes';

// get files using filters and and an optional cursor
const getVendors = async (filters: Filters, cursor: string | null, errorHandler: (error: string) => void) => {
  // reset cursor when filters have changed, caller must know this
  // get files list from API
  const listFilters = structuredClone(filters);
  listFilters.kinds = [Entities.Vendor];
  const { entityList, entityCursor } = await listEntities(listFilters, errorHandler, true, cursor);
  return {
    entitiesList: entityList as Vendor[],
    entitiesCursor: entityCursor,
  };
};

const VendorListHeaders = () => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Row>
          <EntityName>Name</EntityName>
          <EntitySecondary>Country</EntitySecondary>
          <EntityGroups>Group(s)</EntityGroups>
          <EntitySubmitters>Submitter(s)</EntitySubmitters>
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

interface VendorItemProps {
  vendor: Vendor; // Vendor details
}

const VendorItem: React.FC<VendorItemProps> = ({ vendor }) => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link to={`${getDetailsBasePathByEntity(Entities.Vendor)}/${vendor.id}`} state={{ vendor: vendor }} className="no-decoration">
          <LinkFields>
            <EntityName>{vendor.name}</EntityName>
            <EntitySecondary>
              {vendor.metadata.Vendor.countries != undefined
                ? vendor.metadata.Vendor.countries
                    .map((country: Country) => {
                      return country.name;
                    })
                    .join(', ')
                    .substring(0, 75)
                : ''}
            </EntitySecondary>
            <EntityGroups>
              <i>
                {vendor.groups &&
                  (vendor.groups.toString().length > 75
                    ? vendor.groups.toString().replaceAll(',', ', ').substring(0, 75) + '...'
                    : vendor.groups.toString().replaceAll(',', ', '))}
              </i>
            </EntityGroups>
            <EntitySubmitters>
              {vendor.submitter ? (
                <i>{vendor.submitter.length > 75 ? vendor.submitter.substring(0, 75) + '...' : vendor.submitter}</i>
              ) : null}
            </EntitySubmitters>
          </LinkFields>
        </Link>
        {vendor.tags != undefined && <hr />}
        <Row>
          {vendor.tags && Object.keys(vendor.tags).length > 1 ? (
            <CondensedEntityTags resource={Entities.Vendor} tags={vendor.tags} />
          ) : null}
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

const VendorsConfig: EntityBrowseConfig<Entities.Vendor> = {
  docTitle: 'Vendors · Thorium',
  title: 'Vendors',
  typeLabel: '',
  kind: Entities.Vendor,
  creatable: true,
  entityHeaders: <VendorListHeaders />,
  renderEntity: (entity) => <VendorItem vendor={entity} />,
  fetchEntities: getVendors,
};

export default VendorsConfig;
